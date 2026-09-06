# Where a blocked run's output goes.
#
# Blocking bounds the *input* side by construction: `readblock` hands back a view or a windowed read,
# so the eleven input rasters are never materialized whole. The output side needs this protocol to
# get the same bound, because a full-grid result is 108 bytes per point — 19 bands, `4 * 11 + 8 * 8`
# — independent of the block size. A 40000x40000 polar grid is around 170 GB of output, which no
# block size reduces.
#
# A sink is asked for a destination per block and told when that block is filled, so a streaming sink
# consumes each block and reuses one buffer per task. Peak output memory is then
# `ntasks * blocksize * 108 B` rather than `npoints * 108 B`.
#
# `InMemoryOutputs` is the default and is the old behavior exactly: one full-grid `PairGeometry`,
# blocks writing into views of it. That is what makes the existing tests the regression gate for this
# protocol costing nothing when it is not used.

"""
    AbstractOutputSink

Where a blocked run's per-block results go.

[`InMemoryOutputs`](@ref) collects them into one full-grid [`PairGeometry`](@ref) and is the default.
[`BlockCallback`](@ref) hands each block to a function. The `Rasters` extension adds
`GeoTIFFOutputs`, which writes the reference's files incrementally, so a grid whose result does not
fit in memory is written without ever holding it.

A sink is driven by five functions, in this order: [`prepare_sink`](@ref) once,
[`sink_taskstate`](@ref) once per task, then [`blockdest`](@ref) and [`commitblock!`](@ref) per
block, then [`finish_sink`](@ref) once every task has joined. Implement all five for a new sink.
"""
abstract type AbstractOutputSink end

"""
    SinkContext

What a sink is told about the run it is serving.

# Fields
- `grid`: the [`MapGrid`](@ref) being computed on.
- `coordinate`: the [`AbstractImageCoordinate`](@ref) of the pair, which decides the output band
  layout — see [`reference_files`](@ref).
- `window`: the grid indices the whole run covers.
- `blocksize`: the block size in grid points, the bound a per-task buffer is sized to.
- `nodata`: the policy the output sentinels come from.
- `bands`: the [`PairGeometry`](@ref) fields the run's inputs can produce, from
  [`supported_bands`](@ref). A sink creating storage per band uses this to create nothing for a band
  that will be uniformly nodata.
"""
struct SinkContext{G<:MapGrid,K<:AbstractImageCoordinate}
    grid::G
    coordinate::K
    window::CartesianIndices{2}
    blocksize::NTuple{2,Int}
    nodata::NoDataPolicy
    bands::Vector{Symbol}
end

"""
    prepare_sink(sink::AbstractOutputSink, ctx::SinkContext) -> prepared

Whatever `sink` needs for the run, built once before any block is computed.

The return value is passed to every other function of the protocol in place of `sink` itself. It is a
return value rather than a mutation of `sink` so that it is concretely typed: the block loop
specializes on it, and a field that is only populated here would be a `Union` in the loop that
computes every point.
"""
function prepare_sink end

"""
    sink_taskstate(prepared, ctx::SinkContext) -> state

Per-task state, built once inside each task and owned by it for the task's lifetime.

This is where a streaming sink allocates its block buffer, and where anything not safe to share
between tasks belongs. `state` is passed to [`blockdest`](@ref) and [`commitblock!`](@ref).
"""
function sink_taskstate end

"""
    blockdest(prepared, state, ctx::SinkContext, block::CartesianIndices{2}) -> PairGeometry

Where `block`'s geometry is to be written.

Sized to `block`, with every band at its sentinel — the per-point loop leaves a point it skips
untouched, so an unwritten point must already read as nodata. A streaming sink returns views of
`state`'s buffer refilled with sentinels, so nothing is allocated per block.
"""
function blockdest end

"""
    commitblock!(prepared, state, ctx::SinkContext, block, dest::PairGeometry) -> Nothing

`block` has been filled into `dest`; consume it.

Called from the task that computed the block, so an implementation touching state shared between
tasks — a file handle, an accumulating array — must synchronize. `dest` is reused for the task's next
block, so anything to be kept must be copied here.
"""
function commitblock! end

"""
    finish_sink(prepared) -> result

Called once after every task has joined; its return value is what
[`pairgeometry_blocked`](@ref) returns.

[`InMemoryOutputs`](@ref) returns the full-grid [`PairGeometry`](@ref). A sink writing files closes
them here and returns their paths.
"""
function finish_sink end

# A `PairGeometry` whose bands view the given region of `r`'s. Writing through it writes `r`.
#
# `window` and `geotransform` describe the region rather than `r`'s whole extent, so a block handed to
# a sink or a callback is self-describing: a consumer writing it out needs to know where it sits, and
# inheriting the parent's georeferencing would place every block at the window's origin.
function _block_view(r::PairGeometry, local_block::CartesianIndices{2},
                     window::CartesianIndices{2}, geotransform::NTuple{6,Float64})
    ints = ntuple(i -> view(getfield(r, INT_BANDS[i]), local_block), length(INT_BANDS))
    floats = ntuple(i -> view(getfield(r, FLOAT_BANDS[i]), local_block), length(FLOAT_BANDS))
    return PairGeometry(ints..., floats..., geotransform, r.crs, window, r.nodata, r.coordinate)
end

"""
    supported_bands(inputs, coordinate::AbstractImageCoordinate) -> Vector{Symbol}

The [`PairGeometry`](@ref) fields `inputs` can produce a value for.

Every other field is uniformly nodata, because the input its computation needs was not given. This
is the same dependency structure `GeometryInputs` documents and the per-point loop branches on:
slope gates the operator, the scale factors, the expected offset and the search extent; velocity and
search range each gate their own outputs within that; chip size and the stable-surface mask are
independent of slope. The radar path's third off2vel band needs slope and has no projected
counterpart.

A sink creating storage per band uses this to create none for a band that would be written and never
touched — matching [`write_geotiffs`](@ref), which writes no file for an all-sentinel output, and the
reference, which writes no file at all for an output its inputs did not support.

The two criteria differ in one case. This one reads input presence, so it reports the operator bands
supported wherever a slope raster is given; the data-driven one finds them all-sentinel if
`cross_check` fails at every point of the window, which happens where the surface is perpendicular to
the image plane throughout. A streaming sink therefore creates two files there that a
`write_geotiffs` of the same result would skip.

`inputs` is a [`GeometryInputs`](@ref) or an [`AbstractInputSource`](@ref).
"""
function supported_bands(inputs::GeometryInputs, coordinate::AbstractImageCoordinate)
    return _supported_bands(coordinate; slope = inputs.dhdx !== nothing,
                            vel = inputs.vx !== nothing, sr = inputs.srx !== nothing,
                            csmin = inputs.csminx !== nothing, csmax = inputs.csmaxx !== nothing,
                            ssm = inputs.ssm !== nothing)
end

# The predicate in one place, taking the same six booleans the per-point loop derives, so a new input
# source needs only to report which of its rasters are present.
#
# `off2vx_dr`/`off2vy_dr` are the radar path's third off2vel band; the projected path leaves them at
# their sentinel unconditionally.
function _supported_bands(coordinate::AbstractImageCoordinate; slope::Bool, vel::Bool, sr::Bool,
                          csmin::Bool, csmax::Bool, ssm::Bool)
    radar = _has_axis_bands(coordinate)
    keep = (
        :location_x => true, :location_y => true,
        :offset_x => slope && vel, :offset_y => slope && vel,
        :search_x => slope && sr, :search_y => slope && sr,
        :chip_min_x => csmin, :chip_min_y => csmin,
        :chip_max_x => csmax, :chip_max_y => csmax,
        :stable_surface => ssm,
        :off2vx_dx => slope, :off2vx_dy => slope, :off2vy_dx => slope, :off2vy_dy => slope,
        :off2vx_dr => slope && radar, :off2vy_dr => slope && radar,
        :scale_x => slope, :scale_y => slope,
    )
    return [name for (name, supported) in keep if supported]
end

# Whether the coordinate system has the third off2vel band, dispatched rather than tested with `isa`
# for the same reason `reference_files` is: the two paths' band layouts differ, and the difference
# belongs in one place per path.
_has_axis_bands(::ProjectedCoordinate) = false
_has_axis_bands(::RadarCoordinate) = true

"""
    InMemoryOutputs()

The default sink: one full-grid [`PairGeometry`](@ref), which the run returns.

Each block writes into views of it, so the result is the same object the unblocked
[`pairgeometry`](@ref) produces and no copy is made. Total output memory is 108 bytes per grid point
of the window regardless of block size, which is what a streaming sink exists to avoid — see
[`BlockCallback`](@ref) and the `Rasters` extension's `GeoTIFFOutputs`.
"""
struct InMemoryOutputs <: AbstractOutputSink end

# The full-grid result, wrapped so it dispatches as a prepared sink rather than as the
# `PairGeometry` it holds — which is also what a streaming sink's per-task buffer is.
struct PreparedInMemory{G<:PairGeometry}
    result::G
end

prepare_sink(::InMemoryOutputs, ctx::SinkContext) =
    PreparedInMemory(allocate_geometry(ctx.window, window_geotransform(ctx.grid, ctx.window),
                                       ctx.grid.crs, ctx.nodata, ctx.coordinate))

sink_taskstate(::PreparedInMemory, ::SinkContext) = nothing

# The result's arrays are indexed relative to the window's origin, so the block is shifted into that
# frame. Already prefilled with sentinels by `allocate_geometry`, and each block covers a disjoint
# region, so there is nothing to refill and nothing to synchronize.
function blockdest(p::PreparedInMemory, ::Nothing, ctx::SinkContext, block::CartesianIndices{2})
    local_block = _shift(block, ctx.window)
    return _block_view(p.result, local_block, block, window_geotransform(ctx.grid, block))
end

commitblock!(::PreparedInMemory, ::Nothing, ::SinkContext, ::CartesianIndices{2},
             ::PairGeometry) = nothing

finish_sink(p::PreparedInMemory) = p.result

# A grid-indexed block in the frame of arrays covering `window`.
@inline function _shift(block::CartesianIndices{2}, window::CartesianIndices{2})
    off = first(window).I .- 1
    return CartesianIndices((block.indices[1] .- off[1], block.indices[2] .- off[2]))
end

"""
    BlockCallback(f)

A sink calling `f(block, geometry)` as each block completes.

`block` is the grid indices covered and `geometry` a [`PairGeometry`](@ref) sized to it, carrying its
own window and geotransform. Nothing full-grid is allocated: peak output memory is one block per
task.

`f` is called concurrently from every task, so it must synchronize whatever it touches. `geometry`
views a buffer the task reuses for its next block, so `f` must consume it rather than retain it —
copy out anything to be kept.

The run returns `f`. What `f` accumulated into lives in whatever it closes over, which the call site
holds a reference to:

```julia
n = Threads.Atomic{Int}(0)
pairgeometry_blocked(grid, pair, source; transform = tf, ntasks = 8,
                     sink = BlockCallback((block, g) -> Threads.atomic_add!(n, nvalid(g))))
n[]     # valid points over the whole window
```
"""
struct BlockCallback{F} <: AbstractOutputSink
    f::F
end

# Nothing to build up front: the callback is the whole sink, and the buffers are per task.
prepare_sink(s::BlockCallback, ::SinkContext) = s

# One buffer per task, reused for every block it draws. Sized to a whole block even where the window
# is not a whole number of blocks — a short edge block views the top-left corner of it, so no
# reallocation is needed and the largest block a task can draw is already covered.
sink_taskstate(s::BlockCallback, ctx::SinkContext) =
    allocate_geometry(CartesianIndices(min.(ctx.blocksize, size(ctx.window))),
                      window_geotransform(ctx.grid, ctx.window), ctx.grid.crs, ctx.nodata,
                      ctx.coordinate)

blockdest(::BlockCallback, buffer::PairGeometry, ctx::SinkContext, block::CartesianIndices{2}) =
    _refilled_buffer(buffer, ctx, block)

function commitblock!(s::BlockCallback, ::PairGeometry, ::SinkContext, block::CartesianIndices{2},
                      dest::PairGeometry)
    s.f(block, dest)
    return nothing
end

finish_sink(s::BlockCallback) = s.f

# A view of `buffer` shaped like `block`, refilled with sentinels and carrying the block's own
# georeferencing.
#
# Refilled here rather than after a commit so the buffer is clean whatever the sink did with the
# previous block. The per-point loop leaves a skipped point untouched, so a value left over from the
# previous block would be read as that point's geometry.
function _refilled_buffer(buffer::PairGeometry, ctx::SinkContext, block::CartesianIndices{2})
    local_block = CartesianIndices(map(n -> Base.OneTo(n), size(block)))
    dest = _block_view(buffer, local_block, block, window_geotransform(ctx.grid, block))
    out = ctx.nodata.output
    for name in INT_BANDS
        fill!(getfield(dest, name), Int32(out))
    end
    for name in FLOAT_BANDS
        fill!(getfield(dest, name), out)
    end
    return dest
end
