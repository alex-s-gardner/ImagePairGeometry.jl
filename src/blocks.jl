# Blocked and threaded computation over the grid window.
#
# A grid point's geometry depends on nothing but that point, so a window computed in pieces gives
# the same answer as the same window computed whole — bit for bit, in any order, on any number of
# threads. That is what makes both of these free of fidelity cost, and it is asserted rather than
# assumed: `test/blocks.jl` compares every band against an unblocked run at a range of block sizes
# and thread counts.
#
# Blocking exists to bound memory. The inputs are eleven rasters over the window, and at ITS_LIVE's
# 120 m spacing a polar grid is tens of thousands of points on a side, so materializing them whole
# is tens of gigabytes where a block is a few megabytes. There is no halo: a point reads only its
# own DEM, slope, velocity, search-range, chip-size and mask values.
#
# Threading owns a PROJ context per task rather than per block, because building a `Transformation`
# on a fresh context costs about 1.7 ms — dominated by a cold `proj.db` cache — against roughly
# 50 ms of work for a 256×256 block. Per-task amortizes that to nothing; per-block would not.

"""
    block_ranges(window, blocksize) -> Vector{CartesianIndices{2}}

`window` partitioned into blocks of at most `blocksize`, in column-major order.

Blocks tile the window exactly: no overlap, no halo, and the union is the window. The final block
in each direction is short where the window does not divide evenly.
"""
function block_ranges(window::CartesianIndices{2}, blocksize::NTuple{2,Int})
    all(>(0), blocksize) || throw(ArgumentError("blocksize must be positive, got $blocksize"))
    xr, yr = window.indices
    out = CartesianIndices{2}[]
    for jstart in first(yr):blocksize[2]:last(yr), istart in first(xr):blocksize[1]:last(xr)
        push!(out, CartesianIndices((istart:min(istart + blocksize[1] - 1, last(xr)),
                                     jstart:min(jstart + blocksize[2] - 1, last(yr)))))
    end
    return out
end

"""
    DEFAULT_BLOCKSIZE

Block size in grid points when none is given: `(512, 512)`.

262144 points is a few megabytes of input per raster — small enough to bound memory on a
continental grid, large enough that per-block overhead (a PROJ pipeline lookup, a task claim) is
negligible against the per-point work.
"""
const DEFAULT_BLOCKSIZE = (512, 512)

"""
    AbstractInputSource

A source of the per-block input rasters.

Implement [`readblock`](@ref) for a new source. [`InMemoryInputs`](@ref) views arrays already held;
the `Rasters` extension adds a source that reads windows from disk, so a grid larger than memory
never has to be materialized.
"""
abstract type AbstractInputSource end

"""
    InMemoryInputs(inputs::GeometryInputs)

An input source viewing arrays already in memory, indexed relative to the full window.

Blocking such a source bounds the *working set* rather than total memory — the arrays are already
resident — which is what makes it the right thing to compare a blocked run against.
"""
struct InMemoryInputs{G<:GeometryInputs} <: AbstractInputSource
    inputs::G
    window::CartesianIndices{2}
end

"""
    readblock(src::AbstractInputSource, block::CartesianIndices{2}) -> GeometryInputs

The inputs covering `block`.

For [`InMemoryInputs`](@ref) this is a view, so no copy is made and a blocked run over resident
arrays allocates nothing per block.
"""
function readblock(src::InMemoryInputs, block::CartesianIndices{2})
    # `block` indexes the grid; the held arrays are indexed relative to the window's origin.
    off = first(src.window).I .- 1
    local_block = CartesianIndices((block.indices[1] .- off[1], block.indices[2] .- off[2]))
    g = src.inputs
    sub(a) = a === nothing ? nothing : view(a, local_block)
    return GeometryInputs(dem = sub(g.dem), dhdx = sub(g.dhdx), dhdy = sub(g.dhdy),
                          vx = sub(g.vx), vy = sub(g.vy), srx = sub(g.srx), sry = sub(g.sry),
                          csminx = sub(g.csminx), csminy = sub(g.csminy),
                          csmaxx = sub(g.csmaxx), csmaxy = sub(g.csmaxy), ssm = sub(g.ssm))
end

"""
    pairgeometry_blocked(grid, pair, source; transform, window = nothing,
                         blocksize = DEFAULT_BLOCKSIZE, ntasks = nothing,
                         params = GeometryParams(), nodata = nodata_from(nothing))
        -> PairGeometry

[`pairgeometry`](@ref) computed block by block, optionally across threads.

`source` is an [`AbstractInputSource`](@ref) — [`InMemoryInputs`](@ref), or the `Rasters`
extension's disk-backed source. `blocksize` is in grid points. `ntasks` defaults to
`min(nblocks, Threads.nthreads())`; pass `1` to run serially.

The result is bit-identical to the unblocked computation at any block size and any task count,
because each point depends only on its own inputs.

`transform` may be a [`TransformPair`](@ref), an [`AbstractCoordTransform`](@ref), or — for a
threaded run over a PROJ transform — a zero-argument function returning a fresh `TransformPair`.
That last form exists because `Proj.Transformation` wraps a `PJ*` on a context that is not
thread-safe: each task calls it once and owns the result for its lifetime. See
`ProjTransformFactory`, defined when `Proj` is loaded.
"""
function pairgeometry_blocked(grid::MapGrid, pair::CoregisteredPair, source::AbstractInputSource;
                              transform, window = nothing,
                              blocksize::NTuple{2,Int} = DEFAULT_BLOCKSIZE,
                              ntasks::Union{Nothing,Int} = nothing,
                              params::GeometryParams = GeometryParams(),
                              nodata::NoDataPolicy = nodata_from(nothing))
    coord = pair.coordinate
    # Validated before anything is built, so a bad argument is reported rather than surfacing as a
    # failure deeper in.
    ntasks === nothing || ntasks >= 1 ||
        throw(ArgumentError("ntasks must be at least 1, got $ntasks"))

    # Resolved at most once, and only where it is needed: deriving the window needs a transform, and
    # a serial run needs one for the whole loop, but a threaded run builds its own per task. Each
    # resolution of a factory creates a PROJ context and two pipelines — around 1.7 ms, dominated by
    # a cold `proj.db` cache — so a wasted one is worth avoiding.
    serial = ntasks == 1
    shared = (window === nothing || serial) ? _resolve_transform(transform) : nothing
    win = window === nothing ? grid_window(grid, footprint_bounds(shared, coord)) : window

    result = allocate_geometry(win, window_geotransform(grid, win), grid.crs, nodata, coord)
    blocks = block_ranges(win, blocksize)
    n = ntasks === nothing ? min(length(blocks), Threads.nthreads()) : ntasks

    if n == 1
        # One task means one owner, so whatever was already resolved is reused rather than rebuilt.
        tf = shared === nothing ? _resolve_transform(transform) : shared
        for b in blocks
            _run_block!(result, grid, coord, pair.dt, source, tf, params, nodata, b, win)
        end
        return result
    end

    # Blocks are claimed from a shared counter rather than partitioned up front, so a task that
    # draws cheap blocks (mostly out of bounds) moves on to more instead of finishing early.
    next = Threads.Atomic{Int}(1)
    tasks = map(1:min(n, length(blocks))) do _
        # The transform is built inside the task so each owns its own PROJ context for its whole
        # lifetime, and bound in a closure of its own so the local cannot be boxed and shared.
        Threads.@spawn begin
            tf = _resolve_transform(transform)
            while true
                k = Threads.atomic_add!(next, 1)
                k <= length(blocks) || break
                _run_block!(result, grid, coord, pair.dt, source, tf, params, nodata,
                            blocks[k], win)
            end
        end
    end
    foreach(wait, tasks)
    return result
end

# Split out so the block loop specializes on the concrete source and transform types, and so each
# task's writes land in a function whose locals cannot be captured across tasks.
#
# Coordinate-agnostic: it only forwards to `_fill_geometry!`, which dispatches. Blocking a radar run
# is the same operation for the same reason — every point is independent of every other.
function _run_block!(result::PairGeometry, grid::MapGrid, coord::AbstractImageCoordinate, dt::Float64,
                     source::AbstractInputSource, tf::TransformPair, params::GeometryParams,
                     nodata::NoDataPolicy, block::CartesianIndices{2}, win::CartesianIndices{2})
    inputs = readblock(source, block)
    off = first(win).I .- 1
    local_block = CartesianIndices((block.indices[1] .- off[1], block.indices[2] .- off[2]))
    # Each block writes a disjoint region of the result, so no synchronization is needed.
    view_of = _block_view(result, local_block)
    _fill_geometry!(view_of, grid, coord, dt, inputs, tf, params, nodata, block)
    return nothing
end

# A `PairGeometry` whose bands view the given region of `r`'s. Writing through it writes `r`.
function _block_view(r::PairGeometry, local_block::CartesianIndices{2})
    ints = ntuple(i -> view(getfield(r, INT_BANDS[i]), local_block), length(INT_BANDS))
    floats = ntuple(i -> view(getfield(r, FLOAT_BANDS[i]), local_block), length(FLOAT_BANDS))
    return PairGeometry(ints..., floats..., r.geotransform, r.crs, local_block, r.nodata,
                        r.coordinate)
end
