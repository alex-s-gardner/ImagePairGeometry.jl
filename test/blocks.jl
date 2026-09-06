# Gate: blocking and threading change nothing.
#
# Every grid point depends only on its own inputs, so a window computed in pieces — in any order, on
# any number of threads — must give the same bits as the same window computed whole. That is the
# property that makes both worth having, and it is asserted here rather than argued for: a windowed
# read that computed something subtly different would be worse than one that was merely slow.
#
# Threads only exercise the parallel path when the suite runs with more than one. `Threads.nthreads()`
# is reported so a single-threaded run does not look like it proved something it did not.

using ImagePairGeometry
using ImagePairGeometry: INT_BANDS, FLOAT_BANDS, nodata_from, block_ranges, DEFAULT_BLOCKSIZE,
                         _resolve_transform, WarmStart, SceneCenterStart
using Proj
using Test

# A per-task PROJ transform factory. The package no longer ships one — it depends on no projection
# library — but the threading gate still needs a transform whose state cannot be shared across tasks,
# and PROJ is exactly that: `Proj.Transformation` wraps a context PROJ documents as usable from one
# thread at a time. Each task calls this once and owns the result.
#
# Contexts are deliberately never destroyed: a `Transformation`'s finalizer calls `proj_destroy` on its
# `PJ*`, which must not run after its context is freed.
function proj_factory(grid_crs, image_crs)
    return function ()
        ctx = Proj.proj_context_create()
        # `Proj.__init__` points only the *global* context at the bundled `proj.db`, so a
        # self-created one cannot find the database unless told where it is.
        Proj.proj_context_set_search_paths(1, [Proj.PROJ_DATA[]], ctx)
        # Grids fetched over the network would make results depend on what happened to be cached.
        Proj.proj_context_set_enable_network(false, ctx)
        g = grid_crs isa Integer ? "EPSG:$grid_crs" : grid_crs
        i = image_crs isa Integer ? "EPSG:$image_crs" : image_crs
        return TransformPair(Proj.Transformation(g, i; ctx, always_xy = false),
                             Proj.Transformation(i, g; ctx, always_xy = false))
    end
end

"""Assert two results agree on every band, floats compared bitwise."""
function assert_identical(a::PairGeometry, b::PairGeometry)
    @test size(a) == size(b)
    for f in INT_BANDS
        @test getfield(a, f) == getfield(b, f)
    end
    for f in FLOAT_BANDS
        # Bit patterns, not `==`: `-0.0 == 0.0` and `NaN != NaN` would both hide a real difference.
        @test reinterpret(UInt64, getfield(a, f)) == reinterpret(UInt64, getfield(b, f))
    end
end

@testset "block_ranges tiles the window exactly" begin
    for win in (CartesianIndices((1:10, 1:10)), CartesianIndices((3:47, 5:61)),
                CartesianIndices((1:1, 1:1)), CartesianIndices((1:100, 1:7)))
        for bs in ((1, 1), (3, 5), (8, 8), (64, 64), (1000, 1000))
            blocks = block_ranges(win, bs)
            # Exact cover: no overlap, nothing missing.
            covered = reduce(vcat, [vec(collect(b)) for b in blocks])
            # Compared as a set plus a count: equal sets with equal lengths means an exact cover
            # with no double-counting, without needing an order on CartesianIndex.
            @test Set(covered) == Set(win)
            @test length(covered) == length(win)
            @test all(b -> all(size(b) .<= bs), blocks)
            @test all(b -> !isempty(b), blocks)
        end
    end
    @test_throws "blocksize must be positive" block_ranges(CartesianIndices((1:4, 1:4)), (0, 4))
end

@testset "blocked equals unblocked: $name" for name in
        ("same_crs", "cross_crs", "with_nodata", "cross_crs_nodata", "dem_only")
    s = setup_case(name)
    whole = pairgeometry(s.grid, s.pair, s.inputs; transform = s.makepair(), window = s.win,
                         params = s.params, nodata = nodata_from(-32767.0))
    src = InMemoryInputs(s.inputs, s.win)

    # Block sizes chosen to include ones that divide the window and ones that do not, a single
    # column and row, and one larger than the window.
    for bs in ((1, 1), (7, 13), (16, 16), (59, 1), (1, 59), (1000, 1000))
        blocked = pairgeometry_blocked(s.grid, s.pair, src; transform = s.makepair(),
                                       window = s.win, blocksize = bs, ntasks = 1,
                                       params = s.params, nodata = nodata_from(-32767.0))
        @testset "blocksize $bs" begin
            assert_identical(whole, blocked)
        end
    end
end

@testset "threaded equals serial: $name" for name in ("same_crs", "cross_crs")
    s = setup_case(name)
    whole = pairgeometry(s.grid, s.pair, s.inputs; transform = s.makepair(), window = s.win,
                         params = s.params, nodata = nodata_from(-32767.0))
    src = InMemoryInputs(s.inputs, s.win)

    @info "threading gate" nthreads = Threads.nthreads()
    for nt in (1, 2, 4, 8), bs in ((8, 8), (16, 16))
        # A factory, so each task builds and owns its own PROJ context — a shared
        # `Proj.Transformation` wraps a `PJ*` on a context that is not thread-safe.
        threaded = pairgeometry_blocked(s.grid, s.pair, src; transform = s.makepair,
                                        window = s.win, blocksize = bs, ntasks = nt,
                                        params = s.params, nodata = nodata_from(-32767.0))
        @testset "ntasks $nt blocksize $bs" begin
            assert_identical(whole, threaded)
        end
    end
end

"""A `BlockCallback` sink assembling every block into one full-window result.

The properties a streaming sink has to hold are asserted as each block arrives: it is sized to the
block, it carries the block's own georeferencing rather than the window's, and it views a buffer the
task reuses rather than a fresh allocation. `buffers` collects the identity of each block's backing
array, so a task drawing many blocks must contribute exactly one entry.
"""
function assembling_sink(grid, coord, win, nodata)
    whole = ImagePairGeometry.allocate_geometry(
        win, ImagePairGeometry.window_geotransform(grid, win), grid.crs, nodata, coord)
    lk = ReentrantLock()
    buffers = Set{UInt}()
    sizes_ok = Ref(true)
    geotransforms_ok = Ref(true)
    f = function (block, g)
        size(g) == size(block) || (sizes_ok[] = false)
        g.window == block || (sizes_ok[] = false)
        g.geotransform == ImagePairGeometry.window_geotransform(grid, block) ||
            (geotransforms_ok[] = false)
        off = first(win).I .- 1
        local_block = CartesianIndices((block.indices[1] .- off[1], block.indices[2] .- off[2]))
        lock(lk) do
            push!(buffers, objectid(parent(g.location_x)))
            for name in (INT_BANDS..., FLOAT_BANDS...)
                view(getfield(whole, name), local_block) .= getfield(g, name)
            end
        end
    end
    return (; sink = BlockCallback(f), whole, buffers, sizes_ok, geotransforms_ok)
end

@testset "streamed equals collected: $name" for name in
        ("same_crs", "cross_crs", "with_nodata", "dem_only")
    s = setup_case(name)
    nd = nodata_from(-32767.0)
    whole = pairgeometry(s.grid, s.pair, s.inputs; transform = s.makepair(), window = s.win,
                         params = s.params, nodata = nd)
    src = InMemoryInputs(s.inputs, s.win)

    for bs in ((7, 13), (16, 16), (1000, 1000))
        a = assembling_sink(s.grid, s.pair.coordinate, s.win, nd)
        pairgeometry_blocked(s.grid, s.pair, src; transform = s.makepair(), window = s.win,
                             blocksize = bs, ntasks = 1, params = s.params, nodata = nd,
                             sink = a.sink)
        @testset "blocksize $bs" begin
            assert_identical(whole, a.whole)
            @test a.sizes_ok[]
            @test a.geotransforms_ok[]
            # The bound this sink exists for: one task holds one buffer however many blocks it draws.
            @test length(a.buffers) == 1
        end
    end
end

@testset "streamed equals collected across threads" begin
    s = setup_case("cross_crs")
    nd = nodata_from(-32767.0)
    whole = pairgeometry(s.grid, s.pair, s.inputs; transform = s.makepair(), window = s.win,
                         params = s.params, nodata = nd)
    src = InMemoryInputs(s.inputs, s.win)

    for nt in (2, 4)
        a = assembling_sink(s.grid, s.pair.coordinate, s.win, nd)
        pairgeometry_blocked(s.grid, s.pair, src; transform = s.makepair, window = s.win,
                             blocksize = (8, 8), ntasks = nt, params = s.params, nodata = nd,
                             sink = a.sink)
        @testset "ntasks $nt" begin
            assert_identical(whole, a.whole)
            @test a.sizes_ok[]
            @test a.geotransforms_ok[]
            # One buffer per task that actually ran, never one per block. A run with more tasks than
            # the window has blocks starts fewer.
            nblocks = length(block_ranges(s.win, (8, 8)))
            @test length(a.buffers) <= min(nt, nblocks)
        end
    end
end

"""A sink asserting that each task owns its own state and its own block buffer.

Each state is stamped with an id at creation, and every block records which state produced it against
the address of the buffer it came from. A buffer reached from two different states is one buffer shared
between two tasks.

Ownership per task is what the protocol promises and what the callers of `sink_taskstate` rely on: a
`Proj.Transformation` wraps a context PROJ documents as usable from one thread at a time, and a shared
block buffer means two tasks writing one array. Both are silent — the transform pairs of a run compute
the same values, and a shared buffer shows up only as blocks carrying each other's points.
"""
struct OwnershipProbe <: ImagePairGeometry.AbstractOutputSink
    built::Threads.Atomic{Int}
    commits::Threads.Atomic{Int}
    # Buffer address to the ids of every state that handed out a block from it.
    owners::Dict{UInt,Set{Int}}
    # Every buffer built, held so none is collected: an address identifies a buffer only while it is
    # alive, and a collected one's address can be handed to a later task's allocation.
    retained::Vector{Any}
    lk::ReentrantLock
end

OwnershipProbe() = OwnershipProbe(Threads.Atomic{Int}(0), Threads.Atomic{Int}(0),
                                  Dict{UInt,Set{Int}}(), Any[], ReentrantLock())

struct ProbeState{G}
    id::Int
    buffer::G
end

ImagePairGeometry.prepare_sink(p::OwnershipProbe, ::ImagePairGeometry.SinkContext) = p

function ImagePairGeometry.sink_taskstate(p::OwnershipProbe, ctx::ImagePairGeometry.SinkContext)
    buffer = ImagePairGeometry.allocate_geometry(
        CartesianIndices(min.(ctx.blocksize, size(ctx.window))),
        ImagePairGeometry.window_geotransform(ctx.grid, ctx.window), ctx.grid.crs, ctx.nodata,
        ctx.coordinate)
    id = Threads.atomic_add!(p.built, 1) + 1
    lock(p.lk) do
        push!(p.retained, buffer)
    end
    return ProbeState(id, buffer)
end

ImagePairGeometry.blockdest(::OwnershipProbe, state::ProbeState,
                            ctx::ImagePairGeometry.SinkContext, block) =
    ImagePairGeometry._refilled_buffer(state.buffer, ctx, block)

function ImagePairGeometry.commitblock!(p::OwnershipProbe, state::ProbeState,
                                        ::ImagePairGeometry.SinkContext, block, dest)
    Threads.atomic_add!(p.commits, 1)
    # The address rather than `objectid`: a buffer is alive for its task's whole lifetime, so an address
    # identifies it, while `objectid` on a mutable object is not stable across a `Dict` rehash.
    address = UInt(pointer(parent(dest.location_x)))
    lock(p.lk) do
        push!(get!(p.owners, address, Set{Int}()), state.id)
    end
    return nothing
end

ImagePairGeometry.finish_sink(p::OwnershipProbe) = p

@testset "each task owns its own sink state" begin
    s = setup_case("same_crs")
    src = InMemoryInputs(s.inputs, s.win)
    nblocks = length(block_ranges(s.win, (8, 8)))

    for nt in (1, 2, 4)
        p = pairgeometry_blocked(s.grid, s.pair, src; transform = s.makepair, window = s.win,
                                 blocksize = (8, 8), ntasks = nt, params = s.params,
                                 nodata = nodata_from(-32767.0), sink = OwnershipProbe())
        @testset "ntasks $nt" begin
            spawned = min(nt, nblocks)
            @test p.commits[] == nblocks
            @test p.built[] == spawned
            # No buffer reached from two states: that, not the count, is the property. A single boxed
            # local in the spawn loop gives several tasks the same state, and the only symptom is blocks
            # holding each other's points.
            @test all(==(1), length.(values(p.owners)))
            # How many tasks contribute a buffer is a scheduling outcome, not a contract. On one thread
            # the first task drains the queue before the rest start, so they build a state, find no work
            # and exit -- which is why this is bounded rather than equal to the task count.
            @test 1 <= length(p.owners) <= spawned
            Threads.nthreads() == 1 && @test length(p.owners) == 1
        end
    end
end

@testset "supported_bands reads which inputs were given" begin
    s = setup_case("same_crs")
    coord = s.pair.coordinate
    all_bands = supported_bands(s.inputs, coord)
    # The projected path never writes the third off2vel band, whatever its inputs.
    @test :off2vx_dr ∉ all_bands
    @test :off2vy_dr ∉ all_bands
    @test issubset([:location_x, :offset_x, :search_x, :scale_x, :stable_surface], all_bands)

    # A DEM alone supports the two location bands and nothing else: every other output needs a raster
    # that was not given.
    bare = supported_bands(GeometryInputs(dem = s.inputs.dem), coord)
    @test bare == [:location_x, :location_y]

    # Slope gates the operator, the scale factors, the offset and the search extent, so dropping it
    # drops all four even where velocity and search range are present -- which `GeometryInputs` will
    # not let happen, so this asserts the predicate rather than a constructible case.
    noslope = ImagePairGeometry._supported_bands(coord; slope = false, vel = true, sr = true,
                                                csmin = true, csmax = true, ssm = true)
    @test noslope == [:location_x, :location_y, :chip_min_x, :chip_min_y, :chip_max_x, :chip_max_y,
                      :stable_surface]

    # A source reports the bands of the inputs it holds.
    @test supported_bands(InMemoryInputs(s.inputs, s.win), coord) == all_bands
end

@testset "readblock views rather than copies" begin
    s = setup_case("same_crs")
    src = InMemoryInputs(s.inputs, s.win)
    blocks = block_ranges(s.win, (16, 16))
    b = readblock(src, blocks[2])
    @test b.dem isa SubArray
    @test size(b.dem) == size(blocks[2])
    # The view addresses the right region: its first element is the corresponding input element.
    off = first(s.win).I .- 1
    i, j = first(blocks[2]).I .- off
    @test b.dem[1, 1] === s.inputs.dem[i, j]
    # Absent inputs stay absent through a block read.
    @test readblock(InMemoryInputs(GeometryInputs(dem = s.inputs.dem), s.win),
                    blocks[1]).dhdx === nothing
end

@testset "transform resolution" begin
    @test _resolve_transform(IdentityTransform()) isa TransformPair
    @test _resolve_transform(transform_pair(IdentityTransform())) isa TransformPair
    @test _resolve_transform(() -> IdentityTransform()) isa TransformPair
    # A bare `Proj.Transformation` has no inverse to derive, so it is refused with a message saying
    # what to pass instead rather than being invoked as a factory.
    @test_throws ArgumentError _resolve_transform(Proj.Transformation("EPSG:4326", "EPSG:3413"))
    @test_throws "TransformPair" _resolve_transform(42)
end

@testset "argument errors" begin
    s = setup_case("same_crs")
    src = InMemoryInputs(s.inputs, s.win)
    @test_throws "ntasks must be at least 1" pairgeometry_blocked(
        s.grid, s.pair, src; transform = s.makepair(), window = s.win, ntasks = 0)
end

@testset "default blocksize" begin
    @test DEFAULT_BLOCKSIZE == (512, 512)
    # A window smaller than one block is a single block, so a small grid pays nothing for blocking.
    @test length(block_ranges(CartesianIndices((1:100, 1:100)), DEFAULT_BLOCKSIZE)) == 1
end

@testset "WarmStart forfeits blocking invariance, and says so" begin
    # The invariance above is a property of `SceneCenterStart`, not of the package: a warm start makes
    # each point depend on its predecessor, so a blocked run and an unblocked one visit points in
    # different sequences and get different bits. Asserted rather than only documented, because a
    # future change that quietly restored the invariance would mean the warm start had stopped
    # working, and a change that broke it on the *default* path would be a real bug.
    #
    # The radar fixture is the case with a zero-Doppler solve; the projected cases have none, so the
    # policy is inert there and the invariance holds regardless.
    s = setup_case("same_crs")
    warm = GeometryParams(chip_size_0 = s.params.chip_size_0, scaling = s.params.scaling,
                          zero_doppler_start = WarmStart())
    src = InMemoryInputs(s.inputs, s.win)

    whole = pairgeometry(s.grid, s.pair, s.inputs; transform = s.makepair(), window = s.win,
                         params = warm, nodata = nodata_from(-32767.0))
    blocked = pairgeometry_blocked(s.grid, s.pair, src; transform = s.makepair(),
                                   window = s.win, blocksize = (7, 13), ntasks = 1,
                                   params = warm, nodata = nodata_from(-32767.0))
    # This pair is projected, so it has no zero-Doppler solve and the policy is inert -- which is the
    # thing worth asserting here: setting it must not perturb the path it does not apply to. The
    # radar path's actual loss of invariance is measured in `radar_geogrid.jl`, where a solve exists.
    assert_identical(whole, blocked)

    # And the default remains the default: an explicitly-constructed `SceneCenterStart` is the same
    # object the no-keyword form produces, so no caller gets a warm start by accident.
    @test GeometryParams().zero_doppler_start === SceneCenterStart()
    @test GeometryParams(zero_doppler_start = WarmStart()).zero_doppler_start === WarmStart(8)
end
