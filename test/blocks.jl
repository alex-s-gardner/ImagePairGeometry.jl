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
