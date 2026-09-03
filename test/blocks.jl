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
                         _resolve_transform
using Proj
using Test

const ProjExt = Base.get_extension(ImagePairGeometry, :ImagePairGeometryProjExt)
using .ProjExt: ProjTransformFactory

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
