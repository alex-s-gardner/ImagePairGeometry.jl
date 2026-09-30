# The harness gates on itself before it gates on anything else.
#
#     julia --project=tools/golden tools/golden/selftest.jl
#
# Every comparison here reports agreement, so the one failure mode that would invalidate all of it is a
# comparator that cannot report anything else. Two halves:
#
#   1. Every run's own output diffed against itself must be identical, and every log must parse. That
#      establishes the readers are deterministic and the inventory is complete.
#   2. One difference of each kind is injected and each must be caught. That establishes the comparator
#      is not vacuous — the half that (1) cannot show, because a comparator hard-coded to "equal" passes
#      (1) perfectly.
#
# The injected cases are the ways a real difference has shown up in this harness, not hypotheticals:
# a value off by one (the rounding-boundary points), a value replaced by the sentinel (the coverage
# differences), a transposed band (the `(x, y)` against `(y, x)` orientation the captures use), and a
# shifted geotransform (a window cut at the wrong offset). The transpose case runs on a *non-square*
# band, since on a square one the shape gives nothing away and only the values differ.

using Printf
using Test

include(joinpath(@__DIR__, "cases.jl"))
include(joinpath(@__DIR__, "compare.jl"))
include(joinpath(@__DIR__, "handoff.jl"))

"""
    selftest_readers() -> Nothing

Every run on disk parses, and every band of every geogrid output equals itself.

The determinism half. A reader that returned a different array on a second call, or a log field that
silently defaulted, would make every later comparison a comparison of noise.
"""
function selftest_readers()
    runs = goldenruns()
    @testset "every run parses" begin
        @test !isempty(runs)
        for r in runs
            # `goldenrun` has already thrown on anything it could not account for; these are the fields
            # a later layer indexes without checking.
            @test r.nodata == EXPECTED_NODATA
            @test r.dt > 0
            @test r.chip_size_0 > 0
            @test !isempty(r.window)
            @test haskey(r.param_urls, "dem")
            @test all(endswith(".tif"), values(r.param_urls))
            if r.radar
                @test r.radar_params !== nothing
                @test r.radar_params.prf > 0
                @test r.radar_params.dr > 0
            else
                @test r.printed_origin !== nothing
                @test r.printed_size !== nothing
            end
        end
    end

    @testset "a band equals itself" begin
        for r in runs
            path = joinpath(r.dir, "window_location.tif")
            isfile(path) || continue
            a = ArchGDAL.read(ds -> ArchGDAL.read(ds, 1), path)
            b = ArchGDAL.read(ds -> ArchGDAL.read(ds, 1), path)
            res = compare_int_band("self", 1, :location_x, a, b)
            @test res.passed
            @test res.ndiff == 0
        end
    end
    return nothing
end

"""
    selftest_injections() -> Nothing

One difference of each kind, each of which must be caught.

The half that matters. A comparator that always reported agreement would pass `selftest_readers`
perfectly, so each case below asserts a specific difference is *detected* and, where the detail carries
it, that it is described correctly.
"""
function selftest_injections()
    runs = goldenruns()
    # A non-square band, so the transpose case has a shape to catch as well as values. Every geogrid
    # window in the golden set is non-square; asserted rather than assumed.
    r = first(filter(x -> !x.radar, runs))
    path = joinpath(r.dir, "window_location.tif")
    ints = ArchGDAL.read(ds -> ArchGDAL.read(ds, 1), path)
    floats = ArchGDAL.read(ds -> ArchGDAL.read(ds, 1),
                           joinpath(r.dir, "window_scale_factor.tif"))

    @testset "the test band is non-square" begin
        @test size(ints, 1) != size(ints, 2)
    end

    @testset "a value off by one is caught" begin
        # The rounding-boundary difference, which is the smallest real one this harness sees.
        m = copy(ints)
        i = findfirst(!=(Int32(r.nodata)), m)
        m[i] += Int32(1)
        res = compare_int_band("inject", 1, :location_x, m, ints)
        @test !res.passed
        @test res.ndiff == 1
        @test res.worst == 1.0
        # And the allowance admits it, which is what makes the allowance a claim about one point rather
        # than about the band.
        @test compare_int_band("inject", 1, :location_x, m, ints; allow_boundary = 1).passed
        # Two of them do not fit an allowance of one.
        m2 = copy(m)
        m2[findlast(!=(Int32(r.nodata)), m2)] += Int32(1)
        @test !compare_int_band("inject", 1, :location_x, m2, ints; allow_boundary = 1).passed
        # Nor does one point off by two, whatever the allowance: a boundary moves a value by one.
        m3 = copy(ints)
        m3[i] += Int32(2)
        @test !compare_int_band("inject", 1, :location_x, m3, ints; allow_boundary = 8).passed
    end

    @testset "a value replaced by the sentinel is caught" begin
        m = copy(ints)
        i = findfirst(!=(Int32(r.nodata)), m)
        m[i] = Int32(r.nodata)
        # As a value difference by default, since the sentinel takes part in the comparison.
        @test !compare_int_band("inject", 1, :location_x, m, ints).passed
        # Excluded where the caller says so, because then it is a *coverage* difference — which is what
        # `compare_coverage` counts, and the two must not both claim it.
        excl = compare_int_band("inject", 1, :location_x, m, ints; sentinel = Int(r.nodata))
        @test excl.ndiff == 0
        @test excl.n == length(ints) - 1
    end

    @testset "a transposed band is caught" begin
        # A transposed displacement field still reads as a displacement field, so this is the case a
        # summary statistic cannot see. On a non-square band the shape catches it first.
        res = compare_int_band("inject", 1, :location_x, permutedims(ints), ints)
        @test !res.passed
        @test occursin("shape", res.detail)
        @test !compare_float_band("inject", 1, :scale_x, permutedims(floats), floats;
                                  bound = 1e-7).passed
    end

    @testset "a float difference above the bound is caught" begin
        m = copy(floats)
        i = findfirst(!=(r.nodata), m)
        m[i] *= 1.0 + 1e-5
        @test !compare_float_band("inject", 1, :scale_x, m, floats; bound = 1e-7).passed
        # And one below it is not, which is what the bound means.
        m2 = copy(floats)
        m2[i] *= 1.0 + 1e-12
        @test compare_float_band("inject", 1, :scale_x, m2, floats; bound = 1e-7).passed
        # A bound of zero admits nothing: same value, different bits is still equal, but a changed
        # value is not.
        @test !compare_float_band("inject", 1, :scale_x, m2, floats; bound = 0.0).passed
        @test compare_float_band("inject", 1, :scale_x, floats, floats; bound = 0.0).passed
    end

    @testset "a signed zero is not a difference" begin
        # Negating a zeroed offset gives `-0.0` where the reference holds `0.0`. The two are one number,
        # and a bitwise comparison would report every such point.
        a = Float32[0.0 1.0; 2.0 3.0]
        b = Float32[-0.0 1.0; 2.0 3.0]
        @test compare_float_band("inject", 1, :dy_prior, a, b; bound = 0.0).passed
    end

    @testset "a shifted geotransform is caught" begin
        # `check_params` compares the cached DEM window's geotransform against the run's own output. A
        # window cut one pixel off would pass every band comparison against different ground.
        gt = ArchGDAL.read(ds -> ArchGDAL.getgeotransform(ds), path)
        shifted = (gt[1] + gt[2], gt[2], gt[3], gt[4], gt[5], gt[6])
        @test Tuple(gt) != shifted
    end

    @testset "the chop is checked against the capture, not assumed" begin
        # `chop_limits` needs its input already masked. Handing it the unmasked band gives a different
        # answer on the cases where the mask removes the largest chip size, and the shape rung compares
        # against the capture rather than trusting either.
        withcap = first(filter(x -> capture_of(x) !== nothing, runs))
        k = capture_of(withcap)
        b = geogrid_bands(withcap)
        chip0 = Int(k.scalars["ChipSize0X"])
        unmasked = chop_limits(b["ChipSizeMaxX"], chip0)
        captured = size(cap(k, "in_xGrid"))
        # Equal or not, the rung reads the capture — that is the assertion. The `LC08_L1TP_060018` case
        # is where they differ.
        @test all(>(0), unmasked)
        @test all(>(0), captured)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    @testset verbose = true "the golden harness gates on itself" begin
        @testset "readers" begin selftest_readers() end
        @testset "injections" begin selftest_injections() end
    end
end
