# The TOPS azimuth carrier, and its removal across an interpolation.
#
# This is the first thing on the radar path with no isce3 C++ reference: the deramp lives in the Python
# `s1reader` package rather than in `Resample.cpp`, so `reference/topsramp.json` is generated from that
# arithmetic and the agreement is to a tolerance rather than to the bit. `reference/gen_topsramp.py` records
# what the fixture is and how to regenerate it.
#
# Two kinds of case, and neither substitutes for the other. The fixture pins the *conventions* — the
# polynomial argument, the center line, the sign of the reference time — each of which a remove-then-reapply
# test would leave green while being wrong, because both directions share the error. The analytic cases pin
# the *behavior*: that removing the carrier before interpolating is what stops the ramp aliasing, which is
# the claim the feature makes and which the fixture says nothing about.

using ImagePairGeometry
using ImagePairGeometry: TOPSCarrier, ResampledSLC, SincKernel, slant_range, sinc_interpolate,
                         SINC_HALF
using JSON3
using Random
using Test

const TOPS_REF = JSON3.read(read(joinpath(@__DIR__, "reference", "topsramp.json"), String))

# The fixture's own burst, rebuilt as a carrier. Everything comes from the file so the two cannot drift.
function _fixture_carrier()
    r0 = Float64(TOPS_REF.poly_reference_range)
    fm = Tuple(Float64.(TOPS_REF.fm_rate_coeffs))
    dc = Tuple(Float64.(TOPS_REF.doppler_coeffs))
    ka(r) = fm[1] + fm[2] * (r - r0) + fm[3] * (r - r0)^2
    fdc(r) = dc[1] + dc[2] * (r - r0) + dc[3] * (r - r0)^2
    starting_range = Float64(TOPS_REF.starting_range)
    return TOPSCarrier(;
        azimuth_fm_rate = ka,
        doppler_centroid = fdc,
        ks = Float64(TOPS_REF.ks),
        eta_ref_near = fdc(starting_range) / ka(starting_range),
        starting_range,
        range_pixel_spacing = Float64(TOPS_REF.range_pixel_spacing),
        azimuth_time_interval = Float64(TOPS_REF.azimuth_time_interval),
        # The reference's `n_lines // 2` is a zero-based line, so the one-based center is one past it.
        center_line = Float64(TOPS_REF.lines_per_burst ÷ 2 + 1),
    )
end

@testset "the carrier matches s1reader" begin
    c = _fixture_carrier()
    lines = Int.(TOPS_REF.lines)
    samples = Int.(TOPS_REF.samples)

    worst = 0.0
    for (i, line) in enumerate(lines), (j, sample) in enumerate(samples)
        want = Float64(TOPS_REF.carrier[i][j])
        got = c(line, sample)
        # The phase runs to some thousands of radians at the burst edges, so the comparison is relative
        # except at the center line where the answer is exactly zero.
        if want == 0.0
            @test got == 0.0
        else
            @test got ≈ want rtol = 1e-12
            worst = max(worst, abs(got - want) / abs(want))
        end
    end
    @info "TOPS carrier vs s1reader" points = length(lines) * length(samples) worst
end

@testset "the carrier vanishes at the burst center and grows quadratically" begin
    # `eta` is measured from the center line, so the phase there is zero by construction — and the same
    # distance either side gives the same phase, since it enters squared. A center line off by one would
    # break the symmetry rather than the magnitude, which is why this is asserted rather than eyeballed.
    c = _fixture_carrier()
    center = Int(TOPS_REF.lines_per_burst ÷ 2 + 1)

    # At the center, only the range-dependent `eta_ref` remains — zero at the near range, where it cancels.
    @test c(center, 1) == 0.0

    # Quadratic in the distance from the center: doubling it quadruples the phase. Taken at the near range
    # so `eta_ref` is zero and the relation is exact rather than approximate.
    p1 = c(center + 100, 1)
    p2 = c(center + 200, 1)
    @test p2 / p1 ≈ 4.0 rtol = 1e-12
    # And symmetric about the center.
    @test c(center - 100, 1) ≈ p1 rtol = 1e-12

    # Steepest at the burst edges, which is where interpolating without removing it does the most damage.
    @test abs(c(1, 1)) > abs(c(center + 100, 1))
    @test abs(c(Int(TOPS_REF.lines_per_burst), 1)) > abs(c(center + 100, 1))
end

@testset "the carrier's range mapping" begin
    c = _fixture_carrier()
    # One-based: sample 1 is the starting range, not one spacing past it.
    @test slant_range(c, 1) == Float64(TOPS_REF.starting_range)
    @test slant_range(c, 2) == Float64(TOPS_REF.starting_range) +
                              Float64(TOPS_REF.range_pixel_spacing)
    # Fractional samples are what an interpolated position gives.
    @test slant_range(c, 1.5) ≈ Float64(TOPS_REF.starting_range) +
                                0.5 * Float64(TOPS_REF.range_pixel_spacing)
end

@testset "the carrier's arguments are checked" begin
    ok = (azimuth_fm_rate = r -> -2300.0, doppler_centroid = r -> -40.0,
          eta_ref_near = 0.0, starting_range = 8.0e5, range_pixel_spacing = 2.33,
          azimuth_time_interval = 2.0e-3, center_line = 751.0)
    @test TOPSCarrier(; ok..., ks = 7592.0) isa TOPSCarrier
    # Zero sweep is stripmap, which carries no ramp — a caller reaching this has the wrong acquisition.
    @test_throws "does not sweep" TOPSCarrier(; ok..., ks = 0.0)
    @test_throws "must be positive" TOPSCarrier(; ok..., ks = 7592.0, range_pixel_spacing = 0.0)
    @test_throws "must be positive" TOPSCarrier(; ok..., ks = 7592.0, azimuth_time_interval = -1.0)

    # A zero FM rate divides twice, and a singular `kt` divides once. Both are impossible in a real
    # product, so they mean a mis-parse — reported rather than turned into `NaN` samples downstream.
    zero_ka = TOPSCarrier(; ok..., ks = 7592.0, azimuth_fm_rate = r -> 0.0)
    @test_throws DomainError zero_ka(1, 1)
    singular = TOPSCarrier(; ok..., ks = 7592.0, azimuth_fm_rate = r -> 7592.0)
    @test_throws DomainError singular(1, 1)
end

@testset "removing the carrier and reapplying it is the identity" begin
    # A resample at zero offset with the same carrier the samples carry must return them unchanged. This is
    # what says the two halves are inverses: a sign error, or a phase evaluated at the output index instead
    # of the input, shows up here even though both directions use the same function.
    Random.seed!(11)
    n = 64
    c = _fixture_carrier()
    # A synthetic image already carrying the ramp, as a TOPS product's samples do.
    base = ComplexF32.(randn(n, n) .+ 1im .* randn(n, n))
    ramped = [base[i, j] * ComplexF32(cis(c(i, j))) for i in 1:n, j in 1:n]

    r = ResampledSLC(ramped, fill((0.0, 0.0), n, n); carrier = c)
    # Away from the edges, where the stencil fits.
    for i in (SINC_HALF + 2, n ÷ 2, n - SINC_HALF - 1), j in (SINC_HALF + 2, n ÷ 2, n - SINC_HALF - 1)
        # At a zero offset the interpolation is at an integer position, so it reproduces the sample itself
        # up to the kernel's own rounding — and the carrier cancels exactly.
        @test r[i, j] ≈ ramped[i, j] rtol = 1e-4
    end
end

@testset "removing the carrier is what stops the ramp aliasing" begin
    # The claim the feature makes: interpolating ramped samples *without* removing the phase is wrong, and
    # removing it recovers the truth. Both are measured against an analytic ground truth — a smooth signal
    # times a known ramp — so this is not a comparison of two approximations.
    #
    # A pure ramp is the case where the two differ most and where the answer is known exactly: interpolating
    # a constant-magnitude phasor must give a constant-magnitude phasor, and the deramped path does while
    # the ramped path loses magnitude to the aliasing.
    n = 64
    c = _fixture_carrier()
    ramp_only = ComplexF32[ComplexF32(cis(c(i, j))) for i in 1:n, j in 1:n]
    off = fill((0.4, 0.35), n, n)

    with_carrier = ResampledSLC(ramp_only, off; carrier = c)
    without = ResampledSLC(ramp_only, off)

    # The magnitude is 1 everywhere in the truth, so any loss is the aliasing.
    err_with = 0.0
    err_without = 0.0
    for i in (SINC_HALF + 2):(n - SINC_HALF - 1), j in (SINC_HALF + 2):(n - SINC_HALF - 1)
        err_with = max(err_with, abs(abs(with_carrier[i, j]) - 1.0))
        err_without = max(err_without, abs(abs(without[i, j]) - 1.0))
    end
    @info "ramp aliasing, worst magnitude error" deramped = err_with ramped = err_without

    # Removing the carrier keeps the magnitude; leaving it in does not, by orders of magnitude.
    @test err_with < 1.0e-3
    @test err_without > 100 * err_with
end

@testset "a doppler model and a carrier are not both accepted" begin
    # Two hooks for one quantity would both be removed from the same chip, and their sum is not what either
    # means. A caller needing both terms evaluates both in one carrier.
    S = ComplexF32.(randn(8, 8) .+ 1im .* randn(8, 8))
    off = fill((0.0, 0.0), 8, 8)
    c = _fixture_carrier()
    @test ResampledSLC(S, off; carrier = c) isa ResampledSLC
    @test_throws "two ways to say the same thing" ResampledSLC(
        S, off; carrier = c, doppler = (az, rg) -> 0.0)
end

@testset "a zero carrier is bitwise a missing one" begin
    # The no-carrier path does no phasor arithmetic at all, rather than multiplying by a computed `1 + 0im`.
    # So a carrier returning zero must give the identical `Float32` bits, not merely a close value — the
    # same standard the zero-Doppler case is held to, since both are what make the default path exact.
    Random.seed!(12)
    n = 48
    big = ComplexF32.(randn(n, n) .+ 1im .* randn(n, n))
    off = fill((1.25, -0.75), n, n)
    plain = ResampledSLC(big, off)
    zeroed = ResampledSLC(big, off; carrier = (line, samp) -> 0.0)
    @test plain[20, 20] === zeroed[20, 20]
    @test plain[31, 17] === zeroed[31, 17]

    # A constant carrier is also an identity, since only differences from the chip center are used. That is
    # the property letting a carrier return a large absolute phase without losing precision.
    constant = ResampledSLC(big, off; carrier = (line, samp) -> 1.0e6)
    @test constant[20, 20] ≈ plain[20, 20] rtol = 1e-5
end
