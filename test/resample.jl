# The sinc interpolation kernel, against isce3.
#
# Bitwise is achievable here and is asserted: the coefficients are `cos` and `sin` of exactly
# representable arguments followed by a normalization, and the interpolation is a fixed sum of products.
# Nothing in this file needs a tolerance, which is why any that appears is a claim about a deliberate
# divergence rather than about precision.
#
# Two transcribed quirks carry most of the risk, and each has a case aimed at it: the kernel table is
# indexed by *truncation* rather than rounding, and the tap sum runs *downward* through the chip. Either
# one wrong gives a plausible image that is subtly misplaced.

using ImagePairGeometry
using ImagePairGeometry: SincKernel, sinc_interpolate, kernel_length, decimation,
                         SINC_LEN, SINC_HALF, SINC_ONE, SINC_SUB, ResampledSLC,
                         RadarCoordinate, Orbit, LookRight, incidence_angle
using JSON3
using Random
using StaticArrays: SVector
using Test

# A reference coordinate for the Doppler cases: only its range and azimuth scales are read, to turn an
# output pixel into an azimuth time and a slant range.
function _resamp_coord()
    ts = collect(0.0:10.0:400.0)
    R, incl = 7.0e6, deg2rad(98.0)
    om = sqrt(3.986004418e14 / R^3)
    orb = Orbit(; time = ts,
                position = [SVector(R*cos(om*t), R*sin(om*t)*cos(incl), R*sin(om*t)*sin(incl))
                            for t in ts],
                velocity = [SVector(-R*om*sin(om*t), R*om*cos(om*t)*cos(incl),
                                    R*om*cos(om*t)*sin(incl)) for t in ts])
    kw = (; orbit = orb, starting_range = 8.0e5, dr = 2.3295621147, sensing_start = 100.0,
          prf = 486.4863103, nsamples = 500, nlines = 400, look_side = LookRight,
          wavelength = 0.05546576)
    return RadarCoordinate(; kw..., incidence_angle = incidence_angle(; kw...))
end

const SINC_FX = JSON3.read(read(joinpath(@__DIR__, "reference", "sinc.json"), String))

@testset "fixture provenance" begin
    @test SINC_FX.versions.isce3 == "0.25.12"
    @test SINC_FX.kernel.length == SINC_LEN
    @test SINC_FX.kernel.decimation == SINC_SUB
end

@testset "the constants are isce3's" begin
    # `isce3/core/Constants.h:32-35`. `SINC_ONE` is 9 rather than 8 — the chip is one sample wider than
    # the kernel — and a caller sizing a halo from the wrong one reads a sample short.
    @test (SINC_LEN, SINC_HALF, SINC_ONE, SINC_SUB) == (8, 4, 9, 8192)
    @test SINC_HALF == SINC_LEN ÷ 2
    @test SINC_ONE == SINC_LEN + 1
end

@testset "the kernel table is bitwise" begin
    k = SincKernel()
    @test size(k.weights) == (SINC_SUB, SINC_LEN)
    @test kernel_length(k) == SINC_LEN
    @test decimation(k) == SINC_SUB

    # Every tap of seven rows spanning the table, including both ends where the Hamming window is at its
    # extremes. Compared on bit patterns rather than with `==`, since that would let `-0.0` pass for
    # `0.0` — and the first coefficient of row 0 is in fact `-0.0`.
    for (rowstr, taps) in pairs(SINC_FX.kernel.rows)
        r = parse(Int, String(rowstr)) + 1
        for (j, hexstr) in pairs(taps)
            want = parse(Float64, String(hexstr))
            @test reinterpret(UInt64, k.weights[r, j]) == reinterpret(UInt64, want)
        end
    end

    # Each position's taps sum to one, which is what makes a constant signal interpolate to itself.
    #
    # Not bitwise, and the reason is the *test* rather than the kernel: the taps themselves agree to the
    # bit above, but NumPy's pairwise summation and Julia's differ in association, so the two sums can
    # land an ULP apart. A tolerance here is a statement about how the check is performed; the exactness
    # claim is the tap-by-tap comparison.
    for (rowstr, hexstr) in pairs(SINC_FX.kernel.rowsums)
        r = parse(Int, String(rowstr)) + 1
        want = parse(Float64, String(hexstr))
        @test sum(k.weights[r, :]) ≈ want atol = 4eps(Float64)
        @test want ≈ 1.0 atol = 4eps(Float64)
    end
end

@testset "interpolation is bitwise against isce3" begin
    k = SincKernel()
    unhex(rows) = reduce(vcat, [permutedims([parse(Float64, String(v)) for v in row]) for row in rows])
    blk = ComplexF32.(unhex(SINC_FX.block_real) .+ 1im .* unhex(SINC_FX.block_imag))
    @test size(blk) == (32, 32)

    worst = 0.0
    nexact = 0
    for (q, ref) in zip(SINC_FX.queries, SINC_FX.isce3)
        want = ComplexF32(parse(Float64, String(ref[1])), parse(Float64, String(ref[2])))
        # isce3's indices are zero-based; this package's are one-based.
        got = sinc_interpolate(k, blk, Float64(q[1]) + 1, Float64(q[2]) + 1)
        # Bit patterns, so a signed zero cannot pass for an unsigned one.
        exact = reinterpret(UInt32, real(got)) == reinterpret(UInt32, real(want)) &&
                reinterpret(UInt32, imag(got)) == reinterpret(UInt32, imag(want))
        @test exact
        exact && (nexact += 1)
        worst = max(worst, Float64(abs(got - want)))
    end
    @test nexact == length(SINC_FX.queries)
    @info "sinc interpolation vs isce3" queries = length(SINC_FX.queries) exact = nexact worst = worst
end

@testset "the accumulator type is what makes it bitwise" begin
    # isce3 accumulates a 64-tap sum in `complex<float>`, casting each weight to it. The default
    # reproduces that. `ComplexF64` accumulation is *more accurate* and therefore not bitwise, which is
    # worth pinning: a future change that silently widened the accumulator would break the agreement
    # above, and this says which direction that is.
    k = SincKernel()
    unhex(rows) = reduce(vcat, [permutedims([parse(Float64, String(v)) for v in row]) for row in rows])
    blk = ComplexF32.(unhex(SINC_FX.block_real) .+ 1im .* unhex(SINC_FX.block_imag))

    ndiff = 0
    for q in SINC_FX.queries
        x, y = Float64(q[1]) + 1, Float64(q[2]) + 1
        f32 = sinc_interpolate(k, blk, x, y)
        f64 = sinc_interpolate(k, blk, x, y; accumulate = ComplexF64)
        # Same value to within a Float32 ULP either way, so nothing downstream can tell them apart.
        @test abs(f32 - f64) < 4 * eps(Float32)
        f32 !== ComplexF32(f64) && (ndiff += 1)
    end
    # At least one position must actually differ, or this testset is asserting nothing.
    @test ndiff > 0
end

@testset "the stencil runs downward and is bounds-checked" begin
    k = SincKernel()
    # A chip whose value is its column index: the interpolant of a linear ramp is that ramp, which pins
    # the *direction* of the tap sum. A kernel applied in the wrong direction reflects the stencil about
    # its centre and returns the value from the other side of the query — right for a symmetric kernel at
    # an exact half, wrong everywhere else, which is why the queries below are not at halves.
    #
    # The tolerance is 2e-4 rather than machine precision because a *windowed, truncated* sinc does not
    # reproduce a ramp exactly. Two effects, and neither is slack in the test: the sub-pixel position is
    # quantized to 1/8192 ≈ 1.2e-4 of a pixel by the truncating table index, and an 8-tap Hamming-windowed
    # kernel has a finite passband, so even at a position the table holds exactly the recovered ramp is
    # about 1e-6 out. Both are properties of the reference kernel, reproduced rather than corrected.
    #
    # A direction error, by contrast, misplaces the answer by whole samples — so this catches one with
    # three orders of magnitude to spare, which is what it is for.
    n = 24
    ramp = Float64[j for i in 1:n, j in 1:n]
    @test sinc_interpolate(k, ramp, 12.3, 12.0; accumulate = Float64) ≈ 12.3 atol = 2e-4

    # And along the other axis, so a transposed read is caught too.
    ramp_y = Float64[i for i in 1:n, j in 1:n]
    @test sinc_interpolate(k, ramp_y, 12.0, 9.7; accumulate = Float64) ≈ 9.7 atol = 2e-4

    # At a position the table represents exactly — a multiple of 1/8192 — the quantization drops out and
    # only the kernel's own passband error remains, two orders smaller. That separates the two effects
    # rather than lumping them into one tolerance.
    exact = 12.0 + 2048 / SINC_SUB
    @test sinc_interpolate(k, ramp, exact, 12.0; accumulate = Float64) ≈ exact atol = 1e-5

    # A constant interpolates to itself anywhere, since the taps sum to one.
    ones_ = ones(Float64, n, n)
    for (x, y) in ((12.0, 12.0), (12.5, 13.25), (15.75, 11.1))
        @test sinc_interpolate(k, ones_, x, y; accumulate = Float64) ≈ 1.0 atol = 1e-12
    end

    # Outside the stencil's reach the result is zero, as isce3 returns — not a clamped read, which would
    # invent an edge sample. A resampler turns this into its fill value; that decision is the caller's.
    for (x, y) in ((1.0, 12.0), (12.0, 1.0), (Float64(n), 12.0), (12.0, Float64(n)),
                   (-5.0, 12.0), (12.0, 1e6))
        @test sinc_interpolate(k, ones_, x, y; accumulate = Float64) == 0.0
    end
    # The first position whose whole stencil fits is not zero, so the bound above is tight rather than
    # rejecting everything.
    @test sinc_interpolate(k, ones_, Float64(SINC_HALF), Float64(SINC_HALF);
                           accumulate = Float64) ≈ 1.0 atol = 1e-12
end

@testset "the table index truncates" begin
    # `int(frac * 8192)` clamped, not a rounding. So the interpolant is piecewise constant in the
    # sub-pixel coordinate at a step of 1/8192, and two positions inside one step give the *identical*
    # answer — which a rounding index, or a continuous kernel, would not.
    k = SincKernel()
    n = 24
    chip = Float64[sin(0.3i) * cos(0.2j) for i in 1:n, j in 1:n]

    step = 1 / SINC_SUB
    base = 12.0 + 0.25
    a = sinc_interpolate(k, chip, base + 0.1 * step, 12.0; accumulate = Float64)
    b = sinc_interpolate(k, chip, base + 0.9 * step, 12.0; accumulate = Float64)
    @test a === b

    # Across a step boundary it does change, so the quantization is at the stated scale and not coarser.
    c = sinc_interpolate(k, chip, base + 1.5 * step, 12.0; accumulate = Float64)
    @test c !== a

    # A weighted kernel is not symmetric about its centre, so the two extreme table rows differ — which
    # is the check that the Hamming window is present at all.
    @test k.weights[1, :] != k.weights[end, :]
end

@testset "the kernel arguments are checked" begin
    @test_throws "kernel_length must be positive" SincKernel(kernel_length = 0)
    @test_throws "decimation must be positive" SincKernel(decimation = -1)

    # An unweighted kernel is a plain sinc: still normalized, but without the Hamming taper, so its
    # coefficients differ. Available because `_sinc_coef` takes the flag, though the reference never
    # sets it.
    plain = SincKernel(weighted = false)
    @test size(plain.weights) == (SINC_SUB, SINC_LEN)
    @test all(≈(1.0; atol = 1e-12), sum(plain.weights; dims = 2))
    @test plain.weights[4096, :] != SincKernel().weights[4096, :]

    # A smaller kernel for a cheaper test: the shape follows the arguments.
    small = SincKernel(kernel_length = 4, decimation = 16)
    @test size(small.weights) == (16, 4)
    @test kernel_length(small) == 4 && decimation(small) == 16

    # `sinc_interpolate` takes any of them — the caller supplies the chip, so its size is the caller's.
    chip = ComplexF32.(randn(5, 5) .+ 1im .* randn(5, 5))
    @test sinc_interpolate(small, chip, 3.25, 3.5) isa ComplexF32

    # `ResampledSLC` does not. Its chip is `SINC_ONE` on a side as a compile-time constant, which is what
    # keeps it off the heap, so another tap count would read a stencil of the wrong extent — and the sizes
    # would still agree with each other, making it silent. Refused with the reason instead.
    S = ComplexF32.(randn(64, 64) .+ 1im .* randn(64, 64))
    off = fill((0.5, 0.5), 32, 32)
    r = ResampledSLC(S, off; kernel = small)
    @test_throws "4 taps" r[20, 20]
    # The default is accepted, which is the case every caller has.
    @test ResampledSLC(S, off)[20, 20] isa ComplexF32
end

const RESAMP_FX = JSON3.read(read(joinpath(@__DIR__, "reference", "resamp.json"), String))

_unhex(rows) = reduce(vcat, [permutedims([parse(Float64, String(v)) for v in row]) for row in rows])

@testset "the resampled image is bitwise against isce3" begin
    # `ResampledSLC` presents the secondary's samples on the reference's grid; isce3's
    # `resample_to_coords` does the same given index grids, and it is the routine the NISAR workflow
    # calls. So this is the whole of Path A checked end to end against the reference, not just the kernel.
    @test RESAMP_FX.versions.isce3 == "0.25.12"

    S = ComplexF32.(_unhex(RESAMP_FX.block_real) .+ 1im .* _unhex(RESAMP_FX.block_imag))
    rg, az = _unhex(RESAMP_FX.rg_off), _unhex(RESAMP_FX.az_off)
    want = ComplexF32.(_unhex(RESAMP_FX.isce3_real) .+ 1im .* _unhex(RESAMP_FX.isce3_imag))
    baz, brg = Int(RESAMP_FX.base[1]), Int(RESAMP_FX.base[2])

    # The fixture's offsets are relative to an output window placed at `base` on the reference grid;
    # `ResampledSLC` is indexed by output pixel from one, so the placement enters as a constant.
    off = [(rg[i, j] + brg, az[i, j] + baz) for i in axes(rg, 1), j in axes(rg, 2)]
    r = ResampledSLC(S, off)
    @test size(r) == size(want)

    nexact, nfinite, worst = 0, 0, 0.0
    for i in axes(want, 1), j in axes(want, 2)
        got, w = r[i, j], want[i, j]
        if isnan(real(w))
            # The planted hole: an unplaceable point is the fill, not an interpolation somewhere
            # arbitrary. Asserted rather than skipped, since returning a *number* here would be the
            # dangerous outcome.
            @test isnan(real(got)) && isnan(imag(got))
            nexact += 1
            continue
        end
        nfinite += 1
        exact = reinterpret(UInt32, real(got)) == reinterpret(UInt32, real(w)) &&
                reinterpret(UInt32, imag(got)) == reinterpret(UInt32, imag(w))
        @test exact
        exact && (nexact += 1)
        worst = max(worst, Float64(abs(got - w)))
    end
    # The fixture must actually contain a hole and a majority of solved points, or the loop above is
    # asserting less than it appears to.
    @test nfinite == 143
    @test nexact == length(want)
    @info "resampled block vs isce3" finite = nfinite bitwise = nexact worst = worst
end

@testset "the resampled image is lazy and window-shaped" begin
    Random.seed!(11)
    n = 64
    S = ComplexF32.(randn(n, n) .+ 1im .* randn(n, n))

    # Shaped like the offset field, which is shaped like the reference image — so it lines up with a
    # `PairGeometry` over the same window and can be read a block at a time.
    r = ResampledSLC(S, fill((3.0, -2.0), n, n))
    @test size(r) == (n, n)
    @test r isa AbstractMatrix{ComplexF32}

    # An integer offset is a pure shift, so the result must be the sample itself — bitwise, since the
    # kernel's taps sum to one at the zero sub-pixel position and no interpolation is happening.
    @test r[20, 30] === S[20 - 2, 30 + 3]
    # And a zero offset is the identity.
    @test ResampledSLC(S, fill((0.0, 0.0), n, n))[20, 30] === S[20, 30]

    # A window reads that window. The values match element for element, which is what says the halo and
    # the index arithmetic agree between a windowed read and a scalar one.
    w = r[10:14, 20:26]
    @test size(w) == (5, 7)
    @test all(w[a, b] === r[9 + a, 19 + b] for a in 1:5, b in 1:7)

    # A fractional offset is the kernel applied at that position, and nothing more: at zero Doppler both
    # of `resampleToCoords`'s phasors are exactly unity, so the result must equal a direct
    # `sinc_interpolate`. An identity that were only *nearly* one would drift invisibly.
    k = SincKernel()
    rf = ResampledSLC(S, fill((0.25, 0.5), n, n); kernel = k)
    @test rf[20, 30] === sinc_interpolate(k, S, 30 + 0.25, 20 + 0.5)

    # The fill, not zero, where the stencil cannot fit. Zero is a sample value and would be
    # indistinguishable from a real one, which is why the bound is tested here rather than left to
    # `sinc_interpolate`'s own edge return.
    for (i, j) in ((1, 1), (1, 30), (30, 1), (n, 30), (30, n), (n, n))
        @test isnan(real(r[i, j]))
    end
    # A `NaN` offset is a point the geometry could not place, and yields the fill too.
    @test isnan(real(ResampledSLC(S, fill((NaN, 0.0), n, n))[20, 30]))
    @test isnan(real(ResampledSLC(S, fill((0.0, NaN), n, n))[20, 30]))
    # The fill is settable, for a caller who wants a sentinel their reader understands.
    @test ResampledSLC(S, fill((NaN, 0.0), n, n); fill = 0.0f0 + 0.0f0im)[20, 30] === 0.0f0 + 0.0f0im
end

@testset "the amplitude of a resampled image" begin
    # The composition Path A exists for: `abs` of the resampled samples is what a feature tracker reads,
    # and it must be the amplitude *of the interpolated complex value* rather than an interpolation of
    # amplitudes — those differ, and only the first is right for coherent data.
    Random.seed!(5)
    n = 48
    S = ComplexF32.(randn(n, n) .+ 1im .* randn(n, n))
    r = ResampledSLC(S, fill((1.5, -0.5), n, n))
    @test abs(r[20, 20]) ≈ abs(sinc_interpolate(SincKernel(), S, 20 + 1.5, 20 - 0.5))
    # An integer offset makes the two agree exactly, which is the case a caller can check by hand.
    ri = ResampledSLC(S, fill((2.0, -1.0), n, n))
    @test abs(ri[20, 20]) === abs(S[19, 22])
end

@testset "a doppler model needs a coordinate" begin
    # The model is a function of azimuth time and slant range; only the reference's coordinate turns an
    # output pixel into those. Refusing here rather than at the first indexed element, since the mistake
    # is in the construction.
    S = ComplexF32.(randn(8, 8) .+ 1im .* randn(8, 8))
    off = fill((0.0, 0.0), 8, 8)
    @test_throws "needs the reference acquisition" ResampledSLC(S, off; doppler = (az, rg) -> 0.0)
    # Zero Doppler is the default and needs nothing.
    @test ResampledSLC(S, off) isa ResampledSLC

    # A model that returns zero everywhere must give the same answer as no model at all: that is what
    # says the Doppler arithmetic collapses to an identity rather than merely to something small.
    Random.seed!(3)
    n = 48
    big = ComplexF32.(randn(n, n) .+ 1im .* randn(n, n))
    coord = _resamp_coord()
    plain = ResampledSLC(big, fill((1.25, -0.75), n, n))
    zeroed = ResampledSLC(big, fill((1.25, -0.75), n, n); doppler = (az, rg) -> 0.0,
                          coordinate = coord)
    @test plain[20, 20] === zeroed[20, 20]

    # A nonzero model changes the result, and it changes the *magnitude* as well as the phase. That is the
    # physics rather than a defect: derotating the chip by the Doppler phase before interpolating makes the
    # signal smoother in azimuth, so the interpolant of the derotated chip is genuinely a different number
    # — which is the whole reason the derotation is there. Reapplying the phase afterwards restores the
    # carrier at the fractional position, not the pre-interpolation magnitude.
    #
    # Measured against the per-sample phase `f * 2pi / prf`, which is what governs the size of the effect:
    #
    #     0.01 Hz   1.3e-4 rad/sample   magnitude moves 6e-5
    #     1 Hz      1.3e-2 rad/sample   magnitude moves 5e-3
    #     40 Hz     5.2e-1 rad/sample   magnitude moves 0.25
    #
    # So a small Doppler is a small correction and a large one is not, and the transition is smooth. An
    # assertion that the magnitude were *unchanged* would only hold at zero.
    for (f, tol) in ((1.0e-6, 1e-6), (0.01, 1e-3), (1.0, 5e-2))
        d = ResampledSLC(big, fill((1.25, -0.75), n, n); doppler = (az, rg) -> f, coordinate = coord)
        @test abs(d[20, 20]) ≈ abs(plain[20, 20]) rtol = tol
    end
    # A Doppler small enough to be negligible is indistinguishable from none, which is what makes the
    # zero-Doppler path the right default rather than an approximation.
    tiny = ResampledSLC(big, fill((1.25, -0.75), n, n); doppler = (az, rg) -> 1.0e-9,
                        coordinate = coord)
    @test tiny[20, 20] ≈ plain[20, 20] rtol = 1e-6

    # And a large one is a real difference, not noise — so the term is doing something and is not
    # accidentally optimized away.
    big_dop = ResampledSLC(big, fill((1.25, -0.75), n, n); doppler = (az, rg) -> 40.0,
                           coordinate = coord)
    @test abs(abs(big_dop[20, 20]) - abs(plain[20, 20])) > 0.1
end
