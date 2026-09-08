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
                         SINC_LEN, SINC_HALF, SINC_ONE, SINC_SUB
using JSON3
using Test

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
end
