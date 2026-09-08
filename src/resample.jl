# The sinc interpolation kernel, for resampling complex samples onto another acquisition's grid.
#
# Port of `isce3::core::Sinc2dInterpolator` (`cxx/isce3/core/Sinc2dInterpolator.cpp`), which is what
# `resampleToCoords` instantiates and therefore what the NISAR resampling workflow uses. An 8-tap kernel
# tabulated at 8192 sub-pixel positions, Hamming-weighted and normalized per position.
#
# Two details a from-scratch implementation would not have, both of which change the answer:
#
# The table index truncates rather than rounds — `int(frac * 8192)`, clamped — so the interpolant is
# piecewise constant in the sub-pixel coordinate at a step of 1/8192 of a pixel rather than continuous.
#
# `_sinc_eval_2d` sums with a *descending* index, `arrin(intpy - i, intpx - j)`, so the kernel is applied
# reversed relative to how it reads; `interp_impl` compensates by adding the half-length to the integer
# part before calling it. Either half alone lands on the wrong samples.
#
# Both are transcribed as written. `test/resample.jl` asserts the table and the evaluations bitwise
# against isce3, which is achievable here — the coefficients are `cos` and `sin` of exactly
# representable arguments, and a normalization — so it is asserted rather than bounded.

"""
    SINC_LEN
    SINC_HALF
    SINC_ONE
    SINC_SUB

isce3's sinc kernel constants (`isce3/core/Constants.h:32-35`): kernel length 8, half length 4, chip
size 9, and 8192 tabulated sub-pixel positions.

`SINC_ONE` is 9 rather than 8 — the chip `resampleToCoords` reads is one sample wider than the kernel,
so the 8 taps sit inside it with the interpolated position in the middle.
"""
const SINC_LEN = 8
const SINC_HALF = 4
const SINC_ONE = 9
const SINC_SUB = 8192

"""
    SincKernel

The sinc interpolation kernel, tabulated at [`SINC_SUB`](@ref) sub-pixel positions.

`weights` is `(SINC_SUB, SINC_LEN)`: one row per sub-pixel position, each summing to one. Built once —
about 65536 coefficients, half a megabyte — and read per interpolated sample.

Construct with `SincKernel()` for isce3's parameters, which is the only configuration the reference uses.
"""
struct SincKernel
    weights::Matrix{Float64}
end

"""
    SincKernel(; kernel_length = SINC_LEN, decimation = SINC_SUB, beta = 1.0,
                 pedestal = 0.0, weighted = true) -> SincKernel

Tabulate the kernel. The defaults are isce3's, and are what
`Sinc2dInterpolator(SINC_LEN, SINC_SUB)` builds.

Reproduces `_sinc_coef` followed by the per-position normalization: a Hamming window of height
`1 - pedestal` over a sinc of bandwidth `beta`, then each row divided by its own sum so that a constant
signal interpolates to itself.

The unnormalized coefficient at index `i` of `decimation * kernel_length` is

    wgt = (1 - h) + h * cos(pi * (i - soff) / soff),   h = (1 - pedestal) / 2
    s   = floor(i - soff) * beta / decimation
    fct = s == 0 ? 1 : sin(pi * s) / (pi * s)

with `soff = (n - 1) / 2`. Note the `floor` inside `s`: it is in the reference and it makes the sinc
argument piecewise constant across each block of `decimation` entries, which is not what a continuous
resampling kernel would do. Transcribed rather than corrected.
"""
function SincKernel(; kernel_length::Integer = SINC_LEN, decimation::Integer = SINC_SUB,
                    beta::Real = 1.0, pedestal::Real = 0.0, weighted::Bool = true)
    kernel_length > 0 || throw(ArgumentError(
        "SincKernel kernel_length must be positive, got $kernel_length"))
    decimation > 0 || throw(ArgumentError(
        "SincKernel decimation must be positive, got $decimation"))

    n = decimation * kernel_length
    filter = Vector{Float64}(undef, n)
    h = (1.0 - Float64(pedestal)) / 2.0
    soff = (n - 1) / 2.0
    for i in 0:(n - 1)
        wgt = (1.0 - h) + h * cos((pi * (i - soff)) / soff)
        s = (floor(i - soff) * Float64(beta)) / decimation
        fct = s != 0.0 ? sin(pi * s) / (pi * s) : 1.0
        filter[i + 1] = weighted ? fct * wgt : fct
    end

    # Transposed into `(position, tap)` and normalized per position, as the reference's constructor does:
    # `filter[i + decimation * j]` becomes `weights[i + 1, j + 1]`.
    weights = Matrix{Float64}(undef, decimation, kernel_length)
    for i in 0:(decimation - 1)
        ssum = 0.0
        for j in 0:(kernel_length - 1)
            ssum += filter[i + decimation * j + 1]
        end
        for j in 0:(kernel_length - 1)
            weights[i + 1, j + 1] = filter[i + decimation * j + 1] / ssum
        end
    end
    return SincKernel(weights)
end

"""
    kernel_length(k::SincKernel) -> Int
    decimation(k::SincKernel) -> Int

The kernel's tap count and its number of tabulated sub-pixel positions.
"""
kernel_length(k::SincKernel) = size(k.weights, 2)
decimation(k::SincKernel) = size(k.weights, 1)

"""
    sinc_interpolate(k::SincKernel, chip, x, y; accumulate = eltype(chip)) -> eltype(chip)

Interpolate `chip` at the position `(x, y)`, one-based and fractional, with `x` along the second axis.

Ports `Sinc2dInterpolator::interp_impl` and `_sinc_eval_2d` together. Returns zero where the kernel's
stencil would reach outside `chip`, as isce3 does — the caller decides what an unreachable sample means,
and for a resampler that is the fill value rather than zero.

The axis convention is isce3's: `x` indexes columns and `y` rows, so `chip[y, x]`. The names are kept
because the reference's are, and swapping them silently transposes every resampled image.

# `accumulate` decides whether this is bitwise against isce3

isce3 accumulates in the sample type and casts each weight to it: `ret += arrin(...) *
static_cast<U>(_kernel(ifracy, i)) * static_cast<U>(_kernel(ifracx, j))` with `U = complex<float>`. So a
64-tap sum is performed in `Float32`, with two roundings per term.

The default reproduces that, and it is **bitwise** — verified on every probed position against
`isce3.image.v2.resample_slc.resample_to_coords`. Accumulating in `ComplexF64` instead is more accurate
by about one `Float32` ULP and *not* bitwise, so it is offered rather than imposed:

```julia
sinc_interpolate(k, chip, x, y)                        # bitwise against isce3
sinc_interpolate(k, chip, x, y; accumulate = ComplexF64)  # 1 ULP better, not bitwise
```

The difference reaches an amplitude at 6e-8 relative, which no correlator can see. The default is the
reference's because this package's standard is to match it and to say where it does not — see
`REFERENCE.md` — not because the extra rounding is desirable.
"""
function sinc_interpolate(k::SincKernel, chip::AbstractMatrix, x::Real, y::Real;
                          accumulate::Type = eltype(chip))
    kl = kernel_length(k)
    half = kl ÷ 2
    dec = decimation(k)

    # isce3 works in zero-based indices; `chip` is one-based, so the integer parts are taken zero-based
    # and shifted back when the chip is read.
    fx = Float64(x) - 1.0
    fy = Float64(y) - 1.0
    ix = floor(Int, fx)
    iy = floor(Int, fy)
    frx = fx - ix
    fry = fy - iy

    z = zero(eltype(chip))
    # `Rdr2Geo`-style edge test, from `interp_impl`: the stencil needs `half - 1` samples below and `half`
    # above, in zero-based terms.
    (ix < half - 1 || ix > size(chip, 2) - half - 1) && return z
    (iy < half - 1 || iy > size(chip, 1) - half - 1) && return z

    # Truncating, not rounding — see the note at the top of this file. Clamped as the reference clamps.
    ifracx = min(max(0, floor(Int, frx * dec)), dec - 1) + 1
    ifracy = min(max(0, floor(Int, fry * dec)), dec - 1) + 1

    # `interp_impl` adds the half length before calling `_sinc_eval_2d`, which then subtracts the tap
    # index — so the stencil runs downward from here. Plus one for one-based chip indexing.
    xx = ix + half + 1
    yy = iy + half + 1

    # Each weight cast to the accumulator type separately, and the two multiplied against the sample
    # rather than pre-multiplied — the reference's association, which matters at `Float32` because
    # `(a*wy)*wx` and `a*(wy*wx)` round differently.
    acc = zero(accumulate)
    for i in 1:kl
        wy = convert(accumulate, k.weights[ifracy, i])
        row = yy - i + 1
        for j in 1:kl
            acc += chip[row, xx - j + 1] * wy * convert(accumulate, k.weights[ifracx, j])
        end
    end
    return oftype(z, acc)
end

"""
    ResampledSLC(samples, offset; kernel = SincKernel(), fill = NaN32 + NaN32im,
                 doppler = nothing, coordinate = nothing)

The secondary acquisition's samples on the *reference* acquisition's grid, interpolated on indexing.

An `AbstractMatrix{ComplexF32}` shaped like `offset`, so it lines up with the reference image and can be
read a window at a time. Nothing is resampled until it is asked for: indexing a window reads that window
from `samples`, grown by [`SINC_HALF`](@ref) plus whatever the offsets there demand, and interpolates.

`samples` is the secondary's complex samples — anything `AbstractMatrix{<:Complex}`, so
`SLCDatasets.pixels` serves directly. `offset` gives `(dsample, dline)` per *output* pixel, as
[`OffsetField`](@ref) and [`LatticeOffsetField`](@ref) produce, and is indexed the same way this is.

# Why this is worth being lazy about

A resampled Sentinel-1 subswath is 33 million complex samples, and a correlator reads a few hundred
thousand of them. So `amplitude(ResampledSLC(...))` — or a window taken directly — reads the chips a
consumer asks for and no more, which is the same shape `SLCDatasets.Amplitude` already has and composes
with it.

# Doppler

Follows `resampleToCoords` (`cxx/isce3/image/v2/Resample.cpp`): the chip is read with the azimuth Doppler
phase *removed* row by row, interpolated, and the phase for the fractional azimuth position reapplied. For
a zero-Doppler grid — NISAR, and every acquisition this package handles today — the Doppler is zero, both
phasors are unity, and the arithmetic reduces to the interpolation alone. That is why this is usable before
any Doppler model exists.

`doppler` supplies a nonzero one as a callable `doppler(aztime, range) -> Hz`, in which case `coordinate`
is required: converting an output pixel to an azimuth time and a slant range needs the reference's
[`RadarCoordinate`](@ref). Both default to `nothing`, which is the zero-Doppler case.

!!! warning "Not for TOPS data"
    Sentinel-1 IW sweeps the antenna in azimuth within each burst, so the azimuth phase carries a steep
    ramp that must be removed before interpolation and reapplied after. This does not do that, and
    interpolating a TOPS burst without it aliases the ramp — worst at the burst edges. The amplitudes are
    unaffected, since `abs` discards the phase.

    A bare `samples` matrix carries no record of how it was collected, so this type cannot check it. Load
    `SLCDatasets` and pass the acquisition instead of its samples: `SLCDatasets.is_tops` answers the
    question, and the extension's `ResampledSLC(::SLC, offset)` refuses a TOPS acquisition unless
    `amplitude_only = true` says the phase will not be read.
"""
struct ResampledSLC{S<:AbstractMatrix,O<:AbstractMatrix,K,D,C} <: AbstractMatrix{ComplexF32}
    samples::S
    offset::O
    kernel::K
    fill::ComplexF32
    doppler::D
    coordinate::C

    function ResampledSLC{S,O,K,D,C}(samples, offset, kernel, fill, doppler,
                                     coordinate) where {S,O,K,D,C}
        # A Doppler model without a coordinate cannot be evaluated: the model is a function of azimuth
        # time and slant range, and only the coordinate converts a pixel to those.
        (doppler === nothing || coordinate !== nothing) || throw(ArgumentError(
            "ResampledSLC was given a doppler model but no coordinate; evaluating the model needs the " *
            "reference acquisition's RadarCoordinate to turn an output pixel into an azimuth time and " *
            "a slant range. Pass `coordinate`, or leave `doppler` unset for the zero-Doppler case."))
        return new{S,O,K,D,C}(samples, offset, kernel, fill, doppler, coordinate)
    end
end

function ResampledSLC(samples::AbstractMatrix, offset::AbstractMatrix;
                      kernel = SincKernel(), fill = ComplexF32(NaN32, NaN32),
                      doppler = nothing, coordinate = nothing)
    return ResampledSLC{typeof(samples),typeof(offset),typeof(kernel),typeof(doppler),
                        typeof(coordinate)}(
        samples, offset, kernel, ComplexF32(fill), doppler, coordinate)
end

Base.size(r::ResampledSLC) = size(r.offset)
Base.axes(r::ResampledSLC) = axes(r.offset)
Base.IndexStyle(::Type{<:ResampledSLC}) = IndexCartesian()

# The Doppler frequency at an output pixel, in radians per sample, as `Resample.cpp` computes it:
# `lut.eval(az_time, rg_distance) * 2 * pi / prf`. Zero without a model, which collapses both phasors to
# unity and is the only case the reference's own callers exercise on a zero-Doppler grid.
@inline _doppler_rad(::Nothing, ::Any, ::Int, ::Int) = 0.0

@inline function _doppler_rad(dop, c, samp::Int, line::Int)
    # Zero-based, as the reference's indices are.
    rg = c.starting_range + samp * c.dr
    az = c.sensing_start + line / c.prf
    return Float64(dop(az, rg)) * 2 * pi / c.prf
end

Base.@propagate_inbounds function Base.getindex(r::ResampledSLC, i::Int, j::Int)
    @boundscheck checkbounds(r, i, j)

    ds, dl = r.offset[i, j]
    # A `NaN` offset is a point the geometry could not place, and it yields the fill rather than an
    # interpolation at an arbitrary position — the reference's behavior, and the honest one.
    (isnan(ds) || isnan(dl)) && return r.fill

    # `offsets_to_indices`: the absolute input index is the output index plus the offset. This package's
    # indices are one-based on both sides, so the relation carries over unchanged.
    x = j + ds
    y = i + dl

    # The stencil has to fit. `sinc_interpolate` returns zero rather than throwing when it does not, and
    # zero is a *sample value* — indistinguishable from a real one — so the bound is tested here and the
    # fill returned instead.
    kl = kernel_length(r.kernel)
    half = kl ÷ 2
    ix = floor(Int, x)
    iy = floor(Int, y)
    (ix < half || ix > size(r.samples, 2) - half) && return r.fill
    (iy < half || iy > size(r.samples, 1) - half) && return r.fill

    dop = _doppler_rad(r.doppler, r.coordinate, j - 1, i - 1)
    return _resample_at(r.kernel, r.samples, x, y, dop, r.fill)
end

# One resampled sample: the chip read with the Doppler phase removed, interpolated, and the phase for the
# fractional azimuth position put back.
#
# Transcribes `resampleToCoords`'s inner loop. At `dop == 0` both phasors are exactly `1 + 0im`, so the
# multiplications are identities and the result is the plain interpolation — checked in `test/resample.jl`
# against calling `sinc_interpolate` directly, since an identity that is only nearly one would show up as
# a drift no correlator could attribute.
function _resample_at(k::SincKernel, samples::AbstractMatrix, x::Float64, y::Float64,
                      dop::Float64, fill::ComplexF32)
    kl = kernel_length(k)
    half = kl ÷ 2
    chip_size = kl + 1
    iy = floor(Int, y)
    frac_az = y - iy

    # `SINC_ONE` samples on a side, with the interpolated position inside it — the chip `resampleToCoords`
    # reads. Built per sample: a `ComplexF32` 9x9 is 648 bytes and stack-allocated in practice, and
    # threading a buffer through would make this type stateful for no measured gain.
    chip = Matrix{ComplexF32}(undef, chip_size, chip_size)
    ix = floor(Int, x)
    for ci in 1:chip_size
        # The chip's rows run from `half` below the integer position, and the Doppler phase is per row.
        srow = iy + ci - half - 1
        phase = dop * (ci - 1 - half)
        conj_phasor = ComplexF32(cos(phase), -sin(phase))
        for cj in 1:chip_size
            scol = ix + cj - half - 1
            (srow < 1 || srow > size(samples, 1) || scol < 1 || scol > size(samples, 2)) &&
                return fill
            chip[ci, cj] = ComplexF32(samples[srow, scol]) * conj_phasor
        end
    end

    # The position within the chip: the fractional part offset by the half length, one-based.
    frac_rg = x - ix
    val = sinc_interpolate(k, chip, half + frac_rg + 1, half + frac_az + 1)

    # And the phase corresponding to where in azimuth the sample was taken, reapplied.
    phase = dop * frac_az
    return val * ComplexF32(cos(phase), sin(phase))
end
