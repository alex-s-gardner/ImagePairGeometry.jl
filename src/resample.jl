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

The sinc interpolation kernel, tabulated at `SINC_SUB` sub-pixel positions (see [`SINC_LEN`](@ref)).

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
sinc_interpolate(k::SincKernel, chip::AbstractMatrix, x::Real, y::Real;
                 accumulate::Type = eltype(chip)) =
    _sinc_interpolate(k, chip, Float64(x), Float64(y), accumulate)

# The accumulator arrives as a *type parameter* rather than as a value, which is not cosmetic: passing it
# as a field of the argument list leaves `zero(accumulate)` and `convert(accumulate, ...)` uninferable, and
# a 64-tap sum then boxes every partial. Measured at 7.2 us and 336 allocations that way against 82 ns and
# none this way — 87x, for identical values.
function _sinc_interpolate(k::SincKernel, chip::AbstractMatrix, x::Float64, y::Float64,
                           ::Type{A}) where {A}
    kl = kernel_length(k)
    half = kl ÷ 2
    dec = decimation(k)

    # isce3 works in zero-based indices; `chip` is one-based, so the integer parts are taken zero-based
    # and shifted back when the chip is read.
    fx = x - 1.0
    fy = y - 1.0
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
    acc = zero(A)
    for i in 1:kl
        wy = convert(A, k.weights[ifracy, i])
        row = yy - i + 1
        for j in 1:kl
            acc += chip[row, xx - j + 1] * wy * convert(A, k.weights[ifracx, j])
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
from `samples`, grown by `SINC_HALF` plus whatever the offsets there demand, and interpolates.

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
struct ResampledSLC{S<:AbstractMatrix,O<:AbstractMatrix,K,D,C,R} <: AbstractMatrix{ComplexF32}
    samples::S
    offset::O
    kernel::K
    fill::ComplexF32
    doppler::D
    coordinate::C
    carrier::R

    function ResampledSLC{S,O,K,D,C,R}(samples, offset, kernel, fill, doppler, coordinate,
                                       carrier) where {S,O,K,D,C,R}
        # Checked before the coordinate, since supplying both hooks is the more basic mistake: only one of
        # them would take effect, and which one is an implementation detail rather than a choice. Both would
        # otherwise be removed from the same chip, and their sum is not what either means.
        (doppler === nothing || carrier === nothing) || throw(ArgumentError(
            "ResampledSLC was given both a `doppler` model and a `carrier`, which are two ways to say " *
            "the same thing: the azimuth phase to remove before interpolating. Supply one. A carrier " *
            "that needs both terms should evaluate both and return their sum."))
        # A Doppler model without a coordinate cannot be evaluated: the model is a function of azimuth
        # time and slant range, and only the coordinate converts a pixel to those.
        (doppler === nothing || coordinate !== nothing) || throw(ArgumentError(
            "ResampledSLC was given a doppler model but no coordinate; evaluating the model needs the " *
            "reference acquisition's RadarCoordinate to turn an output pixel into an azimuth time and " *
            "a slant range. Pass `coordinate`, or leave `doppler` unset for the zero-Doppler case."))
        return new{S,O,K,D,C,R}(samples, offset, kernel, fill, doppler, coordinate, carrier)
    end
end

function ResampledSLC(samples::AbstractMatrix, offset::AbstractMatrix;
                      kernel = SincKernel(), fill = ComplexF32(NaN32, NaN32),
                      doppler = nothing, coordinate = nothing, carrier = nothing)
    return ResampledSLC{typeof(samples),typeof(offset),typeof(kernel),typeof(doppler),
                        typeof(coordinate),typeof(carrier)}(
        samples, offset, kernel, ComplexF32(fill), doppler, coordinate, carrier)
end

Base.size(r::ResampledSLC) = size(r.offset)
Base.axes(r::ResampledSLC) = axes(r.offset)
Base.IndexStyle(::Type{<:ResampledSLC}) = IndexCartesian()

# The azimuth phase an interpolation has to take out and put back, as a function of position in the
# *secondary's* samples.
#
# Two kinds reach this, and they are not the same shape, which is why the phase rather than a frequency is
# the protocol. `Resample.cpp`'s Doppler is a frequency turned into a phase linear in the chip row. A TOPS
# azimuth ramp is quadratic about each burst's own center and referenced to absolute burst coordinates. A
# frequency hook cannot express the second, and a phase hook expresses both.
#
# The contract: a phase is only ever used as a *difference* from the chip's integer center, so a carrier is
# free to return a large absolute value — the difference is what the arithmetic sees, which keeps
# `cos`/`sin` away from arguments where they lose their low bits.
#
# `_chip_phase` is the phase removed from the sample at `(srow, scol)`; `_out_phase` the phase reapplied for
# the interpolated position `(y, x)`. Both are relative to `(iy, ix)`.

# No carrier: zero, and both phasors collapse to exactly `1 + 0im`. Dispatched rather than computed, so the
# no-carrier path does no phasor arithmetic at all — `_resample_at` skips it entirely.
@inline _chip_phase(::Nothing, ::Int, ::Int, ::Int, ::Int) = 0.0
@inline _out_phase(::Nothing, ::Float64, ::Float64, ::Int, ::Int) = 0.0

# A general carrier, evaluated at absolute secondary indices and differenced against the chip's integer
# center. The constant cancels between removal and reapplication — every tap is multiplied by `exp(-i(phi -
# C))` and the result by `exp(i(phi_out - C))` — so subtracting it changes nothing but the magnitude of the
# arguments `cos` and `sin` see.
@inline _chip_phase(carrier, srow::Int, scol::Int, iy::Int, ix::Int) =
    Float64(carrier(srow, scol)) - Float64(carrier(iy, ix))

@inline _out_phase(carrier, y::Float64, x::Float64, iy::Int, ix::Int) =
    Float64(carrier(y, x)) - Float64(carrier(iy, ix))

# `Resample.cpp`'s Doppler, as a carrier. The phase is `dop * (row - iy)` and does not vary with range
# within a chip, which is the reference's own form: it computes one frequency per output pixel and applies
# it per chip row. Kept as its own type rather than expressed through the general path above because the
# association of the arithmetic is what makes this bitwise against the reference — `dop * (srow - iy)` and
# `dop * srow - dop * iy` are the same number in exact arithmetic and not always in `Float64`.
struct DopplerCarrier
    rad_per_sample::Float64
end

@inline _chip_phase(d::DopplerCarrier, srow::Int, ::Int, iy::Int, ::Int) =
    d.rad_per_sample * (srow - iy)

@inline _out_phase(d::DopplerCarrier, y::Float64, ::Float64, iy::Int, ::Int) =
    d.rad_per_sample * (y - iy)

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

    return _resample_at(r.kernel, r.samples, x, y, _carrier_at(r, i, j), r.fill)
end

# The carrier for one output pixel.
#
# A `doppler` model is a function of azimuth time and slant range, so it is evaluated once per output pixel
# and the resulting frequency carried into the chip loop — the reference's own structure, and where the
# `DopplerCarrier` wrapper comes from. A `carrier` is already a function of position, so it is used as it
# stands. The constructor refuses both at once, so these two cases are exhaustive.
@inline _carrier_at(r::ResampledSLC, i::Int, j::Int) = _carrier_at(r.doppler, r, i, j)

# Dispatching on the `doppler` field's own type rather than on a type parameter's position: the latter is
# fragile — a parameter added to the struct shifts it — and got this wrong once already, silently discarding
# the carrier and resampling as though there were none.
@inline _carrier_at(::Nothing, r::ResampledSLC, ::Int, ::Int) = r.carrier

@inline _carrier_at(dop, r::ResampledSLC, i::Int, j::Int) =
    DopplerCarrier(_doppler_rad(dop, r.coordinate, j - 1, i - 1))

# One resampled sample: the chip read with the azimuth carrier phase removed, interpolated, and the phase
# for the interpolated position put back.
#
# Transcribes `resampleToCoords`'s inner loop. Removing the carrier before interpolating is what makes the
# interpolation valid: a phase varying across the chip is a frequency offset, and a signal whose band is
# off-center aliases when resampled. The reference does this for the Doppler; a TOPS azimuth ramp is the
# same operation with a steeper, quadratic phase.
function _resample_at(k::SincKernel, samples::AbstractMatrix, x::Float64, y::Float64,
                      carrier, fill::ComplexF32)
    kl = kernel_length(k)
    half = kl ÷ 2
    chip_size = kl + 1
    iy = floor(Int, y)
    ix = floor(Int, x)

    # `SINC_ONE` samples on a side, with the interpolated position inside it — the chip `resampleToCoords`
    # reads. An `MMatrix` rather than a `Matrix`: the size is a compile-time constant, so this lives on the
    # stack and the per-sample cost is 233 ns rather than 233 ns plus two heap allocations. Threading a
    # buffer through the type instead would make it stateful, which a lazy array read from several tasks
    # must not be.
    chip = MMatrix{SINC_ONE,SINC_ONE,ComplexF32}(undef)
    _fill_chip!(chip, carrier, samples, iy, ix, half, chip_size) || return fill

    # The position within the chip: the fractional part offset by the half length, one-based.
    frac_rg = x - ix
    frac_az = y - iy
    val = _sinc_interpolate(k, chip, half + frac_rg + 1.0, half + frac_az + 1.0, ComplexF32)

    return _reapply_carrier(carrier, val, y, x, iy, ix)
end

# The chip, with the carrier removed. `false` where the stencil runs off the samples, which the caller turns
# into the fill — zero would be indistinguishable from a real sample value.
#
# Three methods rather than one because what may be hoisted differs, and for the Doppler case the
# association of the arithmetic is what holds it bitwise against the reference.

# No carrier: the samples as they are. Not a multiplication by `1 + 0im` — an actual identity, so a caller
# that supplies no carrier gets exactly what `sinc_interpolate` alone would give.
@inline function _fill_chip!(chip, ::Nothing, samples, iy::Int, ix::Int, half::Int, chip_size::Int)
    for ci in 1:chip_size
        srow = iy + ci - half - 1
        for cj in 1:chip_size
            scol = ix + cj - half - 1
            _in_samples(samples, srow, scol) || return false
            chip[ci, cj] = ComplexF32(samples[srow, scol])
        end
    end
    return true
end

# The Doppler: one phasor per chip row, since the reference's frequency does not vary with range within a
# chip. Hoisting it out of the inner loop is the reference's own structure.
@inline function _fill_chip!(chip, d::DopplerCarrier, samples, iy::Int, ix::Int, half::Int,
                             chip_size::Int)
    for ci in 1:chip_size
        srow = iy + ci - half - 1
        phase = _chip_phase(d, srow, ix, iy, ix)
        conj_phasor = ComplexF32(cos(phase), -sin(phase))
        for cj in 1:chip_size
            scol = ix + cj - half - 1
            _in_samples(samples, srow, scol) || return false
            chip[ci, cj] = ComplexF32(samples[srow, scol]) * conj_phasor
        end
    end
    return true
end

# A general carrier varies with range as well as azimuth — a TOPS ramp's coefficient is a function of slant
# range — so the phase is per sample and nothing may be hoisted.
@inline function _fill_chip!(chip, carrier, samples, iy::Int, ix::Int, half::Int, chip_size::Int)
    for ci in 1:chip_size
        srow = iy + ci - half - 1
        for cj in 1:chip_size
            scol = ix + cj - half - 1
            _in_samples(samples, srow, scol) || return false
            phase = _chip_phase(carrier, srow, scol, iy, ix)
            chip[ci, cj] = ComplexF32(samples[srow, scol]) * ComplexF32(cos(phase), -sin(phase))
        end
    end
    return true
end

@inline _in_samples(samples, srow::Int, scol::Int) =
    srow >= 1 && srow <= size(samples, 1) && scol >= 1 && scol <= size(samples, 2)

# The phase for where the sample was actually taken, reapplied. An identity without a carrier, for the same
# reason the chip fill is.
@inline _reapply_carrier(::Nothing, val::ComplexF32, ::Float64, ::Float64, ::Int, ::Int) = val

@inline function _reapply_carrier(carrier, val::ComplexF32, y::Float64, x::Float64, iy::Int, ix::Int)
    phase = _out_phase(carrier, y, x, iy, ix)
    return val * ComplexF32(cos(phase), sin(phase))
end
