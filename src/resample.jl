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
# That reference value is the same for all 81 taps of a chip, so it is computed once per output sample by
# `_reference_phase` and passed in. Differencing against a freshly evaluated `carrier(iy, ix)` inside the tap
# loop instead costs a second carrier evaluation per tap — half the deramp's total runtime.
#
# `_chip_phase` is the phase removed from the sample at `(srow, scol)`; `_out_phase` the phase reapplied for
# the interpolated position `(y, x)`. Both are relative to `ref`.

# No carrier: zero, and both phasors collapse to exactly `1 + 0im`. Dispatched rather than computed, so the
# no-carrier path does no phasor arithmetic at all — `_resample_at` skips it entirely.
@inline _reference_phase(::Nothing, ::Int, ::Int) = 0.0
@inline _chip_phase(::Nothing, ::Int, ::Int, ::Float64) = 0.0
@inline _out_phase(::Nothing, ::Float64, ::Float64, ::Float64) = 0.0

# A general carrier, evaluated at absolute secondary indices and differenced against the chip's integer
# center. The constant cancels between removal and reapplication — every tap is multiplied by `exp(-i(phi -
# C))` and the result by `exp(i(phi_out - C))` — so subtracting it changes nothing but the magnitude of the
# arguments `cos` and `sin` see.
@inline _reference_phase(carrier, iy::Int, ix::Int) = Float64(carrier(iy, ix))

@inline _chip_phase(carrier, srow::Int, scol::Int, ref::Float64) =
    Float64(carrier(srow, scol)) - ref

@inline _out_phase(carrier, y::Float64, x::Float64, ref::Float64) =
    Float64(carrier(y, x)) - ref

# `Resample.cpp`'s Doppler, as a carrier. The phase is `dop * (row - iy)` and does not vary with range
# within a chip, which is the reference's own form: it computes one frequency per output pixel and applies
# it per chip row.
#
# A closure `(line, sample) -> dop * line` would express the same thing through the general path above, and
# must not replace this: the general path differences two evaluated phases, giving `dop*srow - dop*iy`, where
# this multiplies the differenced index. The two agree in exact arithmetic and disagree in `Float64` — 140 of
# 315 (dop, row) pairs across the plausible range differ in the last bits — and it is this form that is
# bitwise against the reference.
struct DopplerCarrier
    rad_per_sample::Float64
end

# The reference is the chip's integer centre row, carried as a `Float64` like any other.
@inline _reference_phase(::DopplerCarrier, iy::Int, ::Int) = Float64(iy)

@inline _chip_phase(d::DopplerCarrier, srow::Int, ::Int, ref::Float64) =
    d.rad_per_sample * (srow - ref)

@inline _out_phase(d::DopplerCarrier, y::Float64, ::Float64, ref::Float64) =
    d.rad_per_sample * (y - ref)

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
    iy = floor(Int, y)
    ix = floor(Int, x)

    # The chip's size is the compile-time `SINC_ONE`, which is what lets it be an `SMatrix`. So a kernel of
    # another length would read a stencil of the wrong extent — silently, since the sizes would still agree
    # with each other. Refused instead: `SincKernel`'s only documented configuration is the reference's.
    kl == SINC_LEN || throw(ArgumentError(
        "resampling is implemented for isce3's $SINC_LEN-tap kernel, but this one has $kl taps. The chip " *
        "size is a compile-time constant, so another length would read a stencil of the wrong extent. Use " *
        "`SincKernel()`, or call `sinc_interpolate` directly with a chip you have read yourself."))

    # The whole stencil has to fit. Tested here rather than per tap: the two are equivalent — a stencil
    # spans `iy-half` to `iy-half+SINC_ONE-1` and every row of it is in range exactly when the ends are —
    # and one test keeps the chip below immutable, which is what stops it reaching the heap.
    #
    # This is not the caller's test. `getindex` bounds `iy` to `half..size-half`, which admits `iy == half`,
    # where the stencil's first row is 0. So this rejects a one-row band the caller lets through.
    _stencil_fits(samples, iy, ix, half) || return fill

    # The phase every tap is measured against. Loop-invariant, so it is evaluated once here rather than 81
    # times inside the chip.
    ref = _reference_phase(carrier, iy, ix)

    # `SINC_ONE` samples on a side with the interpolated position inside it — the chip `resampleToCoords`
    # reads, with the carrier phase already removed. An `SMatrix`: the size is a compile-time constant and
    # nothing mutates it, so it stays in registers. An `MMatrix` filled in place is the obvious way to write
    # this and costs 648 bytes of heap per output sample, because a mutable static array escapes.
    chip = _chip(carrier, samples, iy, ix, half, ref)

    # The position within the chip: the fractional part offset by the half length, one-based.
    frac_rg = x - ix
    frac_az = y - iy
    val = _sinc_interpolate(k, chip, half + frac_rg + 1.0, half + frac_az + 1.0, ComplexF32)

    return _reapply_carrier(carrier, val, y, x, ref)
end

@inline _stencil_fits(samples, iy::Int, ix::Int, half::Int) =
    iy - half >= 1 && iy - half + SINC_ONE - 1 <= size(samples, 1) &&
    ix - half >= 1 && ix - half + SINC_ONE - 1 <= size(samples, 2)

# The chip, with the carrier removed.
#
# `ntuple` with a compile-time-known length rather than a comprehension: `SMatrix`'s constructor needs a
# tuple whose length it can see, and a nested generator does not supply one.
#
# The elements run in *column-major* order — `n` splits into a row within a column — because that is the
# order the constructor consumes them. The transpose of this would silently mirror every resampled image
# about its diagonal.
#
# Two methods. A carrier's phase depends on the tap, so it is evaluated per element; without one there is
# nothing to remove, and the samples pass through as an actual identity rather than a multiplication by
# `1 + 0im`, so a caller supplying no carrier gets exactly what `sinc_interpolate` alone would give.
@inline _chip(::Nothing, samples, iy::Int, ix::Int, half::Int, ::Float64) =
    SMatrix{SINC_ONE,SINC_ONE,ComplexF32}(ntuple(
        n -> ComplexF32(samples[iy + (n - 1) % SINC_ONE - half, ix + (n - 1) ÷ SINC_ONE - half]),
        Val(SINC_ONE * SINC_ONE)))

@inline _chip(carrier, samples, iy::Int, ix::Int, half::Int, ref::Float64) =
    SMatrix{SINC_ONE,SINC_ONE,ComplexF32}(ntuple(
        n -> _tap(carrier, samples, iy + (n - 1) % SINC_ONE - half,
                  ix + (n - 1) ÷ SINC_ONE - half, ref),
        Val(SINC_ONE * SINC_ONE)))

# One tap with its carrier phase removed.
@inline function _tap(carrier, samples, srow::Int, scol::Int, ref::Float64)
    phase = _chip_phase(carrier, srow, scol, ref)
    return ComplexF32(samples[srow, scol]) * ComplexF32(cos(phase), -sin(phase))
end

# The phase for where the sample was actually taken, reapplied. An identity without a carrier, for the same
# reason the chip fill is.
@inline _reapply_carrier(::Nothing, val::ComplexF32, ::Float64, ::Float64, ::Float64) = val

@inline function _reapply_carrier(carrier, val::ComplexF32, y::Float64, x::Float64, ref::Float64)
    phase = _out_phase(carrier, y, x, ref)
    return val * ComplexF32(cos(phase), sin(phase))
end
