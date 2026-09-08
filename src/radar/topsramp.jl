# The azimuth phase a TOPS acquisition's samples carry, and which has to come off before they are
# interpolated.
#
# TOPS steers the antenna in azimuth across each burst. The Doppler centroid therefore sweeps through the
# burst, and the focused samples carry a phase quadratic in azimuth about the burst's own center. Two
# consequences shape this file.
#
# It is a *carrier*, not a filter. A phase varying across a chip is a frequency offset, and a signal whose
# band is off-center aliases when resampled — so the phase is removed before the sinc convolution and put
# back after, which is why this is a callable handed to `ResampledSLC` rather than an image transform.
#
# It is referenced to the burst, not the image. `eta` counts from the burst's center line, so a merged
# subswath carries one of these per burst; evaluating one against merged-grid rows would be wrong by up to
# half a burst. `SLCDatasets.burst_at` is what resolves a line to its burst.
#
# The arithmetic follows `s1reader`'s `az_carrier_components`, which cites ESA's TOPS deramping note:
#
#     carrier = pi * kt * (eta - eta_ref)^2
#     kt      = ks / (1 - ks/ka)
#     ks      = 2 * |v| * steering_rate / wavelength
#     eta     = (line - center_line) * azimuth_time_interval
#     eta_ref = f_etac(r0)/ka(r0) - f_etac(rng)/ka(rng)
#
# `ka` is the azimuth FM rate and `f_etac` the Doppler centroid, both polynomials in slant range. Only
# `eta_ref`'s second term varies across the swath, which is why the first is hoisted into the type.

"""
    TOPSCarrier(; azimuth_fm_rate, doppler_centroid, ks, eta_ref_near, starting_range,
                range_pixel_spacing, azimuth_time_interval, center_line)

The azimuth carrier phase of one TOPS burst, as a callable `carrier(line, sample) -> radians`.

Pass to [`ResampledSLC`](@ref) as `carrier`, which removes this phase from each chip before interpolating
and reapplies it for the interpolated position. Without that, interpolating TOPS samples aliases the ramp —
worst at the burst edges, where it is steepest — and yields a phase-corrupted image that looks valid.

`line` and `sample` are one-based indices into the *burst's own* samples, and may be fractional: the phase
is evaluated at the interpolated position as well as at the integer taps.

# Fields
- `azimuth_fm_rate`, `doppler_centroid`: callables of slant range in meters, returning Hz/s and Hz.
- `ks`: the Doppler rate from the antenna sweep, `2|v| * steering_rate / wavelength`, in Hz/s.
- `eta_ref_near`: `doppler_centroid(r0) / azimuth_fm_rate(r0)`, the range-invariant part of the reference
  time, in seconds.
- `starting_range`, `range_pixel_spacing`: meters.
- `azimuth_time_interval`: seconds per line.
- `center_line`: the line the ramp is referenced to, one-based.

Build one from a product with `SLCDatasets` loaded, which supplies every term but `ks` — see the extension's
`TOPSCarrier(::SLCDatasets.DerampParameters, orbit)`.
"""
struct TOPSCarrier{P,Q}
    azimuth_fm_rate::P
    doppler_centroid::Q
    ks::Float64
    eta_ref_near::Float64
    starting_range::Float64
    range_pixel_spacing::Float64
    azimuth_time_interval::Float64
    center_line::Float64
end

function TOPSCarrier(; azimuth_fm_rate, doppler_centroid, ks::Real, eta_ref_near::Real,
                     starting_range::Real, range_pixel_spacing::Real,
                     azimuth_time_interval::Real, center_line::Real)
    ks == 0 && throw(ArgumentError(
        "a TOPS carrier needs a nonzero `ks`: it is `2|v| * steering_rate / wavelength`, and zero would " *
        "mean the antenna does not sweep — which is stripmap, not TOPS, and carries no ramp to remove."))
    range_pixel_spacing > 0 || throw(ArgumentError(
        "`range_pixel_spacing` must be positive, got $range_pixel_spacing"))
    azimuth_time_interval > 0 || throw(ArgumentError(
        "`azimuth_time_interval` must be positive, got $azimuth_time_interval"))
    return TOPSCarrier{typeof(azimuth_fm_rate),typeof(doppler_centroid)}(
        azimuth_fm_rate, doppler_centroid, Float64(ks), Float64(eta_ref_near),
        Float64(starting_range), Float64(range_pixel_spacing), Float64(azimuth_time_interval),
        Float64(center_line))
end

"""
    slant_range(c::TOPSCarrier, sample) -> Float64

The slant range of a sample of the burst, in meters.
"""
@inline slant_range(c::TOPSCarrier, sample::Real) =
    c.starting_range + (Float64(sample) - 1.0) * c.range_pixel_spacing

# The phase in radians at one position in the burst. Both indices may be fractional — the interpolated
# position is not on the sample grid.
function (c::TOPSCarrier)(line::Real, sample::Real)
    rng = slant_range(c, sample)
    ka = Float64(c.azimuth_fm_rate(rng))
    # `ka` divides twice below. It is a few thousand Hz/s over the swath of every product measured, so zero
    # means the polynomial was mis-parsed or evaluated at a range it does not cover, and continuing would
    # give an infinite phase that resampling turns into `NaN` samples with no indication of the cause.
    ka == 0 && throw(DomainError(rng,
        "the azimuth FM rate is zero at this slant range, so the TOPS carrier is undefined: it divides " *
        "by the rate twice. The polynomial is most likely being evaluated outside the range it was " *
        "estimated for, or its reference range is wrong."))

    denom = 1.0 - c.ks / ka
    # `kt = ks/(1 - ks/ka)` is singular where the sweep rate meets the FM rate. Physically they differ by
    # orders of magnitude — a few kHz/s against a few thousand Hz/s of opposite sign — so this means the two
    # came from different products or one was mis-parsed.
    denom == 0 && throw(DomainError(ka,
        "the TOPS Doppler centroid rate is singular here: `ks` equals the azimuth FM rate, so " *
        "`ks/(1 - ks/ka)` divides by zero. These two are far apart in every real product, so they most " *
        "likely do not describe the same acquisition."))

    kt = c.ks / denom
    eta = (Float64(line) - c.center_line) * c.azimuth_time_interval
    eta_ref = c.eta_ref_near - Float64(c.doppler_centroid(rng)) / ka
    d = eta - eta_ref
    return pi * kt * d * d
end
