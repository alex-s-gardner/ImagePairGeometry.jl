module ImagePairGeometrySLCDatasetsExt

# Building this package's radar types from a read SAR product.
#
# The types being constructed are this package's, so the constructors are extended here rather than in
# the reader: `RadarCoordinate(slc)` and `CoregisteredPair(reference, secondary)` are the same
# constructors a caller already knows, taking an acquisition instead of eleven loose numbers.
#
# The core stays free of any IO stack — `SLCDatasets` is a weak dependency, so nothing is loaded until a
# caller loads it themselves.
#
# Three things the conversion checks rather than assumes, each of which would otherwise produce a
# plausible wrong answer:
#
# Uniform spacing. `Orbit` interpolates against a uniform axis. Products have supplied uniform state
# vectors everywhere measured, but a product that did not would otherwise be reported by `Orbit`'s own
# error, which names state vector indices rather than the product.
#
# The two clocks. `orbit_epoch_offset` is what `RadarCoordinate` adds to a `sensing_start`-scale time to
# reach the orbit's scale. A `RadarGeometry` reports the azimuth times and the state vector times against
# *one* epoch — that is asserted below — so on this path the two scales already coincide and the offset is
# zero. It is not the epoch's time of day: adding that would move every solve off the orbit by hours,
# which for NISAR is invisible because its epoch is midnight and the two are equal.
#
# Coverage. A solve at a time the orbit does not bracket extrapolates rather than failing, so the
# bracket is checked before the coordinate is built.

using ImagePairGeometry: ImagePairGeometry, Orbit, RadarCoordinate, CoregisteredPair,
                         incidence_angle, chebyshev_orbit, LookLeft, LookRight
using SLCDatasets: SLCDatasets, SLC, StateVectors, orbit, repeat_interval, pixels

# The two look-side enums are distinct types with the same meaning; neither package imports the other's.
_look(side) = side == SLCDatasets.LookLeft ? LookLeft : LookRight

"""
    Orbit(sv::SLCDatasets.StateVectors) -> Orbit

A product's state vectors as an interpolating orbit.

Throws if the times are not uniformly spaced, which the interpolant requires, or if there are too few to
interpolate between.
"""
function ImagePairGeometry.Orbit(sv::StateVectors)
    n = length(sv.time)
    n >= 4 || throw(ArgumentError(
        "an interpolating orbit needs at least 4 state vectors, but the product supplies $n"))
    return Orbit(; time = sv.time, position = sv.position, velocity = sv.velocity)
end

"""
    RadarCoordinate(s::SLCDatasets.SLC; zrange = nothing, chebyshev = false) -> RadarCoordinate

The acquisition as a radar coordinate.

The scene-center incidence angle is computed here because the type stores it: the reference computes it
before running the geometry (`testGeogrid.py:487-488`), so it is an input rather than something derived
on demand.

`chebyshev` swaps the orbit interpolant for [`chebyshev_orbit`](@ref), which is faster and not bitwise
identical to the default. `zrange` overrides the elevation pair the incidence angle is averaged over.

# Examples

```julia
using ImagePairGeometry, SLCDatasets

coord = RadarCoordinate(open_slc("NISAR_L1_PR_RSLC_....h5"))
```
"""
function ImagePairGeometry.RadarCoordinate(s::SLC; zrange = nothing, chebyshev::Bool = false)
    g = s.geometry
    sv = orbit(s)

    g.epoch == sv.epoch || throw(ArgumentError(
        "the azimuth times are measured against $(g.epoch) but the state vectors against " *
        "$(sv.epoch). Converting between them is not implemented, because every product measured " *
        "puts both on one epoch and a product that does not may differ in more than this"))

    # An out-of-range solve extrapolates rather than failing, so a product whose orbit does not span
    # its own acquisition is refused here.
    first(sv.time) <= g.sensing_start && g.sensing_stop <= last(sv.time) || throw(ArgumentError(
        "the state vectors span $(first(sv.time))–$(last(sv.time)) s but the acquisition runs " *
        "$(g.sensing_start)–$(g.sensing_stop) s, so the orbit does not cover it"))

    orb = Orbit(sv)
    chebyshev && (orb = chebyshev_orbit(orb))

    kwargs = (; orbit = orb, starting_range = g.starting_range, dr = g.range_pixel_spacing,
              sensing_start = g.sensing_start, prf = g.prf, nsamples = g.nsamples,
              nlines = g.nlines, look_side = _look(g.look_side), wavelength = g.wavelength,
              # Zero, not the epoch's time of day: `RadarGeometry` puts both clocks on one epoch,
              # asserted above, so a `sensing_start`-scale time is already an orbit-scale time.
              orbit_epoch_offset = 0.0)
    ia = zrange === nothing ? incidence_angle(; kwargs...) : incidence_angle(; kwargs..., zrange)
    return RadarCoordinate(; kwargs..., incidence_angle = ia)
end

"""
    CoregisteredPair(reference::SLCDatasets.SLC, secondary::SLCDatasets.SLC; kwargs...)

The two acquisitions as a pair: the reference's geometry, and the interval between them.

Radar geometry comes from the reference alone — `testGeogrid.py:427-470` takes every radar parameter
from image 1 and the secondary only for the repeat interval — so `secondary` contributes its sensing
time and nothing else. `kwargs` are forwarded to `RadarCoordinate`.

There is no radar [`coregister`](@ref) for the same reason: there is no overlap to compute.

# Examples

```julia
pair = CoregisteredPair(open_slc(url1), open_slc(url2))
pair.dt / 86400   # the repeat interval in days
```
"""
function ImagePairGeometry.CoregisteredPair(reference::SLC, secondary::SLC; kwargs...)
    dt = repeat_interval(reference, secondary)
    dt > 0 || throw(ArgumentError(
        "the secondary acquisition starts $(-dt) s before the reference, so the interval is not " *
        "positive; pass them in acquisition order"))
    # Both coordinates, each self-consistent on its own product's clock. `pixel_offset` needs the
    # secondary's orbit, range origin, range spacing and PRF; the geometry outputs do not read it.
    return CoregisteredPair(RadarCoordinate(reference; kwargs...); dt,
                            secondary = RadarCoordinate(secondary; kwargs...))
end

"""
    TOPSCarrier(d::SLCDatasets.DerampParameters, orbit::Orbit; epoch::DateTime) -> TOPSCarrier

One burst's azimuth carrier, from the annotation and an orbit.

`SLCDatasets.deramp_parameters` supplies every term but the along-track speed, which needs the trajectory
interpolated at the burst's mid-time — `orbit` is where that comes from, and why this is not a method on the
reader.

`epoch` is the instant the orbit's times are measured from, which is what reduces the burst's mid-time to the
orbit's own scale. A product's `geometry.epoch` is it; the orbit's `epoch` must equal it, which the caller of
this checks.

The antenna steering rate is degrees per second in the annotation and radians per second in the arithmetic,
so the conversion happens here.
"""
function ImagePairGeometry.TOPSCarrier(d::SLCDatasets.DerampParameters, orbit::Orbit; epoch)
    # `epoch` is a `DateTime`, but this extension may not name `Dates` as a dependency — an extension sees
    # only its parent's — so the type is left to `UtcTime`'s own constructor to enforce.
    t_mid = SLCDatasets.seconds_between(SLCDatasets.UtcTime(epoch, 0.0), d.burst_mid)
    _, v = ImagePairGeometry.interpolate(orbit, t_mid)
    vs = sqrt(v[1] * v[1] + v[2] * v[2] + v[3] * v[3])

    steer = deg2rad(d.azimuth_steering_rate)
    ks = 2 * vs * steer / d.wavelength

    ka, fdc = d.azimuth_fm_rate, d.doppler_centroid
    near = Float64(fdc(d.starting_range))
    ka_near = Float64(ka(d.starting_range))
    ka_near == 0 && throw(ArgumentError(
        "the azimuth FM rate is zero at the near range ($(d.starting_range) m), so the carrier's " *
        "reference time is undefined. The polynomial is most likely mis-parsed."))

    return ImagePairGeometry.TOPSCarrier(;
        azimuth_fm_rate = ka, doppler_centroid = fdc, ks,
        eta_ref_near = near / ka_near,
        starting_range = d.starting_range,
        range_pixel_spacing = d.range_pixel_spacing,
        azimuth_time_interval = d.azimuth_time_interval,
        # The reference indexes lines from zero and takes `n_lines // 2`; one-based, the centre is one past
        # that. An off-by-one here breaks the ramp's symmetry about the burst centre rather than its
        # magnitude, so it would survive a magnitude check.
        center_line = d.lines_per_burst ÷ 2 + 1)
end

"""
    ResampledSLC(secondary::SLCDatasets.SLC, offset; amplitude_only = false, kwargs...)

The secondary acquisition's samples on the reference's grid, read from the product.

Takes the acquisition where the core takes a matrix, so a caller does not reach for `pixels` and the
samples and the mode travel together. `offset` and `kwargs` are as
[`ResampledSLC`](@ref) documents them.

# TOPS deramping

Sentinel-1 IW steers the antenna in azimuth across each burst, putting a steep ramp on the azimuth phase.
Interpolating those samples without removing the ramp aliases it, worst at the burst edges, and the result
is a phase-corrupted image that looks entirely valid. So for a TOPS acquisition this builds the carrier from
the product's annotation and hands it to the core, which removes it before interpolating and reapplies it
after.

A single burst has one carrier. A **merged subswath has one per burst** — the ramp is referenced to each
burst's own centre — and this package does not yet resample one: `burst_at` says which burst a line belongs
to, but the carrier a chip needs depends on where its *samples* fall, and a chip straddling a burst seam has
no single answer. So a merge is refused rather than deramped with the wrong reference. Resample the bursts
individually.

A product whose annotation omits the fields the deramp needs is also refused, naming the missing one.

`amplitude_only = true` skips the deramp, because taking the magnitude discards the phase and so is
insensitive to the ramp. That is a real use — amplitude feature tracking on Sentinel-1 is what most of this
pipeline does — and it stays a keyword rather than a default so the choice is written at the call site,
where a reader can see which kind of result they are holding.

This is the only place the carrier can be built. A bare samples matrix carries no record of how it was
collected, so the core's `ResampledSLC` cannot ask; it documents the hazard and this supplies the answer.
"""
function ImagePairGeometry.ResampledSLC(secondary::SLC, offset::AbstractMatrix;
                                        amplitude_only::Bool = false, carrier = nothing, kwargs...)
    # The carrier is settled before the samples are reached. Which phase to remove is a question about the
    # geometry, and answering it first means a product whose annotation cannot support a deramp says so
    # rather than failing later on its bytes — and that a caller reading a product with no samples at all
    # still gets told about the deramp.
    c = _tops_carrier(secondary, amplitude_only, carrier)
    return ImagePairGeometry.ResampledSLC(pixels(secondary), offset; carrier = c, kwargs...)
end

# Which carrier a resample of this acquisition needs, or `nothing`.
function _tops_carrier(secondary::SLC, amplitude_only::Bool, carrier)
    # Not TOPS, or the phase will not be read: nothing to remove, and the result is what it was before any of
    # this existed. An explicit carrier is the caller's own — how a merged subswath or an already-deramped
    # product is handled, since those know something about the samples this cannot infer.
    (!SLCDatasets.is_tops(secondary) || amplitude_only) && return carrier
    carrier === nothing || return carrier

    # A merge carries one ramp per burst, each referenced to its own centre. A chip near a seam reads samples
    # from two bursts, so no single carrier describes it — and deramping the whole image with the first
    # burst's would be wrong by up to half a burst everywhere else. Refused rather than guessed; the caller
    # can resample each burst, or pass its own `carrier` if it knows better.
    secondary.backend isa SLCDatasets.MergedBurstBackend && throw(ArgumentError(
        "this is a merge of TOPS bursts, which carries one azimuth ramp per burst rather than one for " *
        "the image: each is referenced to its own burst's centre, and a chip spanning a seam has no " *
        "single answer. Resample the bursts individually, or pass `carrier` if you have one that covers " *
        "the merged grid. `amplitude_only = true` needs no carrier at all."))

    g = secondary.geometry
    sv = SLCDatasets.orbit(secondary)
    g.epoch == sv.epoch || throw(ArgumentError(
        "the azimuth times are measured against $(g.epoch) but the state vectors against $(sv.epoch); " *
        "building a TOPS carrier needs both on one epoch, since the platform speed is interpolated at " *
        "the burst's mid-time"))

    return ImagePairGeometry.TOPSCarrier(SLCDatasets.deramp_parameters(secondary), Orbit(sv);
                                        epoch = g.epoch)
end

end
