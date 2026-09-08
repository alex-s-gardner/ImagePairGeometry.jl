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
    ResampledSLC(secondary::SLCDatasets.SLC, offset; amplitude_only = false, kwargs...)

The secondary acquisition's samples on the reference's grid, read from the product.

Takes the acquisition where the core takes a matrix, so a caller does not reach for `pixels` and the
samples and the mode travel together. `offset` and `kwargs` are as
[`ResampledSLC`](@ref) documents them.

# TOPS is refused here

Sentinel-1 IW steers the antenna in azimuth across each burst, putting a steep ramp on the azimuth phase.
Interpolating those samples without removing the ramp first aliases it, worst at the burst edges, and the
result is a phase-corrupted image that looks entirely valid. `SLCDatasets` does not yet parse the three
annotation fields a deramp needs — see its `deramp_parameters` — so this refuses rather than producing
one.

`amplitude_only = true` permits it, because taking the magnitude discards the phase and so is insensitive
to the ramp. That is a real use: amplitude feature tracking on Sentinel-1 is what most of this pipeline
does. It is a keyword rather than the default so that the choice is written at the call site, where a
reader can see which kind of result they are holding.

This is the only place the check can be made. A bare samples matrix carries no record of how it was
collected, so the core's `ResampledSLC` cannot ask — it documents the hazard and this enforces it.
"""
function ImagePairGeometry.ResampledSLC(secondary::SLC, offset::AbstractMatrix;
                                        amplitude_only::Bool = false, kwargs...)
    if SLCDatasets.is_tops(secondary) && !amplitude_only
        throw(ArgumentError(
            "this is a TOPS acquisition, whose azimuth phase carries a per-burst ramp that must be " *
            "removed before its complex samples are interpolated and reapplied afterwards. " *
            "SLCDatasets does not yet parse the annotation fields that needs — call " *
            "`SLCDatasets.deramp_parameters` to see which — so resampling the complex samples would " *
            "produce a phase-corrupted image that looks valid. Pass `amplitude_only = true` if the " *
            "magnitudes are all you will read, which is insensitive to the ramp."))
    end
    return ImagePairGeometry.ResampledSLC(pixels(secondary), offset; kwargs...)
end

end
