# Radar geometry

Where a projected image reaches a pixel index through a coordinate transform and an affine
calculation, a radar image reaches one through orbit state vectors and a range–Doppler solve. This
page documents that machinery: the reference ellipsoid, orbit interpolation, and the two solves that
convert between ground and radar coordinates.

## Status

The numerics are implemented and verified against isce3 — the same objects NASA JPL's
`geogridRadar.cpp` links against. [`RadarCoordinate`](@ref) is constructible, and supplies a
footprint and the ground pixel sizes.

What is not built yet is the per-point kernel: there is no radar method of `pointgeometry`, so
[`pairgeometry`](@ref) cannot yet take a radar pair. `RADAR_PLAN.md` in the repository tracks the
remaining work.

## Why this is transcribed rather than delegated

The reference hardcodes `a = 6378137.0` and `e2 = 0.0066943799901` — the latter a truncation of
WGS84's `6.69437999014e-3` at eight significant digits — and converts ECEF to geodetic by Vermeille's
2002 closed form. A geodesy library uses the full-precision datum and a different formulation, so it
agrees to about 1e-9 and not to the last bit. Since the radar path's integer outputs are `std::round`
of quantities computed through these conversions, those last bits decide a range index sitting near a
rounding boundary.

`REFERENCE.md` records the measured agreement, band by band, and the two mechanisms that prevent it
being bitwise throughout: `atan2` differs by one ULP between openlibm and the platform libm, and
floating-point contraction reaches the Hermite velocity through a cancellation. The bound that
carries the weight is stated in meters — 1.9e-9 m of ground position, against a range sample of about
2.3 m.

## Building a coordinate

The incidence angle is computed first and passed in, because the reference computes it first
(`testGeogrid.py:487-488`) and the ground range spacing depends on it:

```julia
ia = incidence_angle(; orbit, starting_range = 8.0e5, dr = 2.3295621147,
                     sensing_start = 300.0, prf = 486.4863103,
                     nsamples = 10000, nlines = 8000,
                     look_side = LookRight, wavelength = 0.05546576)

coord = RadarCoordinate(; orbit, incidence_angle = ia, starting_range = 8.0e5,
                        dr = 2.3295621147, sensing_start = 300.0, prf = 486.4863103,
                        nsamples = 10000, nlines = 8000,
                        look_side = LookRight, wavelength = 0.05546576)

pair = CoregisteredPair(coord; dt = 6 * 86400.0)
```

There is no radar [`coregister`](@ref): `testGeogrid.py:427-470` takes every radar parameter from the
reference acquisition and the secondary only for the repeat interval, so the radar grid is image 1's
outright.

## Two time scales

`sensing_start` is seconds since midnight of the acquisition day, and the orbit is on its own epoch.
`orbit_epoch_offset` is the constant between them. Both are carried because the reference carries
both: it measures the azimuth index against the first and interpolates the orbit against the second,
so neither needs a conversion at the point of use.

## API

```@autodocs
Modules = [ImagePairGeometry]
Order = [:type, :constant, :function]
Pages = ["radar/ellipsoid.jl", "radar/orbit.jl", "radar/geo2rdr.jl", "radar/rdr2geo.jl",
         "radar/coordinate.jl", "radar/geometry.jl"]
```
