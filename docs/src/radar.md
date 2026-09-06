# Radar geometry

Where a projected image reaches a pixel index through a coordinate transform and an affine
calculation, a radar image reaches one through orbit state vectors and a range–Doppler solve. This
page documents that machinery: the reference ellipsoid, orbit interpolation, and the two solves that
convert between ground and radar coordinates.

## Status

Complete and verified against the compiled reference. [`pairgeometry`](@ref) takes a radar pair and
produces all nine outputs, checked over eight fixture cases covering both look sides, a grid larger
than the swath, scattered input nodata, and each combination of inputs that gates an output band.

The numerics beneath it are verified against isce3 — the same objects NASA JPL's `geogridRadar.cpp`
links against.

Two float bands agree to 1.07e-4 relative rather than bitwise: bands 2 and 3 of the off2vel files,
which divide by the along-track step. `REFERENCE.md` records the nine hypotheses eliminated in
attributing that to the compiled kernel rather than to this port — including that two independent
reproductions of the reference's algorithm agree with each other to 4e-6 lines while both sit 0.0013
from the kernel.

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

## The two iteration counts assume SAR geometry

Both solves run a fixed iteration count with no convergence test, and both use fewer iterations than
the reference: [`GEO2RDR_ITERATIONS`](@ref) is 16 against its 51, and
[`RANGE_DOPPLER_ITERATIONS`](@ref) is 6 against its 10. Together they are worth about 2× of the
per-point radar cost, and they are the two assumptions in this package that depend on a property of
the acquisition rather than of the code.

The two are governed by **different** parameters, so a geometry that is easy for one is not
necessarily easy for the other:

| | governed by | worst case | requirement | count |
|---|---|---|---|---|
| `GEO2RDR_ITERATIONS` | orbital altitude | high orbits | 12 | 16 |
| `RANGE_DOPPLER_ITERATIONS` | target latitude | mid-latitudes, 30–45° | 4 | 6 |

The range-Doppler count is the less exposed of the two: its rate vanishes at the equator and at the
pole, so polar ITS_LIVE grids sit well inside the margin, and it reaches its arithmetic floor at 6 —
further iterations cannot refine the answer, only land on different rounding noise. Altitude and
terrain height each move its requirement by less than one iteration.

The zero-Doppler count is the one to check against an unusual acquisition.

The solve converges linearly at a rate [`geo2rdr_rate`](@ref) gives in closed form, and that rate is
governed by **orbital altitude**. 16 covers every altitude from 350 to 800 km with four iterations of
margin, which spans every operating SAR mission — Sentinel-1 and NISAR, the two this package is
principally for, need 12. Terrain height, incidence angle, look side, grid spacing and repeat interval
are all second order.

Above roughly **1000 km** the requirement climbs past 16 and the constant must rise with it. If you
are processing an acquisition from an unusually high orbit, or if radar geometry disagrees with an
independent reference in a way that looks like a small azimuth-time error, check this first:

```julia
using ImagePairGeometry
using ImagePairGeometry: geo2rdr_iterations_needed, GEO2RDR_ITERATIONS

# satpos at scene center, target at the far swath edge, half the scene length as the guess error.
needed = geo2rdr_iterations_needed(satpos, target; guess_error = 0.5 * nlines / prf)
needed <= GEO2RDR_ITERATIONS || @warn "orbit needs more iterations than the fixed count" needed
```

`REFERENCE.md` records the mission-by-mission measurements behind the choice, what the fixture does
and does not constrain, and the two things the study did not cover.

## Trading blocking invariance for speed

[`WarmStart`](@ref) starts each zero-Doppler solve from the previous grid point's answer instead of the
scene midpoint. Adjacent points are a few hundredths of an azimuth line apart, so 8 iterations suffice
where a cold start needs 16 — about **1.25×** on the radar path:

```julia
params = GeometryParams(zero_doppler_start = WarmStart())
r = pairgeometry(grid, pair, inputs; transform = tf, window = win, params)
```

The cost is that results become order-dependent, so a blocked or threaded run no longer matches an
unblocked one bit for bit. Integer bands are unaffected — no rounded index moves — but the float bands
differ by up to 5e-7 relative, well inside the 2e-4 they are held to against isce3. If you need
blocked and unblocked runs to agree exactly, keep the default `SceneCenterStart()`.

Grid spacing does not change what the warm start needs: measured from 120 m to 10 km the divergence is
flat and no index moves, so a **sparse pass for later interpolation** onto a finer grid is as safe as
a dense one.

## Trading the interpolant's exactness for speed

[`chebyshev_orbit`](@ref) tabulates the orbit interpolant per bracket as a Chebyshev series and
evaluates it by Clenshaw summation, instead of rebuilding the Hermite weights on each of the seventeen
interpolations a grid point costs — about **1.3×** on a radar window:

```julia
orbit = chebyshev_orbit(Orbit(; time = t, position = pos, velocity = vel))
coord = RadarCoordinate(; orbit, starting_range, dr, sensing_start, prf,
                        nsamples, nlines, look_side, wavelength, incidence_angle)
```

The option is carried by the orbit's type, so `interpolate` dispatches on it and nothing else in the
pipeline changes. `geo2rdr` is 37% of a point, which caps any interpolator change near 1.6× of a point
however fast interpolation becomes.

Every integer band stays bitwise and blocking invariance is kept — the interpolant is still a pure
function of time, so points remain independent, which is the difference from [`WarmStart`](@ref). What
is given up is the interpolant's own bitwise position agreement with isce3: position moves by up to
1.2e-8 m, which reaches an output as roughly 1e-9 of a pixel and leaves the float bands within 6.8e-9
relative. The default `Orbit` is unchanged, and `REFERENCE.md` records why the strict path stays the
default despite the negligible magnitude.

## API

```@autodocs
Modules = [ImagePairGeometry]
Order = [:type, :constant, :function]
Pages = ["radar/ellipsoid.jl", "radar/orbit.jl", "radar/geo2rdr.jl", "radar/rdr2geo.jl",
         "radar/coordinate.jl", "radar/geometry.jl"]
```
