# Radar path implementation plan

Plan for `RadarCoordinate`: the second of the two forward mappings named in
`src/coordinates.jl`, where a grid point reaches a pixel index through orbit state vectors and a
range–Doppler solve rather than through a map projection.

The projected path is implemented and verified; this document covers only what the radar path adds.
`REFERENCE.md` is the authority on the reference implementation and on the exactness standard, and
gains a radar section as this work lands.

## Reference implementation

| Piece | Local path |
|---|---|
| Algorithm core | `../autoRIFT-v2.1.2/geo_autoRIFT/geogrid/src/geogridRadar.cpp` (1369 lines) |
| Python wrapper | `../autoRIFT-v2.1.2/geo_autoRIFT/geogrid/GeogridRadar.py` (470 lines) |
| Production driver | `../hyp3-autorift/src/hyp3_autorift/vend/testGeogrid.py:427-488` |

`geogridRadar.cpp` delegates its numerics to isce3, which the Julia port has to supply itself:

| isce3 piece | Where the source is | Julia equivalent |
|---|---|---|
| `core::Ellipsoid` — `lonLatToXyz`, `xyzToLonLat` | `$GEOGRID_REF/include/isce3/core/Ellipsoid.h:195-238` | CHUNK-001 |
| `core::Basis(pos, vel)` — geocentric TCN | `$GEOGRID_REF/include/isce3/core/Basis.h:50-58` | CHUNK-001 |
| `core::Orbit` + Hermite interpolation | `$GEOGRID_REF/include/isce3/core/detail/InterpolateOrbit.icc` | CHUNK-002 |
| `geometry::rdr2geo` | not in the conda package; `isce-framework/isce3@develop` `cxx/isce3/geometry/detail/Rdr2Geo.icc` | CHUNK-004 |

where `$GEOGRID_REF` is `/opt/homebrew/Cellar/micromamba/2.5.0_4/envs/geogrid-ref`. Every
header-only piece is installed locally and readable; only `Rdr2Geo.icc` must be fetched from the
isce3 repository, and its algorithm is fully determined by that one file plus `Rdr2Geo.h`
(`threshold = 1e-8`, `maxiter = 25`, `extraiter = 15`).

## The ellipsoid is transcribed, not delegated

The reference hardcodes `a = 6378137.0`, `e2 = 0.0066943799901` — a truncation of WGS84's
`6.69437999014e-3` — and converts ECEF to geodetic by Vermeille's closed form. Geodesy.jl uses
GeographicLib's formulation on the full-precision datum, so it agrees to about 1e-9 and not to the
bit. Since the radar path's Tier A outputs are `std::round` of a quantity computed from these
conversions, and a range index near a `.5` boundary is decided in those last bits, the ellipsoid
must be a direct transcription of `Ellipsoid.h`. It is about 30 lines.

Geodesy.jl earns a place in the *test* environment instead, as an independent implementation to
cross-check the transcription against at 1e-9 — a genuinely different formulation agreeing to
that tolerance is stronger evidence of correct transcription than a fixture alone, since a fixture
cannot distinguish a faithful port from a faithful port of a misread.

## What the radar path shares with the projected path

More than the two-implementations framing suggests. Both reduce, per grid point, to: two axis
vectors expressed in *grid* coordinates, the physical length of a one-pixel step along each, a
surface normal, and a nominal spacing pair — which is exactly `PointGeometry`.

`geogridRadar.cpp` does not leave the line of sight in ECEF. It steps one range pixel along the
ECEF look vector, converts back to lon/lat, then through `invTrans` into DEM coordinates, and
differences against the grid point (`:975-996`). The along-track vector is obtained the same way
from the range–Doppler solution at `tline + 1/prf` (`:1000-1085`). So both axis vectors, and both
lengths, are already grid-coordinate quantities with the same meaning as the projected path's.

The consequence for `src/kernel/outputs.jl`: it is reusable nearly verbatim, once two things are
loosened.

*Nominal spacing becomes a value.* `offset_to_velocity` and `scale_factors` currently read
`c.spacing` off the `ProjectedCoordinate`. Radar's equivalents are `dr` for range and
`norm(targXYZ - xyz)` for azimuth (`:1180-1188`) — and the azimuth one is *per point*, where the
projected path's is constant. Passing the pair in as an argument covers both.

*The operator gains a third band.* Radar writes `window_rdr_off2vel_[xy]_vec.tif` with three bands
(`:465`, versus two at `geogridOptical.cpp:461`). Band 3 converts a range or azimuth pixel
displacement directly to a range or azimuth velocity: `dr / dt * yr` and
`norm(da) / dt * yr` (`:1185-1187`). `PairGeometry` needs `off2vx_dr` and `off2vy_dr`, left at
their sentinel on the projected path.

## What is genuinely new

- **Orbit interpolation.** Cubic Hermite over the four state vectors bracketing the target time,
  on a `Linspace` time axis. Self-contained in this package (see *Relationship to SAR.jl*).
- **`geo2rdr`** — the grid point's azimuth time and slant range, by Newton's method on
  `f(t) = (target − sat(t)) · v(t)`, the zero-Doppler condition. `:944-969`.
- **`rdr2geo`** — the inverse, needed for the footprint and the incidence angle. A separate
  fixed-point iteration on target height. `Rdr2Geo.icc`.
- **The range–Doppler solve** at `tline + 1/prf`, giving the along-track axis vector. `:1038-1071`.
  Structurally `rdr2geo` against a constant-height DEM, but written out inline in
  `geogridRadar.cpp` with its own iteration count and no convergence test, so it is transcribed
  separately rather than routed through CHUNK-004.
- **Pixel sizes come from geometry, not from a geotransform.** `grd_res = dr / sin(incidence)` and
  `azm_res = norm(satvmid) / prf` (`:684-686`) replace `X_res`/`Y_res`, and feed the chip-size
  conversion.

## No coregistration on the radar path

`testGeogrid.py:427-470` populates every radar parameter from the *reference* scene's `info` and
touches the secondary scene only for `repeatTime = info1.sensingStart - info.sensingStart`. There
is no radar analogue of `coregister`: the radar grid is image 1's outright.

**`CoregisteredPair` widens** to hold any `AbstractImageCoordinate`. The type's job is a pair of
acquisitions plus their time separation, and the coordinate system is a parameter of that rather than
part of its definition. Coregistration is what *produces* a `ProjectedCoordinate` when two images
differ; it is not what the pair type is for. A radar pair is constructed directly, since image 1's
grid is the answer.

The same is true on the projected path, which is what makes the widening a correction rather than an
accommodation for radar. Two images already sharing a grid need no coregistration either — the caller
passes a view of the overlap and the `ProjectedCoordinate` describes it directly. So `coregister` is
one way to obtain a coordinate, not a precondition of the pair type, and the current coupling
overstates what the type requires. `coregister` keeps its role for the case that needs it: two
footprints whose overlap has to be computed.

This keeps one carrier and one `dt` convention across both paths, so `pairgeometry` and
`pairgeometry_blocked` need no radar-specific entry point.

What CHUNK-005 has to settle is `reference_offset` and `secondary_offset`. They are where each image's
window starts inside a *computed* overlap, so they mean something only when `coregister` produced the
coordinate. A radar pair has no such offsets, and neither does a projected pair built from a view the
caller already sliced. `(0, 0)` is the honest value in both cases — the coordinate *is* the window —
but that is a default worth stating in the constructor rather than leaving a reader to infer that two
zeros mean "not applicable."

## Reproduced quirks specific to the radar path

Each is a reference behavior that looks like a defect and is matched anyway, and each gets a
comment at its site plus an entry in `REFERENCE.md`.

- **`geo2rdr` runs a fixed 51 iterations with no convergence test** (`:944`). Because the derivative
  drops the acceleration term the iteration is linear, not quadratic — a measured factor of 10.9 per
  step, so about 15 iterations from a scene-center start, then oscillation at 1e-12 s. Reproduce the
  count: lowering it changes the answer and raising it changes the last bits. See *Findings*.

- **`rngpix` is one iteration stale.** It is computed at the top of the loop from the *previous*
  iteration's satellite position (`:954-957`), so the slant range that produces the range index at
  `:971` does not correspond to the converged time at `:972`. Reproduce the staleness. Since the
  final iterations are effectively no-ops this is a sub-ULP effect in practice, but it is the
  reference's arithmetic.

- **The range–Doppler loop runs a fixed 10 iterations and discards its residual.** `rdiff` is
  computed at `:1070` and never read; there is no threshold test. Contrast isce3's `rdr2geo`,
  which breaks on `dr < 1e-8`.

- **The range–Doppler loop's `alpha` differs from isce3's.** `geogridRadar.cpp:1046` writes
  `alpha = dopfact - gamma * dot(nhat,vhat) / dot(vhat,that)`; `Rdr2Geo.icc` writes
  `alpha = (dopfact - gamma * ndotv) / vdott`. The two agree only because `dopfact` is hardcoded to
  `0.0` at `:1009`. Transcribe geogridRadar's form on the kernel path — it is what production
  runs — and isce3's in CHUNK-004, which is a port of isce3.

- **The incidence angle is one scalar for the whole scene.** `GeogridRadar.py:253-297` evaluates
  `rdr2geo` at scene center for `zrange = [-200, 4000]` and takes the mean of the two incidence
  angles. Every grid point's ground-range pixel size derives from that one number, so it is scene
  metadata, not per-point geometry.

- **`nodata` is left uninitialized when `vxname` is empty and `csminxname` is set** (`:508-512`).
  Refuse that configuration rather than reproduce it — reading an uninitialized double is not a
  behavior with a value to match. This joins the deliberate divergences.

- **`acos` in `cross_check`.** On the projected path this sits at 82–90°, so the one-ULP
  openlibm-versus-system-libm difference cannot flip the `> 1.0` gate. Radar geometry is oblique
  and can approach it. CHUNK-008 measures where `cross_check` actually lands on the fixture cases
  and records it; if any point sits within a ULP of the gate, that is a documented
  platform-dependent output rather than something to paper over.

## Relationship to SAR.jl

`../SAR.jl/PLAN-slc-reader.md` plans an `Orbit` type, Hermite interpolation, and NISAR/Sentinel-1
metadata ingest. This work does not depend on it: the orbit code here is a bit-exact transcription
of `InterpolateOrbit.icc`, which is a stronger constraint than SAR.jl's reader needs to meet, and
coupling this package's release to an unwritten reader would block it.

The boundary that keeps extraction cheap later: **no orbit file parsing in this package.**
`geogridRadar.cpp:333-427` scrapes an orbit EOF with string matching on `<UTC>`/`<X unit>` tags;
that is metadata ingest and belongs in SAR.jl. Here, an `Orbit` is constructed from state vectors
the caller supplies. The fixture generator writes the EOF for the reference to read, and the Julia
side takes the same state vectors from JSON.

## Fixtures

The committed-fixture, no-Python, no-network pattern of `test/reference/` carries over, and the
radar path is *easier* to fixture than the projected one: `geogridRadar` takes every radar
parameter as a scalar and reads the orbit from a file path it parses itself. Nothing requires a
real SLC. A synthetic circular orbit written as an EOF, plus a small DEM window and matching
parameter rasters, exercises the compiled kernel end to end.

Generators, each run under `micromamba run -n geogrid-ref python ...` and each recording the
versions it ran against, as the existing ones do:

| Generator | Exercises | Against |
|---|---|---|
| `gen_radar_numerics.py` ✓ | ellipsoid, TCN inputs, Hermite interpolation, `rdr2geo` | `isce3.core`, `isce3.geometry` |
| `gen_geogrid_radar.py` | the whole kernel, all nine outputs | compiled `geogrid.geogridRadar` |

The three numerics generators the plan first listed separately are one file: they share an orbit, an
ellipsoid and a set of radar parameters, and splitting them would mean three copies of that setup
drifting apart. It records every float as a hex literal alongside its decimal form, since
`float.hex()` round-trips exactly and a JSON decimal repr does not reliably, and these are the basis
for bitwise assertions.

`geo2rdr` and the range–Doppler solve have no directly callable reference — they exist only inside
`geogridRadar.cpp` — so they are verified through `gen_geogrid_radar.py`'s whole-kernel outputs and
against `gen_rdr2geo.py` as a round trip: `geo2rdr` then `rdr2geo` must return the starting point.

Cases to cover, chosen so each isolates one thing: same-CRS and cross-CRS grids (the Tier A/Tier B
split turns on this); left and right look; a grid partly outside the radar swath, exercising the
`rgind`/`azind` bounds test at `:1108`; and the input combinations that gate output bands — DEM
only, DEM + slope, and the full set.

## Exactness stance

Aim for bit-identical, and relax only where measurement shows the difference cannot reach an output.
The reason is not that the last bit matters in itself — it is that a bit-identical target forces an
exact translation, which is a stronger correctness constraint than a tolerance and a better starting
point for later optimization. A tolerance can be widened to hide a transcription error; a bitwise
assertion cannot.

So a bound is a result, not a goal. Where one is stated it carries an identified cause and a
measurement of what it costs at the output, and the option of closing it has been considered rather
than skipped. CHUNK-001's `atan2` gap is the open case: the system function reproduces the fixture
exactly on all 12 cases, so this is a live option and not a hypothetical — see *Open questions*.

## Exactness expectation

Tier A — the integer bands — should hold bitwise as it does on the projected path, and for the
same reason: `std::round` absorbs a last-bit difference. Two things could break that and are
therefore what the fixture work is really testing.

*A range index can land on an exact `.5`.* At a 90 m grid over 30 m imagery, every projected-path
grid point does (`REFERENCE.md`, "`std::round` is ties-away-from-zero"). Radar geometry is
oblique, so exact ties are less structured — but `(rngpix - startingRange) / dr` with a round
`startingRange` and `dr` can still produce them, and `cround` already handles ties-away.

*Iteration amplifies input differences.* PROJ is not bit-reproducible across platforms, and the
projected path already shows 2 ULP of projected easting emerging as ~1e4 ULP in an axis unit
vector. Here that difference enters a 51-iteration Newton solve. Newton is self-correcting — it
converges to the same fixed point from a perturbed start — so the expectation is that this does
*not* amplify, but it is an expectation to verify per platform rather than assume. CHUNK-008
reports the observed maximum on every run, as the projected path does.

Tier B — the Float64 bands — cross-CRS: relative bound, same 1e-7 as the projected path.
Same-CRS: bitwise is the projected path's standard, and CHUNK-001 through 004 have established it is
*not* reachable here. The ellipsoid inverse alone carries a 1-ULP `atan2` difference, which is not a
consequence of the 51-iteration solve and cannot be removed by any choice this port makes. So the
same-CRS radar floats get a stated bound with an identified cause, and the measured 1.9e-9 m of ground
position is the margin Tier A has to survive.

## Chunks

Ordered so each is verifiable against a fixture before the next depends on it. CHUNK-001 through
004 are self-contained numerics with their own reference oracles; only from 005 does the port touch
existing package code.

**Status: 001–005 complete.** `src/radar/` holds five files; `test/radar_numerics.jl` and
`test/radar_coordinate.jl` hold 1034 assertions against two fixtures generated from isce3 0.25.12. The
full suite passes at 6889 and the docs build clean. What measurement changed from the plan as written
is recorded in *Findings* below and in `REFERENCE.md`.

`RadarCoordinate` is constructible and supplies a footprint and the ground pixel sizes;
`CoregisteredPair` is parameterized on the coordinate type. What 006 and 007 still need is the
per-point kernel — there is no radar `pointgeometry`, so `pairgeometry` cannot take a radar pair yet.

### CHUNK-001: ellipsoid and TCN basis — done

`src/radar/ellipsoid.jl`. `lonLatToXyz`, `xyzToLonLat` (Vermeille), `geodetic_tcn`, and the
reference's `WGS84_GEOGRID` constants. Transcribed from `Ellipsoid.h:195-238` and `Basis.h:50-58`.

Verify: `gen_ellipsoid.py` bitwise; Geodesy.jl cross-check at 1e-9; round-trip lon/lat → ECEF →
lon/lat.

### CHUNK-002: orbit and Hermite interpolation — done

`src/radar/orbit.jl`. An `Orbit` over a uniform time axis, with `interpolate` returning position
and velocity. Transcribed from `InterpolateOrbit.icc:11-108`, Hermite only — production never
selects Legendre. The `idx = search(t) - 2` clamp to `[0, size-4]` is load-bearing at the domain
edges and is transcribed as written.

Border handling: the reference passes `OrbitInterpBorderMode::Error`, so out-of-domain throws.

Verify: `gen_orbit.py` bitwise, at state-vector nodes and between them, and at both domain edges
where the index clamp engages.

### CHUNK-003: geo2rdr — done

`src/radar/geo2rdr.jl`. The 51-iteration zero-Doppler Newton solve, returning azimuth time, slant
range, and the ECEF look vector. Transcribed from `:944-971`, staleness of `rngpix` included and
commented at its site.

Two parallel time variables, as the reference carries them: `tline` in seconds since the
acquisition day's midnight, feeding `azind`; `tlined` in seconds since the orbit's reference epoch,
feeding the interpolator. Both updated by the same Newton step.

Verify: round trip against CHUNK-004 once that lands; until then, that the residual
`(target − sat) · v` reaches zero to machine precision, and that iterations 5 through 51 do not
move the result by more than a ULP.

### CHUNK-004: rdr2geo — done

`src/radar/rdr2geo.jl`. Port of `Rdr2Geo.icc:44-160` with a constant-height DEM, which is all the
reference's callers use (`DEMInterpolator(zz)`). `threshold = 1e-8`, `maxiter = 25`,
`extraiter = 15`, near-nadir break, and isce3's `alpha` form.

Verified to 1.9e-9 m of ground position rather than bitwise, for the reasons in *Findings*; round
trip with CHUNK-003 closes to 1e-9 s and 1e-6 m across all 48 cases.

## Findings

Four things measurement settled differently from how this plan first stated them. Each is recorded in
`REFERENCE.md` with its evidence; the corrections matter because three of them were assumptions this
plan would otherwise have carried into CHUNK-007 and CHUNK-008.

**`geo2rdr` converges linearly, not quadratically.** This plan said "zero-Doppler Newton on a
near-linear function converges in three or four" iterations. It does not: `fnprime` drops the
acceleration term, making it a fixed-point iteration with a slightly wrong slope, and the error falls
by a measured factor of 10.9 per step. From a scene-center start it needs about 15 iterations, and at
iteration 4 is still 4 ms off. So 51 is a defensible choice rather than obvious overkill — and the
conclusion the plan drew from the wrong premise still holds, for a better reason.

**The radar numerics are not bitwise, and the causes are upstream of this port.** The plan expected
bitwise agreement absent PROJ. Two mechanisms prevent it: `atan2` differs by 1 ULP between openlibm
and the platform libm, and floating-point contraction reaches the Hermite velocity through a
cancellation in `g0`. Both were confirmed by independent implementation rather than assumed —
the system `atan2` via `ccall` reproduces the fixture exactly, and a NumPy transcription of the
Hermite formula reproduces *this package's* answer rather than isce3's. Bounded at 1.9e-9 m of ground
position, 8e-10 of a range sample.

This also revises the plan's expectation for CHUNK-008. Tier B same-CRS bitwise is now unlikely on the
radar path, and not because of the 51-iteration solve the plan worried about: the ellipsoid inverse
alone forecloses it. Tier A should still hold — 1.9e-9 m cannot move a `std::round` boundary — but
that is now a prediction with a measured margin behind it.

**The two time scales are preserved to rounding, not exactly.** Subtracting the same increment from
two floats of different magnitude rounds each to its own exponent. The drift is an ULP of the times,
under 1e-8 of an azimuth line over 51 iterations, and zero when the offset is zero — the production
case. The plan's claim of exact preservation was wrong in general and right where it matters.

**The two time scales are two clocks, not two instants — there is no offset quirk.** An earlier
reading of this plan claimed `tline` and `tlined` start one pulse interval apart, and that CHUNK-005
would have to reproduce the discrepancy. That was a misreading, corrected here because acting on it
would have introduced a one-line azimuth error.

`setAzimuthParameters` (`bindings/geogridRadarmodule.cpp:150-160`) receives the `aztime` computed at
`GeogridRadar.py:328-330` as **seconds since midnight** of the acquisition day, and stores it as
`sensingStart`. So `tmid = sensingStart + 0.5 * nLines / prf` (`:328`) is on the seconds-since-midnight
clock — the one `azind` is measured against (`:972`). Meanwhile `tmids` (`GeogridRadar.py:347`) is an
absolute UTC timestamp, parsed to `secondsSinceEpoch()` (`:432`), and initializes `tlined` only: the
orbit's clock.

The two are therefore the same instant expressed on two clocks, and `tline - tlined` is a constant
epoch offset. Comparing `0.5 * nLines / prf` against `(floor(nLines / 2) - 1) / prf` compares
quantities in different coordinate systems and means nothing. CHUNK-005 reproduces both initialization
expressions verbatim; there is no discrepancy to carry.

### CHUNK-005: RadarCoordinate, footprint, incidence angle — done

`src/radar/coordinate.jl`. Replaces the throwing `RadarCoordinate` in `src/coordinates.jl` with the
real type: `starting_range`, `dr`, `sensing_start`, `prf`, `nlines`, `nsamples`, `look_side`,
`wavelength`, `orbit`, `incidence_angle`.

`footprint_bounds(transform, ::RadarCoordinate)` and `incidence_angle(::RadarCoordinate)`, ported
from `GeogridRadar.py:140-297`. The 160-point footprint sample — 21 range values × 2 heights on
the first and last range lines, plus the two range edges of 19 intermediate lines — is transcribed
as written; a different sampling would move the bounding box and shift the grid window.

`xsize`/`ysize` return `grd_res`/`azm_res` (`:684-686`), so the chip-size conversion needs no
change.

Decides the `CoregisteredPair` question above.

Verified against `gen_radar_coordinate.py`, which runs `determineBbox` and `getIncidenceAngle` as the
reference defines them: bounding box and incidence angle to 1e-12 relative, and the grid window the
box produces bitwise at three grid spacings. The box cannot be bitwise while `rdr2geo`'s corners are
not; the window is the quantity a shift would corrupt, and that is exact.

One finding, from a test that failed and was right to: **`azm_res` uses the orbit-clock midpoint, not
the `sensing_start`-clock one.** `geogridRadar.cpp:686` computes it from `satvmid`, interpolated at
`tmidd` (`:435`) — the `tmids` timestamp, i.e. `orbit_midtime`. My first implementation used
`midtime`, which differs by one pulse interval and changed `azm_res` in the last bits. This is the
first place the two-clock distinction has an observable consequence, which is why it is worth having
transcribed both expressions rather than reconciling them.

### CHUNK-006: generalize the shared kernel

No new behavior — make `outputs.jl` serve both paths, verified by the existing suite staying
bitwise.

`offset_to_velocity` and `scale_factors` take a nominal spacing pair as an argument rather than
reading `ProjectedCoordinate`. `PairGeometry` gains `off2vx_dr` and `off2vy_dr`, and
`REFERENCE_FILES` maps the off2vel files to three bands each. The projected path leaves the new
fields at their sentinel, so its GeoTIFF output must stay two-band — the reference writes two there
and a downstream reader would break on three.

Verify: every existing test unchanged and still bitwise. This chunk is a refactor and any output
difference is a defect.

### CHUNK-007: the radar per-point kernel

`src/radar/geometry.jl`. `pointgeometry(::RadarCoordinate, ...)` returning the same
`PointGeometry` the projected path produces, plus the per-point azimuth nominal spacing.

Composes CHUNK-003 for time and range, then the range–Doppler solve at `tline + 1/prf` for the
along-track axis vector (`:1000-1085`, its own 10-iteration transcription), then the two
grid-coordinate axis vectors via the ECEF → lon/lat → `invTrans` route (`:975-996`).

`pixel_index` becomes `(rgind, azind)` from `:971-972` — already-rounded values, unlike the
projected path where rounding happens inside `pixel_index`. The bounds test at `:1108` tests
`rgind > nPixels - 1 | rgind < 0` on integers, where the projected path tests pre-conversion
Float64.

Verify: through CHUNK-008. This chunk has no independent oracle.

### CHUNK-008: driver dispatch and whole-kernel fixture

`_fill_geometry!` dispatches on `AbstractImageCoordinate`. Since CHUNK-006 moved the coordinate
system's contribution behind `pointgeometry` and a spacing pair, the loop body should be shared;
where it cannot be — the integer-versus-Float64 bounds test, band 3 — the difference is a small
method, not a second copy of the loop.

`gen_geogrid_radar.py` and `test/radar_geogrid.jl`: all nine outputs, every case listed above,
Tier A bitwise and Tier B bounded, with the observed maximum reported on every run.

Records the measured `cross_check` range across the fixture cases, per the `acos` concern above.

### CHUNK-009: blocked and threaded

`pairgeometry_blocked` for radar. Points remain independent, so this is threading only — but
`geo2rdr` is far more expensive per point than three PROJ calls, which changes what is worth
caching. Whether `InterpolatedTransform` extends to the radar mapping is an open question, not a
commitment: the mapping is smooth in the grid coordinates, so lattice interpolation should apply,
but the accuracy budget is different because the result feeds an iteration.

Measure first, in `benchmark/`, before adding anything.

### CHUNK-010: extensions and documentation

`ImagePairGeometryRastersExt`: three-band off2vel writing on the radar path, two on the projected.

`ImagePairGeometryAutoRIFTExt`: **the y-sign negation is radar-only, and dispatches.** Both sites are
guarded on `optical_flag == 0` — `testautoRIFT.py:405-407` negates the prior it supplies to the
correlator, `:790-791` negates the displacement it receives back — so this is not shared behavior that
the projected path also performs. `:405`'s own comment gives the reason: *"convert azimuth offset to
vertical offset as used in autoRIFT convention."* Azimuth increases along the track, while autoRIFT's
`Dy` points down a north-up raster.

The projected path's `+y`-down convention comes from the raster geometry itself and needs no
negation. So the extension gets a method per coordinate type, not one shared path with a flag.

`REFERENCE.md` gains a radar section: the quirks above, the exactness table extended with the radar
cases, and the "Not yet implemented" section removed. `docs/src/index.md` and `README.md` drop the
"radar path is not implemented" statements.

## Open questions

- Does Tier A survive on the radar path? The margin is now known — 1.9e-9 m of ground position
  against a 2.33 m range sample — so the question is whether any fixture point sits within 1e-9 of a
  `std::round` boundary. CHUNK-008 measures.
- Does `cross_check` ever approach its `> 1.0` gate on real radar geometry? CHUNK-008 measures.
- Is a lattice-interpolated radar mapping worth its accuracy cost? CHUNK-009, after measurement.

Resolved since the plan was written:

- **The `atan2` gap stays open, and CI settled why.** The question was whether `ccall`ing the platform
  libm would close it. It would not: `atan2` is itself platform-dependent. The fixture's angles come
  from aarch64 macOS's libm, and on x86-64 Linux and Windows the system function agrees with openlibm
  instead — so an assertion that the system libm matches the fixture passes on macOS and fails on both
  others, which is exactly what the first CI run showed. There is no single reference last bit to
  match. Both implementations are faithful and neither is correctly rounded against a 256-bit
  evaluation, so openlibm is kept for the one property that is invariant: the same value everywhere.
- `CoregisteredPair` widens to hold any `AbstractImageCoordinate` — see *No coregistration on the
  radar path*.
- The y-sign negation is radar-only and dispatches — see CHUNK-010.
- There is no `tline`/`tlined` offset quirk; they are two clocks, not two instants — see *Findings*.
- Tier B same-CRS bitwise is not reachable on the radar path, and the cause is the ellipsoid inverse
  rather than the 51-iteration solve.
