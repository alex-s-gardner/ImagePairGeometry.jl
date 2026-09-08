# Resampling plan

The kernel computes, per grid point, one pixel index, and hands it to the correlator for **both**
images of the pair. That is correct only where the secondary image sits on the reference image's
grid. Neither path guarantees it, and they fail differently:

*Radar.* The two acquisitions have different orbits, so the same ground point falls at a different
range sample and azimuth line in each. `CoregisteredPair` records this in its own field names — for a
radar pair the secondary contributes `dt` and nothing else (`src/pair.jl:47-48`), and
`RadarCoordinate` says outright that it "describes the *reference* image alone."

*Projected.* Two orthorectified images **are** on one grid — an L1TP product is geolocated and
terrain-corrected, so the pixel index the kernel computes is right for both to the accuracy the producer
delivers. There is nothing here for this plan to compute. What `coregister` does *not* check is whether
the two are on the same CRS, which it cannot: `ImageFootprint` carries none.

So the radar case is a missing computation and the projected case is a missing *check*. The radar step is
missing rather than deliberate: the reference pipeline resamples the secondary SLC onto the reference grid
*before* geogrid runs, so by the time geogrid sees the pair the assumption holds, and nothing in this
repository does that resampling. See *The projected path needs the interface, not a parallax correction*
below for why the projected side gets an interface and a refusal rather than an offset.

For amplitude feature tracking the consequence is bounded on both paths: `abs` discards the phase, so
a misregistration is a translation the correlator can measure, provided its search window reaches. For
complex correlation it is not bounded — the phase inside each chip is wrong, and no search extent
recovers it. Complex data is radar-only, so Path A below is a radar path; Path B serves both.

This plan covers two ways to close the gap. They share a foundation and are useful independently:

| | Path A — resample | Path B — correct after correlation |
|---|---|---|
| Applies to | radar | radar and projected |
| What it produces | image 2's complex samples on image 1's grid | the misregistration, as a prior and a correction |
| Sub-pixel phase | correct | not addressed |
| Cost | a sinc interpolation per output sample | a lattice of solves per pair |
| Needs a DEM | for large relief | for large relief |
| Needs TOPS deramping | yes, on Sentinel-1 IW | no |
| Unblocks | complex correlation, interferometry | amplitude tracking on real pairs, both paths |

Path B is the whole requirement for amplitude tracking and is a much smaller job. Path A is the
requirement for anything that reads the phase. Both rest on the same new operation, so Path B is not
a detour on the way to Path A.

That operation is where the two coordinate systems meet: **one interface, two methods**, so the field,
the lattice, the fit and the correlator wiring are written once and a projected pair gets them for the
cost of a view vector. See *The abstraction* below.

## The measurement that sizes the problem

One S1A/S1A pair, 24 days apart — two exact orbit cycles, so the smallest baseline a repeat pass
offers. Computed from the orbits, with no reliance on scene contrast:

| | mean | spread across scene |
|---|---|---|
| azimuth (line) | −2.8 px | 0.16 px |
| range (sample) | +17.6 px | 4.17 px |

Azimuth is flat to 0.16 px across 12244 lines, so a constant absorbs it. Range is not: it drifts
monotonically 15.6 → 19.7 px across the swath, about 10 m on the ground. The two behave differently
because the same baseline projects differently onto the line of sight at near and far range, where
the incidence angle differs — 0.66 rad at scene center. So range misregistration is a function of
look angle: a ramp, not a constant. A DC shift leaves ±2 px of residual, which on a velocity of tens
of pixels is a systematic bias rather than noise.

Terrain sensitivity is small for this geometry: 0 → 2000 m of elevation moves the range
misregistration by 0.18 px. It is not zero, though. Over 3000 m of relief the height term reaches
0.27 px, which is above the 0.1 px target below, so the height dependence is carried rather than
dropped.

**These are one pair's coefficients, not a general result.** 24 days, S1A-to-S1A, same relative
orbit, near-polar. A different relative orbit or an S1A/S1B pair has a larger baseline and a steeper
ramp. Even the functional form is not promised: first order is what this pair needs, and a wider
baseline may need second. Everything below therefore *computes* the field per pair and *fits to a
residual*, choosing the order rather than assuming one — see CHUNK-005.

## The projected path needs the interface, not a parallax correction

An earlier draft of this plan proposed a residual-parallax correction for the projected path, on the
argument that an ascending/descending pair views the ground from opposite directions and so a DEM error
would displace the two scenes oppositely. **That was wrong as a justification for building anything**,
in two independent ways, and both are worth recording so it is not re-proposed.

*The correction has already been applied.* A Landsat L1TP product is geolocated and terrain-corrected.
The orthorectification has applied `dh · tan(θ)` using the DEM and the view model, and the accuracy
delivered is the residual *after* that. Whatever is left is not a computable function of view geometry,
because the computable part is gone. What remains is the error in the DEM the processor used and in its
own view model — neither reported by the product, neither reconstructible here. The L1GT and L1GS
products in the golden set are the ones to be careful with rather than the L1TP ones.

*And the view geometry barely differs anyway.* Measured from the USGS STAC catalogue across both
cross-path golden pairs and a same-path control, `view:off_nadir` is **0 for all six scenes**. Landsat 7
and 8 are nadir-pointing push-broom instruments with no cross-track steering, so there is no roll to
report and USGS reports none. The ±7.5° quoted in the earlier draft is the *field of view across the
swath*, not a difference between acquisitions: a given ground point sits at a similar view angle in
both scenes of an adjacent-path pair, so the residual does not double. Cross-path Landsat pairs are
geometrically no worse than same-path ones, which is the opposite of what the earlier draft assumed and
removes the motivating case entirely.

So there is no geometric correction to apply here. `dh` is unknowable, and a correction proportional to
an unknowable quantity is a free parameter dressed as geometry. The honest statement is that **two
orthorectified products are coregistered to the accuracy their producer delivers**, and where that is
not good enough the remedy is measurement — a shift estimated from the imagery over stable ground, which
is what a correlator already does and which needs no geometry from here.

What survives from that section is smaller and still worth having:

*The interface, and the refusals under it.* `pixel_offset` is defined for `(Projected, Projected)`
because the *question* is meaningful on that path even when the current answer is zero: the two images
must share a coordinate system, a spacing and a CRS for a pixel shift to describe them at all, and
those are exactly the preconditions `coregister` half-checks today. Stating them, and refusing a pair
that violates them, is worth doing independently of whether any nonzero offset is ever computed.

*A place for a measured shift to live.* If a caller estimates a residual shift over stable ground — by
correlation, or from a published control — the field, lattice, fit and correlator wiring built for the
radar path accept it unchanged. That is a supplied offset rather than a derived one, and the plumbing
already exists once CHUNK-004 through 006 land.

**Consequence for the chunks.** CHUNK-007 shrinks from "compute the parallax correction" to "define the
projected method and its preconditions." No `ViewGeometry`, no `dh` argument, and no `src/parallax.jl`.
The projected method returns a zero offset for a matching pair and refuses a non-matching one, which is
the correct behavior rather than a placeholder — and the CRS check it enables is the part with real
value. The magnitude table and the sensitivity argument are deleted rather than kept as motivation,
since keeping them would leave a reader thinking the correction is merely deferred.

### Reading view geometry: deferred, and where it lives

The geotransform is already read — `image_footprint` and `mapgrid` take origin, spacing and size off the
raster, and the CRS comes from `MapGrid`. Nothing to add there.

View geometry is **not in the raster**. A Landsat `_B8.TIF` carries two metadata items,
`AREA_OR_POINT=Point` and the compression tag, plus the CRS and geotransform; there is no RPC domain and
no angle information. It lives in sibling assets, and the pipeline's inputs do not include them — the
golden runs consume a bare `_B8.TIF`, and `tests/landsat/` holds two of those and no siblings.

Where it is, for whoever needs it next:

| source | what it gives | how to get it |
|---|---|---|
| `_VZA.TIF`, `_VAA.TIF` | per-pixel view zenith and azimuth, as COGs on the scene grid | STAC assets `VZA`/`VAA`, Collection 2 L1 |
| `_ANG.txt` | per-band rational-polynomial satellite vectors in (line, sample, height), plus 54-point ephemeris and `BANDnn_MEAN_SAT_VECTOR` | STAC asset `ANG.txt`; 1509 lines of ODL |
| STAC item properties | `view:off_nadir`, `view:sun_azimuth`, `view:sun_elevation`, `proj:epsg` | `landsatlook.usgs.gov/stac-server`, collection `landsat-c2l1` |
| Sentinel-2 `MTD_TL.xml` | `Viewing_Incidence_Angles_Grids`, per band and detector | inside the `.SAFE` |

The per-pixel `VZA`/`VAA` rasters are the important entry in that table: they make any future correction a
matter of reading two GeoTIFFs rather than writing an ODL parser, and they are already in the shape
`pairgeometry` consumes its other per-point inputs in. Note also that `landsatlook.usgs.gov` file
downloads redirect to a USGS ERS login, and the `usgs-landsat` S3 bucket is requester-pays; the STAC
*catalogue* is open, which is where the numbers above came from.

**Why reading it is deferred rather than started now.** Not difficulty — it is a bounded job. The
blocker is that nothing consumes it, so three design questions have no answer, and each would be settled
by guessing:

- *Which representation?* Mean vector, per-pixel raster, or the polynomials. A correction needs the
  variation across the scene, so not the mean; between the other two the choice depends on an accuracy
  requirement that does not exist yet.
- *Which bands?* ANG carries all 11. Tracking uses one. Whether the type holds one or all is a consumer
  question.
- *How does it meet the DEM?* The polynomials take height as an input, and `pairgeometry` already takes a
  DEM. That coupling is exactly what the dropped parallax argument got wrong, and getting it right
  requires knowing what the consumer does with both.

This package holds itself to bitwise reproducibility and documents every divergence, so a public type
introduced on a guess is a liability even with no users: it invites tests and fixtures that pin the wrong
shape, and those are what make a later change expensive. The cheap and correct thing is to record the
above and start when something needs it. Should a parallax correction turn out to be wanted, the work is
a `Rasters`-extension reader for `VZA`/`VAA` plus a method under `pixel_offset` — additive, and no change
to anything this plan builds.

## The shared foundation: image 1's pixel to image 2's

Everything below needs one operation this package does not have: **given a pixel of image 1, the pixel
of image 2 imaging the same ground point.** That is the general statement, and it is the same question
on both coordinate systems — which is what lets one field type, one lattice and one fit serve both.
What differs is how it is answered:

| | how image 1's pixel becomes a ground point | how that becomes image 2's pixel |
|---|---|---|
| radar | `rdr2geo` against orbit 1 | `geo2rdr` against orbit 2 |
| projected | the grid point itself — both images are terrain-corrected onto it | the same point; the offset is zero unless one is supplied |

### The abstraction

One function, dispatching on the coordinate types, exactly as `pointgeometry` and `footprint_bounds`
already do:

```julia
pixel_offset(pair, x, y, z) -> NTuple{2,Float64}   # (dsamp, dline) or (dcol, drow)
```

`pair` carries both coordinates (CHUNK-001), and the return is **image 2's pixel minus image 1's, in
image 1's axes**, fractional. That signature is the whole interface. Above it, nothing knows which
coordinate system it is on: the lazy field, the lattice, the polynomial fit, the order selection and
the correlator wiring are written once and tested once, against whichever method is cheaper to
construct in a test.

This is the same shape the package already uses for the forward mapping. `pointgeometry` has a
`ProjectedCoordinate` method that is three transform calls and a `RadarCoordinate` method that is three
solves, and `kernel/outputs.jl` consumes the `PointGeometry` both produce without knowing which it got
— the two paths differing "in how the forward mapping is obtained" is the package's existing framing,
in `src/coordinates.jl` and in the module docstring. Misregistration is the same story about the
*secondary* image, so it gets the same treatment rather than a parallel one.

Two things that fall out of doing it this way, both of which are the point:

*Neither method is privileged.* The radar method is expensive and load-bearing and lands first
(CHUNK-002); the projected method is a few multiplications once a view vector exists (CHUNK-007). But
`pixel_offset` is not "the radar operation, generalized later." The interface is defined for both in
CHUNK-001, and the projected method exists from then as a stub that throws naming what it needs — so
the gap is a visible, documented hole rather than a `MethodError` from an interface that only ever
anticipated one path. `src/result.jl` is the model: the band layout is two constants and a
`reference_files` dispatch, chosen so "a caller cannot write the wrong band count for the path it is
on," rather than one layout with a flag added later.

*The two images must be in one coordinate system, and that is a precondition rather than a
convenience.* The offset `pixel_offset` returns is a **pixel shift in image 1's axes**, and a pixel
shift is only a meaningful description of the relationship between two images when both are indexed the
same way. Two `RadarCoordinate`s share a range/azimuth frame; two `ProjectedCoordinate`s on one CRS and
one spacing share a row/column frame. A radar reference against a projected secondary shares neither:
there is no single `(dsamp, dline)` that carries a point from one to the other, because the axes do not
correspond. The correction Path B applies after correlation is a shift of the *correlator's own output*,
which lives in one image's pixel axes, so it inherits the same requirement.

So `pixel_offset` dispatches on both coordinate types together and has **methods only for matching
pairs** — `(Radar, Radar)` and `(Projected, Projected)`. What matters is that the mismatch is reported
as a mismatch, in the style of `y_displacement_sign`'s named error (`src/coordinates.jl:176-177`), and
not as a `MethodError` naming internal types. The rule is worth stating as its own idea because it is
easy to read the abstraction above as "any two coordinates" when it is in fact "any two of the same,"
and the second is what makes a shared field, lattice and fit sound rather than merely convenient.

For projected pairs, matching types are not sufficient — the two must be on the same CRS and the same
spacing. `coregister` already requires the spacings to match and throws if they differ, and it already
*cannot* check the CRS: `ImageFootprint` carries none, deliberately, so the function stays pure
arithmetic and testable without GDAL. `REFERENCE.md` records that as a divergence from the reference,
which does compare EPSG codes, with the note that "a caller holding CRSs must compare them." On this path
the *whole* of what CHUNK-007 contributes is making that comparison rather than computing an offset — two
scenes differenced across mismatched frames give plausible numbers with rotated directions, which no test
of the arithmetic would catch, and a cross-path pair is the case most likely to straddle two UTM zones.

For radar pairs, matching types are likewise not sufficient — the two orbits must be **close enough that
a pixel shift describes the relationship at all.** Everything Path B rests on is that the offset field
is smooth and low-order over the scene, and that is a property of a small baseline rather than of the
geometry in general. As the two orbits separate, three things degrade in order, and only the first is
about the fit:

| separation grows | what breaks |
|---|---|
| moderate | the field stops being low-order; the fit's residual rises and CHUNK-005 already reports it |
| larger | the terrain term stops being negligible, so a DEM becomes required rather than optional |
| larger still | the two images see different geometry — layover and shadow differ — and no shift, of any order, relates them |

The third is the real limit and it is not a fit-quality question: past it the images do not contain the
same scene as seen from the same place, so correlation is measuring decorrelation. `dsamp`'s dependence on
height is the natural measure, since it is exactly the sensitivity that vanishes for a repeat pass — 0.18
px per 2000 m on the pair above, which is why terrain is nearly irrelevant there. A pair where that
sensitivity is a pixel per hundred meters is a pair where the DEM is doing the work and the offset is
mostly parallax.

So the check is on **the derived sensitivity, not on the orbits' separation in meters**. Baseline in
meters is the wrong instrument: what matters is how much of the range offset is height-dependent, and
that depends on the incidence angle and the wavelength as well as the separation. Computing `dsamp` at
two heights at scene center — two `pixel_offset` calls, microseconds — gives `∂dsamp/∂h` directly, and
that is the number to threshold. CHUNK-002 establishes it, alongside the iteration count, on the two real
pairs available.

A caller past the threshold is warned rather than refused, and the number is reported either way. The
bound is a property of what the caller will accept, not of the arithmetic — an interferometric use has a
far tighter tolerance than amplitude tracking, and this package cannot know which it is serving. What it
can do is refuse to be silent: the same stance `GEO2RDR_ITERATIONS` takes, where
`geo2rdr_iterations_needed` "returns a requirement, not a verdict."

*Fail before any work is done.* The offset field is lazy, so a bad pair could otherwise get as far as
the first indexed window — past lattice construction, past the fit, potentially into a threaded block
loop — before anything objects. Every check above is cheap and runs when the pair is constructed, not
when the field is first read. That follows the package's own stance (`fail fast`, and `coregister`'s four
up-front refusals) and it is the difference between a one-line error and a stack trace from inside a
task. Summarizing what is decided where:

| condition | when | outcome |
|---|---|---|
| coordinate types differ | construction | refuse, naming both |
| projected: spacings differ | construction | refuse, naming both |
| projected: CRSs differ | construction, where the CRS is known (CHUNK-007) | refuse, naming both |
| radar: height sensitivity above tolerance | construction | warn, reporting the value |
| no `secondary_coordinate` | first `pixel_offset` call | refuse, naming what to supply |
| fit residual above target | `fit` call (CHUNK-005) | refuse; the lattice is the fallback |

The two that are not refusals are the two that are judgement calls rather than incoherence: an orbit
separation the caller may find acceptable, and a fit order the caller can route around by using the
lattice. Everything above them describes a pair for which the operation has no meaning, and those refuse.

A third coordinate system added later — a geocoded radar product, which `src/coordinates.jl` already
notes belongs on the projected path — needs one method and inherits the field, lattice, fit and
correlator wiring unchanged. Pairing one *against* a slant-range product is refused by the same rule,
which is the right answer: geocoding is a resampling, so that comparison is asking to correlate an
image with a resampled version of a different geometry.

### The radar method

A composition of two operations already present:

```
(range₁, aztime₁) --rdr2geo(orbit₁)--> (lon, lat, h) --geo2rdr(orbit₂)--> (range₂, aztime₂)
```

isce3 calls this `rdr2rdr` and ships it (`isce3/geometry/rdr2rdr.py`), which makes it a directly
callable oracle. The misregistration in pixels is then

```
dsamp = (range₂ − r₀₂) / dr₂ − (range₁ − r₀₁) / dr₁
dline = (aztime₂ − t₀₂) · prf₂ − (aztime₁ − t₀₁) · prf₁
```

with each acquisition's own range origin, range spacing, sensing start and PRF. Both differences are
taken in pixels rather than in meters and seconds, because the two images need not share a range
spacing or a PRF, and for a burst they do not share a sensing start either.

Note that `dsamp`/`dline` are *not* `location_x`/`location_y` differences: those are `std::round`ed to
whole pixels (`src/radar/geo2rdr.jl:215-228`), and a resampler needs the fractional part. The
misregistration field is computed from the unrounded range and azimuth time.

## What the two paths add on top

**Path B** evaluates that field on a coarse lattice, hands it to the correlator as an a-priori shift
so the search only has to span actual ice motion, and subtracts it from the measured displacement
before the displacement-to-velocity operator runs. Nothing is interpolated except the field itself,
which is smooth. **Coordinate-agnostic**: everything above the field dispatch is shared, so the same
lattice, fit and correlator wiring serve a radar pair and an ascending/descending optical pair.

**Path A** evaluates the field at every output pixel and sinc-interpolates image 2's complex samples
there, removing and reapplying the Doppler phase across the interpolation chip. That is
`isce3::image::v2::resampleToCoords`, which is also directly callable and is therefore also an
oracle. **Radar only**, since only radar carries a phase to preserve — a projected pair needing
resampled *amplitudes* is asking for a reprojection, which GDAL does and this package should not.

## Laziness

Both paths are lazy in the same two senses, for the same reasons the package is already lazy
elsewhere.

*The field is not materialized.* A Sentinel-1 IW subswath mosaic is roughly 1500 × 22000 samples. At
about 5 µs per `rdr2rdr` — the existing radar point is 5.9 µs and this is the same two solves —
evaluating both components at every pixel is around 165 s and 528 MB. On a lattice at 100-pixel
spacing it is 3300 solves, about 16 ms, and 50 kB. The field is smooth over a hundred pixels for the
same reason a map projection is smooth over a few hundred meters, which is the argument
`src/interpolate.jl` already makes for `CoordLattice`. So the field is a **lattice plus an
interpolation**, and the exact per-pixel form exists to measure the lattice against.

*The resampled image is not materialized.* A resampled subswath is 33M complex samples. `ResampledSLC`
is an `AbstractMatrix{ComplexF32}` that reads the window it is indexed with, plus a sinc halo, from
image 2 — the same shape as `SLCDatasets.Amplitude`, which wraps a whole subswath for nothing and
costs only the windows taken from it. That composes: `amplitude(resample(...))` reads exactly the
chips a correlator asks for and no more, and `AutoRIFT`'s blocked entry point already reads its
inputs by window.

## TOPS deramping is out of scope, and blocked upstream

Sentinel-1 IW is TOPS: the antenna sweeps in azimuth within each burst, so the Doppler centroid
ramps steeply across the burst and the azimuth phase carries that ramp. Interpolating complex TOPS
samples without first removing the ramp aliases it, and the error grows toward the burst edges where
the ramp is steepest. The standard treatment removes the deramp and demodulation phase, resamples,
and reapplies it on the output grid.

This plan does not implement it, for two reasons.

It is not needed by anything here yet. Path B never touches the samples. Path A on a NISAR RSLC is
stripmap at zero Doppler, so the Doppler LUT is flat and the existing chip-phase handling is
sufficient — which is why Path A is worth building before deramping exists.

It is also blocked by its inputs. Deramping needs the azimuth FM rate polynomials, the Doppler
centroid estimates and the azimuth steering rate from the Sentinel-1 annotation. `SLCDatasets`
parses none of the three (`src/sentinel1.jl` reads the geometry scalars and the valid-region arrays
only), so there is nothing to deramp *with* until that reader is extended. That extension belongs in
`SLCDatasets`, next to the annotation parsing, not here.

**CHUNK-010 makes this a refusal rather than a silent wrong answer**: the complex resampling path
checks whether the acquisition is TOPS and throws, naming the missing metadata. Amplitude-only use
of the same path is permitted, since `abs` is insensitive to the ramp.

## Where the code lives

`src/` gains three files and no dependencies:

| File | Contents |
|---|---|
| `src/misregistration.jl` | the interface and both its methods, the lazy and lattice forms, the polynomial fit and its order selection |
| `src/radar/rdr2rdr.jl` | the radar solve pair, and the height source it solves against |
| `src/resample.jl` | the sinc kernel and `ResampledSLC` — radar only |

The projected method is a few lines and lives in `misregistration.jl` beside the interface; it needs no
file of its own now that it computes nothing. What the split still buys is that the field, lattice and
fit hold no radar concept and call only `pixel_offset` — so a method added later, for a geocoded product
or for a supplied measured shift, inherits all of it. That mirrors how `kernel/outputs.jl` already serves
both paths while `radar/geometry.jl` and `kernel/geometry.jl` supply the mapping each its own way.

Samples arrive as a plain `AbstractMatrix{<:Complex}`. Reading one from a product stays
`SLCDatasets`' job, as it is for the geometry, so `src/` acquires no IO stack and the
`juliac --trim` entry point in `app/` keeps working. The `SLCDatasets` extension gains nothing:
`pixels(slc)` is already an `AbstractMatrix`.

**Open scope question.** The offset field is unambiguously this package's — it is pair geometry, it
is what `CoregisteredPair` was missing, and nothing else can compute it. A complex-image resampler is
less obviously so: the package's own docstring says the pixel displacement "is estimated elsewhere,"
and `AutoRIFT.jl` already has a `src/resample.jl` for displacement fields. The recommendation is to
keep it here anyway — it is one 150-line kernel, its only consumer is the field above it, and a third
package for it would be worse than the mild scope stretch. Worth confirming before CHUNK-008, since
it is the one decision that moves files.

## Chunks

CHUNK-001 through 006 are Path B on the radar path and ship as a unit: after 006, amplitude tracking on
a real radar pair is geometrically correct, which it is not today. CHUNK-007 is Path B on the projected
path, which needs 001, 004 and 005 and nothing else. CHUNK-008 through 010 are Path A, radar only, and
CHUNK-011 is documentation.

007 is placed before Path A because it completes the coordinate-agnostic story while that abstraction is
still fresh, and because it is the smaller job. Its ordering against Path A is a priority call rather
than a dependency: nothing in 008–010 needs it, and nothing in it needs them.

Each chunk is verifiable before the next depends on it. **001, 004, 005 and 006 are the
coordinate-agnostic spine** and are written without a radar concept in them; 002 and 011 are the two
methods under `pixel_offset`.

### CHUNK-001: carry the secondary coordinate, and define the interface — done

Landed as described, with two departures worth recording.

*The field is named `secondary_offset_coordinate`, not `secondary_coordinate.* The shorter name reads as
a peer of `coordinate` — as though the pair had two coordinate systems of equal standing — when in fact
the geometry is the reference's outright and this exists only to feed `pixel_offset`. The longer name says
which.

*The same-kind checks live in `misregistration.jl`, not with the struct.* `pair.jl` is included before
`radar/coordinate.jl`, so `RadarCoordinate` does not exist where the struct is defined; `_check_secondary`
therefore has its mismatched-kind method there and its accepting methods beside the `pixel_offset` methods
they are preconditions for. That is the better placement independently of load order — a precondition
belongs with the operation it protects.

CHUNK-007's projected method landed here too, since it is one line once the interface exists: two
orthorectified images are on one grid, so the offset is `(0.0, 0.0)`. What remains of CHUNK-007 is the CRS
refusal.

Suite: 55666 → 55680, with 14 new assertions in `test/misregistration.jl`; the extension's pair testset
3 → 8. Every existing assertion unchanged, including the 47181 bitwise radar ones.


`CoregisteredPair` holds one coordinate. The offset needs both.

Add a `secondary_coordinate` field, holding any `AbstractImageCoordinate` or `nothing`. This is the
same widening the retired radar plan made to `CoregisteredPair` itself, and for the same stated reason:
the type's job is a pair of acquisitions plus their separation, and holding one coordinate overstates
what a pair is. `nothing` means "no secondary geometry supplied," which is what every pair built today
has, so every existing constructor keeps its behavior and `pairgeometry` continues to read
`pair.coordinate` alone.

**This package and `AutoRIFT.jl` have no users yet, so the field goes in the type rather than beside
it.** The alternative — a wrapper carrying a pair plus a secondary coordinate — exists only to avoid
breaking a struct that nothing constructs outside this repository and its two siblings, and it would
leave the package permanently describing a pair in two places. Take the breaking change now. The same
licence applies wherever else this plan touches an existing type: prefer the shape that will still be
right once these use cases exist, provided it does not slow the current path. It does not here — a
`nothing` field on an immutable struct is free, and the radar and projected kernels never read it.

What that licence does *not* extend to is introducing types on a guess about a consumer that does not
exist — see *Reading view geometry* above. Changing a type whose right shape is known is cheap now;
committing to a shape that is unknown is expensive whenever it happens.

Define `pixel_offset(pair, x, y, z)` here too, with the abstract signature and both methods as stubs
that throw naming what they need. Defining the interface in the chunk that adds the field — rather than
in whichever method lands first — is what keeps the projected case from becoming a widening of a radar
function later.

**Validate at construction, not at first use.** A `secondary_coordinate` is accepted only where the pair
is one `pixel_offset` can answer, and the check runs in the constructor:

- the two coordinates are the **same type**, so the offset is a pixel shift in a shared frame — the rule
  in *The abstraction* above;
- for two `ProjectedCoordinate`s, the **spacings match**, as `coregister` already requires of two
  footprints and for the same reason;
- the CRSs match where they are known, which is not here — see CHUNK-007.

Each is a comparison of a few numbers against a lazy field that would otherwise defer the failure to the
first indexed window, so there is no reason to defer it and one good reason not to. The message names
which condition failed and both offending values, per the package's existing practice of saying which
value offended rather than that something was wrong.

Note what this does *not* refuse: a pair with `secondary_coordinate = nothing`. That is every pair built
today, it is the honest state when no secondary geometry was supplied, and `pairgeometry` runs on it
unchanged. Only `pixel_offset` needs the field, and it throws naming what to supply. Making a missing
secondary an error at construction would break every existing caller to prevent a failure that is
already reported clearly at the point of use.

For a radar pair the `SLCDatasets` extension's `CoregisteredPair(reference, secondary)` fills it. This
is where the two-clock reduction has to be right, and it is the part most likely to produce a plausible
wrong
answer: the two products have **different epochs**, so `rdr2rdr` interpolating orbit 2 at a time
measured against product 1's epoch is off by the difference between them. That difference is not the
same as either product's `orbit_epoch_offset`, which relates a `sensing_start` clock to its own
orbit's clock. `SLCDatasets.epoch_offset` gives the seconds between a product's epoch and midnight,
so both reduce to a common clock through it.

The existing code base has already been bitten by exactly this class of error twice — see
`3d72505` ("fix the epoch conversion") and `e853066`, where a burst's epoch is neither the mosaic's
nor midnight. So the offset is computed and asserted, never assumed.

Verify: a pair built from two products whose epochs differ by a known constant reduces to that
constant; the same pair built from the *same* product twice gives an identically zero
misregistration field, which is the strongest single check available and costs no granule
(reintroducing an epoch error breaks it immediately); a pair built from two bursts of one
subswath, whose epochs differ from both the mosaic's and midnight; and a projected pair from
`coregister` carries `nothing`, with every existing test unchanged.

The refusals get their own tests, each asserting the *message* rather than the exception type, per the
package's `@test_throws "message"` practice: a radar-against-projected pair names the mismatch and does
not surface a `MethodError`; two projected coordinates with different spacings name both spacings; and a
pair with `nothing` reaches `pixel_offset` and is told what to supply, rather than being refused at
construction.

### CHUNK-002: the radar method

`src/radar/rdr2rdr.jl`. The `pixel_offset` method for two `RadarCoordinate`s: takes `(sample, line)` on
image 1, returns `(dsamp, dline)`, via `rdr2geo(orbit₁)` then `geo2rdr(orbit₂)`.

Reuses both existing solves. `geo2rdr` needs a starting guess and an iteration count: the natural
guess is image 1's own azimuth time, which for a repeat pass is within a few lines of the answer, so
`WARM_START_MIN_ITERATIONS`-scale counts suffice rather than `GEO2RDR_ITERATIONS`. That is a
measurement, not an assumption — establish the required count as `GEO2RDR_ITERATIONS` and
`RANGE_DOPPLER_ITERATIONS` were established, and record it with the same table.

Note which solve is which: `rdr2geo` here is the isce3 port in `src/radar/rdr2geo.jl`, not the
`_range_doppler` loop in `src/radar/geometry.jl`. The two are deliberately separate and the
distinction is documented at both sites; using the wrong one gives an answer that differs in the last
bits for no reason.

**The orbit-separation tolerance is established here**, since it is a derived quantity of this method
rather than a parameter of it. `∂dsamp/∂h` at scene center, from two `pixel_offset` calls at two heights,
is the number; it is what the paragraph on radar preconditions above thresholds, and it is reported by an
exported function that returns the value rather than a verdict, following
`geo2rdr_iterations_needed`'s precedent exactly. Measured on both real pairs available — the S1A/S1A
24-day pair at 0.18 px per 2000 m, and the S1A/S1B pair, which is the wide case — plus synthetic orbits
spanning separations between and beyond them. That measurement is what sets the default warning
threshold, and it goes in `REFERENCE.md` as a table in the shape the two iteration counts already have.

Two things the study has to state, because a threshold without them is a number with no standing: what
the sensitivity is at the point where the affine fit stops holding, and what it is where layover begins
to differ between the two geometries. Those are different limits, the second is the hard one, and the
default should sit below both with margin.

Verify: against `isce3.geometry.rdr2rdr` built from `isce3.geometry.rdr2geo` + `isce3.geometry.geo2rdr`
— the same non-bracketing solvers this package transcribes, so the agreement is the existing
1.9e-9 m of ground position rather than a solver tolerance. Also against the shipped
`isce3.geometry.rdr2rdr`, which uses the `_bracket` solvers and will agree only to their tolerances
(1e-5 m of height, 1e-7 s of azimuth time); reporting both separates a transcription error from a
solver difference. Round trip: from 1 to 2 then 2 to 1 returns the starting pixel. And a pair whose
sensitivity exceeds the threshold warns, with the value in the message, while still returning an
answer — a warning that suppressed the result would be a refusal wearing the wrong name.

### CHUNK-002: the radar method — done

Landed, and it reproduces the measurement this plan opens with from the orbits alone. At scene center of
the 24-day Sentinel-1 pair: **+17.78 samples, −2.76 lines**, against the +17.6 and −2.8 quoted above. A
product paired with itself gives exactly `(0.0, 0.0)`, which is the check that would fail on any clock
error.

Across the swath, sampled on a 5×5 grid of the reference's own range/azimuth extent at sea level:

| | min | max | spread |
|---|---|---|---|
| `dsamp` | 15.357 | 19.931 | 4.575 |
| `dline` | −2.8525 | −2.6715 | 0.181 |

**The range ramp is curved, and by enough to matter.** Along the first line the samples are 15.357,
16.492, 17.540, 18.512, 19.417 — first differences 1.135, 1.048, 0.972, 0.905, falling monotonically. A
straight line through the two ends misses the middle by **0.153 px**, which is above the 0.1 px target,
so first order does *not* suffice even on the smallest-baseline pair available. That settles the question
CHUNK-005 was left to answer and vindicates making the order a fit parameter rather than fixing it: had
this been built as an affine model it would have shipped with a systematic 0.15 px error at mid-swath.

Height sensitivity at scene center is `3.36e-5` samples/m and `−1.53e-7` lines/m — 0.067 samples across
2 km of relief, so terrain is nearly irrelevant on this pair, as expected for a near-repeat pass.
`height_sensitivity` is exported and returns the number rather than a verdict.

One departure: the method does **not** go through `rdr2geo`. The caller passes a grid point with its
elevation, exactly as `pairgeometry` does, so the ground point is already known and both acquisitions are
solved directly with `geo2rdr`. That is one fewer fixed-point iteration, no dependence on `rdr2geo`'s
convergence, and the same answer.

### CHUNK-003: a height source rdr2geo can sample — done

`src/radar/height.jl`: `AbstractHeightSource`, `height_at`, `reference_height`, `ConstantHeight`, plus
`RasterHeight` in the `Rasters` extension. `rdr2geo`'s `height` keyword now takes a number, a source, or
any callable of `(lon, lat)` in **radians**.

**The constant path is bitwise**, asserted directly rather than inferred: a scalar and a
`ConstantHeight` of the same value agree bit for bit across five heights and both look sides, and a flat
raster agrees with the equivalent constant bitwise too. All 47181 bitwise radar assertions and all 1070
isce3 numerics are unchanged.

Three findings, two of them real defects the work exposed.

*`Near` clamps rather than failing.* A query outside the raster returns its nearest corner, so a swath
reaching past a regional DEM would be solved against a height from hundreds of kilometres away — plausible
and wrong, which is the failure mode this package exists to refuse. `RasterHeight` now tests the extent
explicitly, with half a cell of tolerance, and returns `missing_height` (default 0.0, or `NaN` on request).

*A varying source can drive the solve into a geometry that has no solution*, and it threw a bare
`DomainError` from inside `sqrt`. Two roots can go negative: `cos_theta` outside `[-1, 1]` means the range
sphere and the height sphere do not intersect, and `(r·sin θ)² − α² < 0` means the zero-Doppler plane
misses the intersection circle. A `_reachable` predicate now tests both before `update_llh` and before the
final evaluation, leaving `converged` false and the last good estimate standing. Deliberately a separate
predicate rather than a guard inside `update_llh`: those three lines are asserted bitwise against isce3, so
they are duplicated rather than touched.

*A closure is not quite a `ConstantHeight`.* A closure declares no `reference_height`, so the iteration
starts from sea level rather than from the height itself and converges from a different direction — 1.4e-12
of a degree apart for an 8848 m constant, 1.5e-7 m of ground position. Immaterial, but it is why
`ConstantHeight` is a type rather than a spelling of `(lon, lat) -> h`, and the test records it rather than
asserting a bitwise equality that does not hold.

This was deferred from Path B on the grounds that `pixel_offset` takes a grid point and its elevation, so
no `rdr2geo` is involved. That still holds. It is needed now because Path A's field is indexed by *output
pixel* — confirmed against `resample_slc.py`, whose offsets are "the displacement from a pixel in the
output grid to the corresponding pixel in the input grid" — and inverting a reference pixel to a ground
point over terrain is exactly this.

Suite: 55888 → 55937.

### CHUNK-003 as originally scoped

Not needed by Path B, and the reason is the departure above. This chunk was premised on the field being
computed as `rdr2geo` against the reference followed by `geo2rdr` against the secondary, where the first
half must find a ground point from a radar coordinate and so needs the DEM sampled per iteration. It is
not: `pixel_offset` takes a grid point *and its elevation*, so the DEM enters as an argument the way it
already does everywhere else in this package, and no `rdr2geo` call is involved.

What still needs it is Path A. A resampler's field is indexed by *output pixel* — reference range and
azimuth — rather than by grid coordinate, and getting from a reference pixel to a ground point is exactly
`rdr2geo` over terrain. So this moves to sit before CHUNK-009 rather than being deleted, and Path B does
not wait on it.


`rdr2geo` snaps to a constant height each iteration (`src/radar/rdr2geo.jl:158-161`), which is all
the reference's callers need. A misregistration field over real terrain needs the DEM sampled at each
candidate location instead.

Introduce a height source callable as `h(lon, lat)`. A `ConstantHeight` method must reproduce today's
arithmetic **bitwise** — the existing footprint and incidence-angle tests assert it — so it is
dispatch, not a branch, and the constant case compiles to what it does now.

A raster-backed method belongs in the `Rasters` extension, since it is IO. The fixed-point structure
is unchanged by a varying height; what changes is that convergence is no longer guaranteed in a
handful of steps, so `rdr2geo_converged`'s flag becomes load-bearing rather than informational and
non-convergence has to surface.

Verify: `ConstantHeight` is bitwise against the current results across the existing fixtures; a
raster DEM that happens to be flat agrees with the constant to the interpolator's own tolerance; a
synthetic sloped DEM moves the answer in the direction and by the magnitude the local slope and
incidence angle predict.

### CHUNK-004: the lazy field

`src/misregistration.jl`. Two forms of one thing, and **no radar concept in the file** — it calls
`pixel_offset` and nothing below it. That is the constraint that keeps CHUNK-007 from being a rewrite,
and it is worth checking by grep rather than by intention.

`MisregistrationField <: AbstractMatrix{NTuple{2,Float64}}` — or a pair of matrices, whichever reads
better at the consumer — evaluating `pixel_offset` per element on indexing, exact and slow. Indexed by
the reference image's own axes, so windowed indexing reads only the window.

`LatticeMisregistration` tabulating the field on a coarse lattice and interpolating between nodes.
`CoordLattice` already does exactly this for a coordinate transform, including the two-elevation
level and the stencil halo, so extend it rather than write a second one. One divergence to state
plainly: for a datum shift, linear interpolation in elevation is *exact*, and `CoordLattice`'s
docstring says so; here it is an approximation, accurate because the height term is very nearly
linear over the tabulated range but not exact. That difference is worth a comment at the site and an
entry in `REFERENCE.md`, since the existing docstring's claim would otherwise read as covering both.

The node spacing is the accuracy knob and gets the same treatment `docs/interpolated-transform.md`
gives the transform lattice: a table of measured error against spacing, so a caller can pick against
a budget rather than a default.

Verify: the lattice against the exact field over a real acquisition, reporting the maximum error in
pixels at each spacing; the field is identically zero for a pair of one product with itself; windowed
reads agree elementwise with a whole-array read; `OffsetArray` and `view` inputs are handled, per
`rules/julia-generic-indexing.md`. And the lattice and fit are exercised against a **synthetic
`pixel_offset` method** — a closed-form field over a dummy coordinate type — so the spine is tested
without an orbit, a product or a solve, in the way `AffineTransform` lets the kernel's arithmetic be
tested without PROJ (`src/transforms.jl:52-59`). That test is what says the abstraction holds; if the
spine cannot be tested without radar, it is not actually coordinate-agnostic.

### CHUNK-004: the lazy field — done

`OffsetField` and `LatticeOffsetField`, both `AbstractMatrix{NTuple{2,Float64}}` over the window, in
`src/misregistration.jl`. The lattice form is built on `CoordLattice` rather than duplicating it, as
planned.

Two findings.

*The lattice needed a cell of slack, and not having it was a real bug.* `_grid_bounds` returns the bounds
of the queried grid-point centres exactly, and `build_lattice` extends by whole nodes from there, so a
corner query lands precisely on the last node the stencil reaches and `CoordLattice` refuses it — which
is the right behavior, and it fired on the first real window. `_inverse_bounds` already gives itself the
same slack for the same reason; `LatticeOffsetField` now does too.

*The dominant error at fine spacings is the elevation interpolation, not the node spacing.* Measured over
a 128×128 grid at 200 m on the 24-day pair, maximum absolute error in samples:

| `lattice` | default `zrange` (−200…4000 m) | `zrange` = 400…600 m |
|---|---|---|
| 4 | 6.4e-5 | 3.6e-6 |
| 8 | 6.4e-5 | 1.5e-5 |
| 16 | 6.4e-5 | 6.2e-5 |
| 32 | 1.9e-4 | 2.5e-4 |
| 64 | 9.3e-4 | 9.9e-4 |

Under the default range the error floors at 6.4e-5 samples for any spacing of 16 or finer. That floor is
the linear-in-elevation approximation across 4200 m — the caveat this plan predicted would need stating,
now measured. Narrow the range to the relief actually present and it disappears, with the spacing then
scaling as the square, exactly as bilinear interpolation should. Both figures are far below what a
correlator resolves, so `lattice = 32` is ample; the table is in the docstring because a caller who
tightens the lattice and sees no improvement needs to know why.

The consequence for the tests is that error is **not monotone in spacing** under the default range, so the
assertion is that the coarsest is worse than the finest rather than `issorted`.

The spine is asserted coordinate-agnostic by a `_ToyCoordinate` declared in the test file with a
closed-form affine offset — no orbit, no solve, no product. The lattice reproduces it to 1e-9 at any
spacing, since bilinear interpolation is exact for an affine field. If that test ever needs radar
machinery to run, the abstraction has moved to the wrong place.

Suite: 55693 → 55721.

### CHUNK-005: a polynomial fit whose order the residual chooses

Fit each component as a polynomial in sample, line and height:

```
dsamp = Σ a_ijk · sampⁱ · lineʲ · hᵏ,   i + j + k ≤ n
dline = Σ b_ijk · sampⁱ · lineʲ · hᵏ
```

by least squares over the lattice nodes, returning the coefficients **and the residual**.

The order `n` is a parameter, and the accuracy target is what fixes it. There is nothing special
about first order — it is what the pair this plan opens with needs, at a 4.17 px range ramp that is
monotonic and smooth. A wider baseline curves more, and if second order halves the residual there is
no reason not to use it: the fit is 10 coefficients instead of 4, evaluated once per point, against a
`rdr2rdr` pair costing microseconds. Cost is not the constraint here; over-fitting a smooth field on
a coarse lattice is the only real risk, and it is bounded by keeping the node count well above the
term count.

So the entry point takes a residual target and returns the **lowest order that meets it**, reporting
both the order chosen and the residual achieved. A caller that wants a specific order asks for it and
gets its residual back either way. This is deliberately the inverse of naming a model and hoping:
`n = 1` succeeding on a near-repeat pair and `n = 2` being needed on a wide-baseline one are then the
same call with the same guarantee, rather than two decisions a caller has to know to revisit.

Height enters as a term of the same polynomial rather than as a special case. It is very nearly
linear over any plausible relief — 0.18 px per 2000 m — so `k ≤ 1` will hold in practice, but there
is no reason to hard-code that when the order selection can establish it.

The residual is the load-bearing output, not a diagnostic. Above the target at the highest order
tried, the fit **fails** rather than returning coefficients that quietly describe the field badly,
and the lattice from CHUNK-004 is then the answer — it needs no fit and no order. The polynomial
exists because a handful of coefficients can be written into a product's metadata and re-applied
later, where a lattice is a table.

Verify: on the S1A/S1A 24-day pair, `n = 1` meets a 0.1 px target, reproducing the numbers this plan
opens with; on the S1A/S1B pair on disk
(`/private/tmp/s1ref/S1A_..._20151120...` and `S1B_..._20200926...`) the order selected and the
residual at each order from 1 to 3 are reported — that table is the deliverable, and it is what says
which order a wide-baseline pair needs; a synthetic field constructed to be exactly polynomial of
order `n` is recovered to machine precision at order `n` and no better at `n + 1`; and a fit is
refused where the node count does not exceed the term count by a stated margin.

### CHUNK-005: the polynomial fit — done

`src/offsetfit.jl`: `OffsetFit`, `fit_offset`, `offset_fit_terms`. **The open question is answered, and
first order is not enough.** On the 24-day pair over a realistic scene extent, maximum absolute residual
in samples:

| order | terms | residual, sample | residual, line |
|---|---|---|---|
| 1 | 4 | **0.0296** | 3.0e-4 |
| 2 | 9 | 1.42e-4 | 7.5e-5 |
| 3 | 19 | 1.26e-5 | 2.5e-5 |
| 4 | 31 | 8.6e-7 | 3.2e-5 |

Order 2 is 200× better than order 1 for six more coefficients. So the affine model this work started from
would have carried 0.03 samples of *structured* error — worst at mid-swath, where a straight line through
the swath edges deviates most — on the easiest pair there is. Selecting the order rather than naming it is
what caught that, and the plan's decision to make it a fit parameter is vindicated by measurement rather
than by argument.

Three findings.

*The elevation exponent has to be capped by the number of heights sampled.* With two heights, `z²` is a
linear combination of `1` and `z` over the sampled points, so its coefficient is not determined and the
normal equations are singular. My first version offered the term anyway and the pivoting guard caught it
immediately — which is the argument for having written the guard rather than assuming a well-conditioned
system. `offset_fit_terms` now takes `zorder`, and `fit_offset` sets it to `length(heights) - 1`.

*No `LinearAlgebra` dependency.* The systems are at most 35×35, so the normal equations plus Gaussian
elimination with partial pivoting is a page of code and microseconds. Adding the stdlib would pull BLAS
into a core that is deliberately dependency-light and `juliac --trim`-compilable, for a solve this size.

*The residual is a maximum, not an RMS.* A fit that is excellent on average and 0.5 px out at mid-swath is
precisely the failure the order selection exists to catch, and an RMS would average it away.

The design matrix is centred and scaled to roughly `[-1, 1]` before solving; raw projected coordinates
cubed are 1e18 and would spend most of a `Float64` on conditioning.

Suite: 55721 → 55762.

### CHUNK-006: wire Path B through the correlator — done

In `AutoRIFT.jl`, additively: `pointset(g::PairGeometry; offset = nothing)` adds the misregistration to the
existing velocity-derived prior, and `remove_misregistration` subtracts it from a measured displacement.
`remove_misregistration` is declared in `src/points.jl` without a method and defined in the extension,
since the offset field is this package's to compute.

**The sign convention was the trap, and measurement was the only way to settle it.** AutoRIFT returns the
offset from secondary back to reference — the *negative* of the feature displacement — while
`pixel_offset` returns where a point sits in the secondary relative to the reference. Those are opposite
senses, so a field taken straight from `pixel_offset` must be negated before use. Established on a
synthetic pair built with a known shift: a total shift of +9 samples is reported as −9, so a
misregistration of +7 contributes −7 to the output and removing it means subtracting −7. Documented as a
warning admonition on `remove_misregistration` rather than left for a caller to discover, because the
failure is silent and 18 pixels wide.

The closed-loop test is what the plan asked for and it earns its place: a secondary built from a known
misregistration *plus* a known motion, correlated, corrected, and asserted to return the motion alone.
Six assertions, one of them stated the way a user reads it. It passes exactly.

AutoRIFT's extension testset goes from 3 testsets to 5 over this handoff, 16 new assertions, Aqua clean.
ImagePairGeometry's own suite is unchanged at 55769 — nothing in `src/` moved for this chunk.

### CHUNK-006 as originally scoped

Two edits, and the sign conventions are where this goes wrong.

*Before correlating.* The misregistration is added to the a-priori shift the correlator already
takes. It does not replace it: `PointSet`'s `dx_prior`/`dy_prior` already carry geogrid's
`offset_x`/`offset_y`, which is the displacement implied by the reference velocity field
(`ext/AutoRIFTImagePairGeometryExt.jl` in `AutoRIFT.jl`). The two are different quantities that
compose — one is where the ice is expected to have moved, the other is where the grid itself is
offset — so the prior becomes their sum, and `y_displacement_sign` applies to the velocity-derived
term only. The misregistration is already in image axes.

*After correlating.* `AutoRIFT` adds the prior back into its output (`src/track.jl:426-427`), so the
returned `dx`/`dy` include it. The misregistration is therefore subtracted from the returned
displacement before it reaches `off2vel`. Doing this in the other order, or forgetting that the prior
is already added back, produces a velocity field that is wrong by the misregistration — 17.6 px on
this pair, which is comparable to the signal.

The composition is small enough to write out once and test as a closed loop rather than reason about
per site. A synthetic pair with a known imposed misregistration and a known imposed motion must
recover the motion; that single test pins every sign.

The natural home is the existing `AutoRIFT`/`ImagePairGeometry` extension, alongside `pointset`,
which already negotiates index base, the half pixel, missing values and the y sign for the same
reason. It reads the field through the `AbstractMatrix` interface only, so it is coordinate-agnostic
like the rest of the spine and CHUNK-007 needs no edit here.

Verify: the closed loop above; on the real pair, the recovered velocity over stable ground is zero to
within the correlator's precision, where without the correction it is offset by the misregistration
— that is the end-to-end statement of what this chunk fixes; and the search extent needed falls to
what actual ice motion requires, which is the secondary benefit the a-priori shift buys.

### CHUNK-007: the projected method and its preconditions — done

The `pixel_offset` method landed with CHUNK-001. What this chunk added is the CRS refusal, and it went
somewhere the plan did not anticipate: **`ImageFootprint`, not a `ViewGeometry`.**

The plan assumed the CRS would arrive with view geometry, since an azimuth-of-view direction needs a frame
to be expressed in. With the parallax correction dropped there is no `ViewGeometry` — but the check is
still available, because `coregister` is where the reference makes it and `ImageFootprint` is what
`coregister` takes. So `ImageFootprint` gains an optional `crs`, `coregister` refuses a mismatch naming
both codes, and the `Rasters` extension supplies it from the raster so a caller reading scenes from disk
gets the check without asking.

Optional rather than required, because the intersection arithmetic needs no CRS and requiring one would
make `coregister` untestable without GDAL — the fixtures build footprints from numbers. Absent means *not
checked here*, not *assumed to agree*.

One bug found by its own test: the default inner constructor accepted a raw `Integer` and stored it
unnormalized, so `32607` and `GFT.EPSG(32607)` — the same CRS — compared unequal and would have been
refused. `MapGrid` guards this with an inner constructor; `ImageFootprint` now does too.

`REFERENCE.md`'s divergence entry is rewritten: the check is no longer unavailable, it is conditional on
being given the codes.

Suite: 55762 → 55769.

### CHUNK-007 as originally scoped

A few lines in `src/misregistration.jl`, and the reason it is a chunk at all is the preconditions rather
than the arithmetic.

The `pixel_offset` method for two `ProjectedCoordinate`s returns **zero** for a pair that passes its
checks. That is the correct answer, not a stub: both images are terrain-corrected onto the same map grid,
so the pixel index the kernel computes is right for both, and any remaining disagreement is the
producer's geolocation residual — not a function of anything this package holds. See *The projected path
needs the interface, not a parallax correction* above for why the earlier plan to derive a parallax
correction here was dropped.

An optional supplied offset is accepted, as a constant or as an array over the window. A caller that has
measured a residual shift over stable ground — the only sound way to obtain one — gets the lattice, the
fit and the correlator wiring for free. Supplied, never derived, and absent by default so that "no shift
applied" is distinguishable from "a shift was applied."

**The CRS refusal is the substance of this chunk.** `coregister` already refuses a pair whose spacings
differ and already *cannot* compare CRSs, since `ImageFootprint` carries none by design —
`REFERENCE.md` records that as a divergence from a reference that does compare EPSG codes, with the note
that "a caller holding CRSs must compare them." Nothing has ever made that comparison available.

Here it can be, because `MapGrid` already carries a CRS and `PairGeometry` already carries the grid's.
So the check is: where both images' CRSs are known to the caller, `pixel_offset` refuses a pair whose
CRSs differ, naming both. Where they are not known it says that, rather than assuming they match — an
unchecked assumption reported as a check is worse than no check. `REFERENCE.md`'s entry is updated to say
where the comparison is now made and where it remains the caller's.

This matters most for exactly the pairs that motivated the projected path here: a cross-path optical pair
is the case most likely to straddle two UTM zones, and two scenes differenced across mismatched frames
give plausible numbers with rotated directions, which no test of the arithmetic would catch.

Verify: a matching pair returns identically zero, so every existing projected test stands unchanged — the
whole suite is the test here, and any movement in it means this chunk did something it should not; a pair
whose spacings differ is refused, naming both; a pair whose CRSs are known and differ is refused, naming
both; a pair whose CRSs are unknown is refused with a message saying so rather than passing silently; a
supplied constant offset flows through the field, lattice and fit unmodified, checked by this chunk adding
no lines to those; and the two cross-path Landsat pairs in the golden set
(`LE07_061018_20120428 × LE07_060018_20120726`, and `LE07_061018_20130314 × LC08_060018_20130330`) are run
end to end to confirm they are accepted or refused for the right reason and that their delivered products
still reproduce.

### CHUNK-008: the sinc kernel — done

`src/resample.jl`: `SincKernel`, `sinc_interpolate`, and the four `SINC_*` constants. **Bitwise against
isce3** on all 73 probed kernel coefficients and all 7 interpolated positions.

The finding worth recording is what bitwise depended on. isce3 accumulates its 64-tap sum in
`complex<float>`, casting each weight to it — `ret += arrin(...) * static_cast<U>(wy) *
static_cast<U>(wx)`. My first version accumulated in `Float64`, which is *more accurate* and differs by
about one `Float32` ULP, so 3 of 5 positions disagreed in the last bit. Matching the reference's
accumulation type made all of them exact.

That is offered as a keyword rather than hidden: `accumulate = ComplexF64` is the better answer
numerically and is not bitwise, and a test pins the direction so a future widening of the accumulator
cannot silently break the agreement. The association matters too — `a * wy * wx` and `a * (wy * wx)` round
differently at `Float32`, so the reference's grouping is transcribed.

Both transcribed quirks have tests aimed at them: the table index truncates (two positions inside one
1/8192 step return the *identical* value, which a rounding index would not), and the tap sum runs
downward (a ramp is recovered along each axis, which a reflected stencil would not).

One correction to my own reasoning, made mid-chunk: I first attributed a 1e-6 ramp-recovery error to the
truncating index, and it is not — at a position the table holds exactly, the error is still 1.1e-6. It is
the finite passband of an 8-tap windowed sinc. The test now separates the two effects rather than lumping
them into one tolerance.

Suite: 55769 → 55888.

### CHUNK-008 as originally scoped

`src/resample.jl`. Port `isce3::core::Sinc2dInterpolator` — an 8-tap kernel tabulated at 8192
sub-pixel positions, Hamming-weighted (`pedestal = 0.0`, `beta = 1.0`), normalized per sub-pixel row
(`cxx/isce3/core/Sinc2dInterpolator.cpp`). The constants are `SINC_LEN = 8`, `SINC_HALF = 4`,
`SINC_ONE = 9`, `SINC_SUB = 8192` (`isce3/core/Constants.h:32-35`).

Two details a from-scratch implementation would not have, both of which change the answer and both of
which are transcribed:

The kernel table is indexed by `min(max(0, Int(frac * 8192)), 8191)` — a truncation, not a rounding,
so the interpolant is piecewise constant in the sub-pixel coordinate at a 1/8192-pixel step rather
than continuous. And `_sinc_eval_2d` sums with a **descending** index (`arrin(intpy-i, intpx-j)`),
so the kernel is applied reversed relative to how it reads; combined with `interp_impl`'s
`xx = ix + halfKernelLength` this lands on the right samples, but either half alone does not.

Verify bitwise against `isce3.core` through a generator in `geogrid-ref`, on the kernel table itself
and on evaluations at sub-pixel positions including the truncation boundaries. Bitwise is achievable
here — the kernel is `cos` and `sin` of exactly representable arguments and a normalization — and
should be asserted rather than bounded, per the exactness stance in `REFERENCE.md`.

### CHUNK-009: the lazy resampled SLC — done

`ResampledSLC` in `src/resample.jl`: an `AbstractMatrix{ComplexF32}` over the reference's grid wrapping the
secondary's samples and an offset field. **144 of 144 samples bitwise against
`isce3.image.v2.resample_slc.resample_to_coords`**, worst difference exactly 0.0, over a 12×12 window whose
offsets mix an integer shift, an exact half, a ramp, and a planted `NaN`.

Targets `resampleToCoords` rather than the older `ResampSlc`, as planned — the v2 form is driven by index
grids and carries no carrier polynomials or flattening. The index relation is isce3's: absolute input index
is output index plus offset, which carries over unchanged because both sides here are one-based.

Two departures worth recording.

*The bounds test is here, not left to `sinc_interpolate`.* The kernel returns *zero* when its stencil does
not fit, following isce3 — but zero is a sample value, indistinguishable from a real one. So the fit is
tested before the call and the fill returned instead, which is what `resampleToCoords` does at the block
level.

*One assertion of mine was wrong about the physics, not the code.* I asserted that a nonzero Doppler
changes the phase and leaves the magnitude alone. It does not: derotating the chip before interpolating
makes the signal smoother in azimuth, so the interpolant is genuinely a different number — which is the
entire reason the derotation exists. Measured against per-sample phase:

| Doppler | rad/sample | magnitude moves |
|---|---|---|
| 0.01 Hz | 1.3e-4 | 6e-5 |
| 1 Hz | 1.3e-2 | 5e-3 |
| 40 Hz | 5.2e-1 | 0.25 |

Smooth in between, and zero at zero. The test now asserts that scaling rather than an invariance that
holds only at zero. A Doppler model requires a `coordinate`, since only that turns an output pixel into an
azimuth time and a slant range, and it is refused at construction.

Suite: 55937 → 56111.

### CHUNK-009 as originally scoped

`ResampledSLC <: AbstractMatrix{ComplexF32}` over image 1's grid, wrapping image 2's samples and a
misregistration field. Indexing a window reads that window from image 2, offset by the field and
grown by `SINC_HALF`, and returns the interpolated samples.

Follows `resampleToCoords` (`cxx/isce3/image/v2/Resample.cpp`), which is the current isce3 entry
point and the one the NISAR workflow calls, rather than the older `ResampSlc::_transformTile`. The
difference matters: `resampleToCoords` is driven by index grids and has no carrier polynomials or
flattening, where `ResampSlc` is driven by offset rasters and carries both. The v2 form is the right
target — flattening is an interferometric step, and the carrier polynomials are a legacy of ISCE2
products.

The Doppler handling transcribes as written: the chip is read with the azimuth Doppler phase
*removed* per chip row (`exp(-i·f_dop·(i_chip − chip_half))`), interpolated, and the phase
corresponding to the fractional azimuth position reapplied (`exp(+i·f_dop·frac_az)`). For a
zero-Doppler grid — NISAR, and every acquisition this package currently handles — `f_dop` is zero and
both phasors are unity, which is why Path A is usable before any Doppler LUT exists. Carrying the
term anyway keeps the arithmetic the same shape as the reference's, and it is what a nonzero Doppler
would need.

Out-of-bounds and NaN handling are the reference's: a `NaN` index, or a chip reaching outside the
input, yields the fill value rather than a clamped read. The default fill is `NaN + NaN·im`, matching
`resample_slc_blocks`.

Verify against `isce3.image.v2.resample_slc.resample_to_coords`, which is directly callable in
`geogrid-ref` and takes exactly this package's inputs — index grids and a complex block — so the
comparison is bitwise on `ComplexF32` and needs no product. Also: a synthetic complex image shifted
by a known sub-pixel offset resamples back to itself within the kernel's own accuracy, which is the
check that catches a transposed axis or a sign-flipped offset, neither of which the isce3 comparison
would catch if the index grids were built the same wrong way on both sides. And `amplitude` of the
result agrees with resampling the amplitudes only, which is what says the amplitude path is unharmed.

### CHUNK-010: refuse TOPS on the complex path — done

Split across the two packages, and the split is the finding. The plan put the whole check in this
package's `SLCDatasets` extension. **Half of it belongs in `SLCDatasets`**: whether an acquisition is
TOPS is a fact about how the product was collected, which that package reads and this one cannot infer.

`SLCDatasets` gains `is_tops` and `deramp_parameters`, following the `AbstractBurstBackend` pattern its
own previous commit established — a backend answers a question about itself rather than callers naming
formats. Every burst backend is TOPS by construction, since that abstract type already means "one burst
of a TOPS acquisition", so both Sentinel-1 forms inherit it. `MergedBurstBackend` is the exception: it
subtypes `AbstractSLCBackend` directly and would default to `false`, which is exactly the case that
would reach a resampler unnoticed, so it has its own method and its own test.

`deramp_parameters` throws naming the three annotation fields — `azimuthFmRateList`, `dcEstimateList`,
`azimuthSteeringRate` — and noting they sit in the annotation already parsed for the geometry. So the
gap is documented at the place a consumer will hit it rather than only in this plan.

This package's extension gains `ResampledSLC(::SLC, offset; amplitude_only = false)`, which refuses a
TOPS acquisition unless the keyword says the phase will not be read. A keyword rather than a default so
the choice is written at the call site.

One thing the committed Sentinel-1 fixture cannot exercise: it carries annotation and no `measurement`
directory, so the amplitude path fails there when it reaches the samples. That failure is turned into
the assertion — it comes from the sample reader rather than from the TOPS check, which is what says the
keyword let it through.

Suites: `SLCDatasets` 2866 → 2896, `ImagePairGeometry` unchanged at 56111 with 11 new extension
assertions.

### CHUNK-010 as originally scoped

A TOPS acquisition reaching the complex resampler without deramping produces a phase-corrupted
result that looks like a valid image. Throw instead, naming the three annotation fields that are
missing and where they would come from.

The check is a property of the acquisition, so it belongs where the acquisition is known. That is the
`SLCDatasets` extension for a product-built pair, since `src/` has no notion of a product; a
hand-assembled `RadarCoordinate` carries no way to tell, so the core cannot answer it and should not
pretend to.

Amplitude-only use is explicitly permitted and takes a keyword to say so, since `abs` is insensitive
to the ramp — a caller doing amplitude tracking on Sentinel-1 has a legitimate use for the resampler
today.

Verify: a Sentinel-1 IW pair throws on the complex path with a message naming the missing fields;
the same pair succeeds with the amplitude-only keyword; a NISAR pair succeeds without it.

### CHUNK-011: documentation, benchmarks, REFERENCE.md

`docs/src/coregistration.md` — named for the operation rather than for `resampling`, since resampling is
one of two things the page covers and the smaller one. In the shape of `docs/src/radar.md`: what
misregistration is on each coordinate system, the two paths, the measured cost and accuracy of the
lattice against its spacing, and the fit residual by polynomial order on both real radar pairs. On the
projected side it states what the package does *not* do and why — an orthorectified pair is already
coregistered, the residual is the producer's and not computable here, and the remedy is a measured shift
— since a reader who has just seen a radar offset field will otherwise assume an optical one exists.

`REFERENCE.md` gains a section: this is the first thing in the package that is **not** a
reimplementation of geogrid — geogrid assumes the coregistration has already happened, on both paths.
So the reference is isce3 for the radar side, and the
exactness standard is stated against each with the same two tiers.

`benchmark/`: the per-pixel field cost on both methods, the lattice build cost against spacing, and the
resampler's throughput against a window size, so the laziness claims above are measured rather than
asserted.

Three places currently overstate what is done, and each gets a correction rather than a caveat buried in
a docstring. The README's radar paragraph and `docs/src/index.md`'s status section both call the radar
path complete — true against geogrid, narrower than a reader will take it. And `src/pair.jl`'s header
comment argues at length that the secondary image matters, using `coregister` as the example; that
argument is right and is now incomplete, since it covers the secondary's *footprint* and not its view
geometry. All three get a sentence pointing here.

## What this plan does not settle

Four things, each of which needs a decision or a measurement rather than more design.

**Whether the sinc resampler belongs in this package.** See *Where the code lives*. The
recommendation is yes; it is worth confirming before CHUNK-008, since it is the only decision here
that moves files between repositories.

**Whether a measured optical shift is worth plumbing through.** CHUNK-007 accepts a supplied offset but
nothing here produces one, so that path is untested by any real case. Whether it earns its place depends
on whether cross-path optical pairs in practice show a residual shift large enough to matter — measurable
from the golden set by comparing recovered velocity over stable ground against zero, on the two cross-path
pairs versus the same-path controls. If the cross-path residual is no worse, the supplied-offset path is
speculative generality and should come out.

**What polynomial order a wide-baseline pair needs.** CHUNK-005 measures the residual at orders 1
through 3 on the S1A/S1B pair on disk, which is the stress case. First order is what the near-repeat
pair needs; if second or third is needed there, that is a fine outcome and only changes the default
order. If no order within that range meets the target, the fit is the wrong instrument for a wide
baseline and the lattice is the general answer — which changes which of the two is presented as the
default.

**Whether Path B is sufficient for the production pipeline.** It corrects the geometric
misregistration but leaves up to half a pixel of sub-pixel residual inside every chip, since the chip
is still cut at integer samples of image 2. For amplitude tracking that residual is a translation the
correlator measures and is harmless. Whether it is harmless for the multi-chip-size pyramid, where
the same residual appears at every level and is decimated differently at each, is not established
here and is a measurement rather than an argument.
