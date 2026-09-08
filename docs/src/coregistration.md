# Coregistration

The kernel computes one pixel index per grid point and hands it to a correlator for **both** images of a
pair. That is right only where the secondary image sits on the reference's grid — and for radar it does
not.

Two acquisitions have different orbits, so the same ground point falls at a different range sample and
azimuth line in each. On a 24-day Sentinel-1 pair that is about 18 samples of range, comparable to the ice
motion being measured. geogrid does not compute this because it does not have to: the reference pipeline
resamples the secondary SLC onto the reference grid *before* geogrid runs, so by the time geogrid sees the
pair the assumption holds. Nothing in this package did that resampling, so the gap was inherited rather
than chosen.

This page covers what closes it. [`pixel_offset`](@ref) measures the disagreement from the two orbits, with
no reliance on scene contrast; [`OffsetField`](@ref) and [`LatticeOffsetField`](@ref) evaluate it over a
grid; [`fit_offset`](@ref) reduces it to a few coefficients; and [`ResampledSLC`](@ref) resamples the
secondary's complex samples onto the reference's grid.

## Two ways to use it

They are independent, and which one you need depends on whether you read the phase.

| | correct after correlation | resample |
|---|---|---|
| what you get | the offset as a prior and a correction | the secondary's samples on the reference's grid |
| sub-pixel phase | not addressed | correct |
| cost | a lattice of solves per pair | 190 ns per output sample |
| for | amplitude feature tracking | complex correlation, interferometry |

**Correcting after correlation** is the whole requirement for amplitude tracking. `abs` discards the
phase, so a misregistration is a translation the correlator can measure — provided its search window
reaches, which is the second reason to supply the offset as an a-priori shift rather than searching for
it. The residual left inside each chip is sub-pixel and harmless to a magnitude.

**Resampling** is required for anything that reads the phase, because there the misregistration is not a
translation the correlator can absorb: the phase inside each chip is wrong and no search extent recovers
it.

Both rest on the same measurement, so neither is a detour on the way to the other.

## Measuring the offset

```julia
using ImagePairGeometry, SLCDatasets

pair = CoregisteredPair(open_slc(url1), open_slc(url2))
pixel_offset(pair, x, y, z)      # (dsample, dline) at one grid point
```

The pair must carry the secondary's coordinate system, which `CoregisteredPair(reference, secondary)`
supplies. A pair built from a single acquisition does not have one and says so.

The result is `(dsample, dline)`: where the point sits in the secondary *minus* where it sits in the
reference, in the reference's own pixel axes, and **fractional** — the rounded indices
[`pairgeometry`](@ref) reports have already discarded the part a resampler needs.

### It is a pixel shift only where that describes the pair

[`height_sensitivity`](@ref) is how you check. It reports how much of the offset moves with elevation,
which is the measure of whether a shift describes the two images at all:

```julia
height_sensitivity(pair, x, y)   # (dsample/dm, dline/dm) at one point
```

For a near-repeat pass the two orbits are close, their look directions at a given ground point nearly
agree, and the offset barely depends on elevation — measured at `1.2e-4` samples per metre on the
synthetic repeat pair, so a quarter of a sample across 2 km of relief. As the orbits separate three things
degrade in order, and only the first is a fit-quality question:

| separation grows | what breaks |
|---|---|
| moderate | the field stops being low-order, and [`fit_offset`](@ref)'s residual rises |
| larger | the terrain term stops being negligible, so a DEM becomes required rather than optional |
| larger still | the two images see different geometry — layover and shadow differ — and no shift of any order relates them |

The third is the real limit and is not about fitting: past it the two images do not contain the same
scene as seen from the same place, so a correlator is measuring decorrelation. `height_sensitivity`
returns the number rather than a verdict, because what threshold matters depends on what you are doing —
an interferometric use tolerates far less than amplitude tracking.

## Over a grid, lazily

An offset field is an `AbstractMatrix{NTuple{2,Float64}}` shaped like the window, so it lines up with a
[`PairGeometry`](@ref) over the same window and can be read a block at a time.

```julia
field = LatticeOffsetField(pair, grid, window; dem, lattice = 32)
field[i, j]        # (dsample, dline) at that grid point
```

[`OffsetField`](@ref) evaluates [`pixel_offset`](@ref) per element — exact, and too slow to use densely.
[`LatticeOffsetField`](@ref) tabulates it on a coarse lattice and interpolates, which is what makes it
affordable: the offset is set by two orbits' geometry and varies over kilometres, not pixels.

The scale of the difference, measured:

| | cost |
|---|---|
| one `pixel_offset` | 2.0 µs |
| a whole S1 IW subswath, exact | 66 s, 528 MB if materialized |
| a whole NISAR swath, exact | 613 s, 4.9 GB |
| a 64×64 window, exact | 10.2 ms |
| the same window, `lattice = 16` | 0.33 ms to build, 1.3 ms to read |

So a lattice is not an optimization here, it is what makes a dense field possible at all. And the accuracy
it gives up is far below what a correlator resolves — maximum error against the exact field over a
128×128 grid at 200 m spacing:

| `lattice` | default `zrange` | `zrange` narrowed to the relief |
|---|---|---|
| 4 | 6.4e-5 sample | 3.6e-6 |
| 16 | 6.4e-5 | 6.2e-5 |
| 32 | 1.9e-4 | 2.5e-4 |
| 64 | 9.3e-4 | 9.9e-4 |

`lattice = 32` is ample. Note what the left column does at fine spacings: the error **floors** at 6.4e-5
and tightening the lattice further buys nothing. That floor is the linear-in-elevation approximation across
the default 4200 m `zrange`, not the node spacing — narrow `zrange` to the relief actually present and it
disappears. Worth knowing because a caller who tightens the lattice and sees no improvement is looking at
the wrong knob.

## As a polynomial

A lattice is a table. A polynomial is a handful of coefficients, which can be written into a product's
metadata and re-applied later by something that has neither this package nor the orbits.

```julia
fit = fit_offset(field, grid, window; heights = (0.0, 1000.0, 2000.0), target = 0.01)
fit.order        # the lowest order meeting the target
fit.residual     # what it achieved, in pixels
fit(x, y, z)     # evaluate anywhere
```

**The order is chosen by the residual, not named in advance**, and that matters more than it sounds.
Azimuth is flat across a scene and range drifts monotonically, which reads as a ramp — so an affine model
is the obvious guess. It is wrong. On the 24-day Sentinel-1 pair the range offsets along one line are
15.357, 16.492, 17.540, 18.512, 19.417: first differences of 1.135, 1.048, 0.972, 0.905, falling steadily.
A straight line through the swath edges misses mid-swath by **0.153 samples**.

Residual by order on that pair:

| order | terms | residual | evaluation |
|---|---|---|---|
| 1 | 4 | 0.0296 sample | 45 ns |
| 2 | 9 | 1.4e-4 | 107 ns |
| 3 | 19 | 1.3e-5 | 165 ns |
| 4 | 31 | 8.6e-7 | — |

Second order is 200 times better than first for six more coefficients. So an affine model would have
shipped 0.03 samples of *structured* error — worst at mid-swath, where a line through the edges deviates
most — on the smallest-baseline pair a repeat pass offers.

Where no order up to `max_order` meets the target, `fit_offset` refuses rather than returning coefficients
that describe the field badly, and a [`LatticeOffsetField`](@ref) is the answer: it needs no fit and no
order. Selecting an order over a 64×64 window at three heights costs 28 ms, once per pair.

## Feeding a correlator

With `AutoRIFT` loaded, the offset joins the a-priori shift its search is centred on, and comes back out of
the measured displacement:

```julia
using AutoRIFT
pts = AutoRIFT.pointset(geometry; chip_size = 32, offset = field)
out = autorift(reference, secondary, pts)
corrected = AutoRIFT.remove_misregistration(out, field)
```

It *adds* to the existing prior rather than replacing it: `offset_x`/`offset_y` are where the ice is
expected to have moved, the misregistration is where the secondary's grid sits, and a chip has to be cut at
the sum.

!!! warning "The offset must be negated first"
    `AutoRIFT` returns the offset from secondary back to reference, which is the **negative** of the
    feature displacement. [`pixel_offset`](@ref) returns the opposite sense. So a field taken straight from
    here must be negated before it is passed to either `pointset` or `remove_misregistration`.

    Both take it in the same convention, so negating once and passing the same array to both is consistent.
    Passing it to one and not the other is silent and 18 pixels wide.

## Resampling the samples

```julia
using SLCDatasets
resampled = ResampledSLC(secondary, field)      # complex, on the reference's grid
amplitude(resampled)                            # magnitudes, for a feature tracker
```

An `AbstractMatrix{ComplexF32}` over the reference's grid. Nothing is resampled until it is indexed:
reading a window reads that window from the secondary, grown by the sinc halo, and interpolates. A
resampled subswath is 33 million complex samples and a correlator reads a few hundred thousand, so laziness
is what makes this usable rather than a convenience.

Measured: 190 ns per output sample, about 5 Msamples/s, so a whole S1 IW subswath is 6.4 s
single-threaded and a NISAR swath 59 s.

Out-of-bounds reads and unplaceable points (a `NaN` offset) give the fill value, `NaN + NaN·im` by default
— not zero, which is a sample value and would be indistinguishable from real data.

## TOPS data carries an azimuth ramp

Sentinel-1 IW steers the antenna in azimuth across each burst, putting a steep phase ramp on the samples —
some thousands of radians between a burst's centre and its edge. Interpolating without removing it first
aliases it, worst where it is steepest, and the result is a phase-corrupted image that looks entirely valid.

So the phase is removed before the sinc convolution and reapplied for the interpolated position.
[`TOPSCarrier`](@ref) evaluates it, and with `SLCDatasets` loaded it is built from the product's own
annotation:

```julia
using ImagePairGeometry, SLCDatasets

b = bursts("S1A_....SAFE"; orbit = "S1A_....EOF", swath = 2)
# The carrier is built from the annotation; nothing extra to pass.
r = ResampledSLC(b[1], offset)
```

Measured on a synthetic ramp: interpolating with the carrier removed holds a unit-magnitude signal to
6.0e-7, and leaving it in loses 1.1e-2 — the aliasing, four orders of magnitude larger.

**The deramp costs 12.5×**: 2.0 µs per output sample against 160 ns without, so a whole S1 IW subswath is 66 s
single-threaded rather than 5.3 s. The carrier is evaluated at every chip tap — 81 of them, plus two — because
a TOPS ramp varies with slant range as well as azimuth, so nothing can be hoisted out of the inner loop the
way the Doppler's per-row phasor is. Hoisting it would be an approximation: the range variation reaches
0.07 rad across one chip at a burst edge, about 4° of phase. The path *without* a carrier is unchanged.

`amplitude_only = true` skips the deramp, since `abs` discards the phase. That is what most of this pipeline
does with Sentinel-1, and it is a keyword rather than a default so the choice is visible at the call site —
and on this cost, it is also the fast path by an order of magnitude.

!!! note "A merged subswath is refused"
    The ramp is referenced to each burst's own centre, so a merge carries one per burst rather than one for
    the image, and a chip spanning a seam has no single answer. Resample the bursts individually, or pass an
    explicit `carrier` covering the merged grid. `SLCDatasets.burst_at` says which burst a merged line
    belongs to.

The verification standard here is weaker than elsewhere on this path, and deliberately stated: the deramp
lives in the Python `s1reader` package rather than in isce3's C++, so `test/reference/topsramp.json` is
generated from that arithmetic and agreement is to 2.9e-16 relative rather than to the bit. See
`REFERENCE.md`.

## What this does *not* do on the projected path

[`pixel_offset`](@ref) is defined for two `ProjectedCoordinate`s and returns **zero**. That is the answer,
not a placeholder.

An orthorectified product — a Landsat L1TP scene, say — is already geolocated and terrain-corrected. The
displacement an orthorectification introduces has been removed by the producer, and what remains is the
error in the DEM it used and in its own view model: neither reported by the product, neither reconstructible
from an origin, a spacing and a size. A correction proportional to an unknowable elevation error would be a
free parameter dressed as geometry.

It is also smaller than it first appears. Measured from the USGS STAC catalogue, `view:off_nadir` is 0 for
every Landsat scene in the ITS_LIVE golden set — the instruments do not steer cross-track — so even a
cross-path pair views its overlap at nearly one geometry, and the residual does not double the way an
ascending/descending pair's would.

Where the producer's accuracy is not enough, the remedy is measurement: a shift estimated over stable
ground, which a correlator already produces. Supply it as an offset field and the lattice, the fit and the
correlator wiring above accept it unchanged.

What the projected path *does* gain is a check. [`coregister`](@ref) has never been able to compare
coordinate systems, because `ImageFootprint` carried no CRS; it now carries an optional one, and the
`Rasters` extension supplies it from the raster. Two scenes in different systems have no meaningful
overlap, and both cross-path Landsat pairs in the golden set straddle UTM zones — 32607 against 32608 — so
this is reached in practice.

## API

```@autodocs
Modules = [ImagePairGeometry]
Order = [:type, :constant, :function]
Pages = ["misregistration.jl", "offsetfit.jl", "resample.jl", "radar/rdr2rdr.jl",
         "radar/topsramp.jl"]
```
