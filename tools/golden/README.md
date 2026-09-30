# The golden tests: geogrid, and the handoff to AutoRIFT.jl

`s3://its-live-data/test-space/golden/` holds the acceptance products for the Python autoRIFT — 22
ITS_LIVE granules built by `hyp3_autorift` 0.28.4 from the job list in
[`autorift_golden.json.j2`](https://github.com/ASFHyP3/hyp3-testing/blob/develop/hyp3_testing/templates/autorift_golden.json.j2).
This harness checks that for those pairs, on production data, what this package computes and what
AutoRIFT.jl hands the correlator are what Python autoRIFT computed and received.

`REFERENCE.md` records agreement against *fixtures*: the compiled reference driven on synthetic
scenes, with inputs this repository chooses. That cannot exercise a real acquisition's parameter
rasters, a real nodata pattern, a reprojected scene, or the two files of Python between geogrid's
output and the correlator's input. This does.

## What runs, and what each layer establishes

Four layers. Each is a set of rungs, and every rung is fed the reference's own input and diffed
against the reference's own output — never against another rung's result, which would rebuild the
composed comparison the layers exist to take apart.

```bash
julia --project=tools/golden tools/golden/params.jl              # cache the parameter windows
julia --project=tools/golden -t 8 tools/golden/geogrid_optical.jl
julia --project=tools/golden -t 8 tools/golden/geogrid_radar.jl --clock
julia --project=tools/golden -t 8 tools/golden/geogrid_radar.jl
julia --project=tools/golden tools/golden/handoff.jl
julia --project=tools/golden -t 8 tools/golden/pointset.jl
```

| layer | what it compares | coverage |
|---|---|---|
| 1 optical | `pairgeometry_blocked` against the nine delivered `window_*.tif` | 18 runs, 12 cases |
| 1 radar | the same, at a clock solved from the reference's own azimuth indices | 8 runs |
| 2 handoff | the geogrid GeoTIFFs against `capture/in_*`, the correlator's own inputs | 26 runs |
| 3 pointset | this package's geometry, converted by `AutoRIFT.pointset`, against `capture/in_*` | 18 runs |

Layer 2 is `testautoRIFT.py`'s and `autoRIFT.runAutorift`'s work rather than this package's, and it
is here because it decides whether a correct geometry ever reaches the correlator correctly. Layer 3
composes Layers 1 and 2 through the conversion between them, so a difference there that both of them
pass is a difference in `AutoRIFT.pointset`.

## Results

One sweep of `run_all.jl`: **1,090 of 1,090** band and rung verdicts pass, across 70 run-comparisons
and all five layers.

| layer | runs | verdicts |
|---|---|---|
| 0 self-gate | — | 290 assertions, 8 injected differences each caught |
| 1 optical | 18 | 18 bands each |
| 1 radar | 8 | 21 bands each |
| 2 handoff | 26 | 14 rungs each |
| 3 pointset | 18 | 13 rungs each |

Optical, all 18 runs, every band within its gate:

| | |
|---|---|
| grid points | 72,317,917 |
| computed on both sides, counts equal | 61,914,335 |
| the eleven `Int32` bands | **bitwise**, one point excepted |
| the eight `Float64` bands | worst 1.8e-8 against a 1e-7 bound |
| same-CRS case (`LC09`, EPSG:3031 throughout) | **bitwise on all eight float bands** |

The handoff, all 26 runs, every rung agreeing: 127,007,808 grid points and 1,016,062,464
array-point comparisons, covering the grid convention, the truncation, the pass-through of six arrays
and three scalars.

The `PointSet`, all 18 optical runs, every rung agreeing over 35,820,289 searched points. The index
convention is measured rather than argued: all 36 scans (x and y × 18 runs) land on the same offset
with every point matching and every other offset matching none.

## The gates, and the measurement behind each

A tolerance is a claim requiring a measurement. Every gate below that is not "bitwise" has one, and
the number that forced it is recorded beside it in the source.

**Tier A, the integer bands — bitwise.** These pass through a rounding or truncating conversion that
absorbs any last-bit difference in their inputs, so exact agreement is achievable and anything less
is a real difference. One exception across 18 optical runs: `search_x` at (1132, 1489) of
`LC08_L1TP_062018` evaluates to `-34.499999762929185`, 2.4e-7 below a `.5` boundary, so a 6.9e-9
relative difference — inside what the same case's float bands show — rounds it the other way.
`compare_int_band`'s `allow_boundary` permits a bounded number of such points and only when the
difference is *also* one.

**Tier B, the float bands — 1e-7 relative.** `test/geogrid.jl`'s bound, holding on production data at
the same value. Two things compose to set its size. The kernel builds each axis unit vector from the
*difference* of two inverse-transformed coordinates one pixel apart, amplifying any transform error by
`|x| / spacing` — about 1e5 at ITS_LIVE scale — and `off2vy_dy` then divides by `xunit[1]`, which is
near zero wherever the image axes are nearly perpendicular to the grid's. Measured across three cases:
`xunit[1]` of 0.99, 0.10 and 0.003 give 1×, 9.6× and 332× amplification of one shared absolute error,
matching the observed 1e-9, 1.3e-7 and 4.2e-6 self-normalized differences.

That is why `compare_float_band` normalizes the four components of one displacement-to-velocity
operator by the operator's largest component rather than each by itself. They share the unit vectors
they are computed from, so an absolute error in those is shared; normalizing each by itself reports
`off2vy_dy` at 4.2e-6 and its sibling `off2vy_dx` at 1.3e-11 for one error in one operator.

Substituting PROJ for FastGeoProjections drops the band to exactly zero at most points, which is what
identifies the transform rather than the kernel as the source. The coordinates themselves agree to
3.7e-15 relative for every pair the golden set uses; the one-pixel difference the unit vectors are
built from agrees to 2.1e-11 for EPSG:3413↔32622 and only 2.7e-9 for 3413↔32607.

**The radar index bands — one index on ≤ 18% of computed points in range, ≤ 6% in azimuth.** Looser than
the fixture bound, and the reason is the log rather than the kernel. `dr` prints as `2.32956`: six
figures, so ±5e-6, which is 0.14 index units at the far edge of a 66,000-sample swath. The range index
inherits that directly and cannot be bitwise while `dr` is read from the log.

Measured over the eight runs as a fraction of the points the reference computed, range differs on
6.73–15.06% and azimuth on 0.35–4.13%, with *every* disagreement on either exactly one. Range being
consistently four times worse is what identifies `dr` as the dominant term rather than the clock, and
why the two axes get separate bounds: one wide enough for range would be six times looser than azimuth
needs and would stop saying anything about azimuth.

Fitting `dr` was tried and rejected. Scanning it lifts the range agreement to a peak of 94.6%, but
the peak sits at 2.329569, which renders as `2.32957` — outside the interval the log's own `2.32956`
admits. A fitted value that contradicts the printed one is absorbing some other error, so the printed
value stands and the bound is stated instead.

## What the log cannot tell you

The container's stdout is the only record of the parameters `runGeogrid` was called with, and it is
lossy in three ways that each produce a plausible wrong answer rather than a failure.

**Six significant figures.** `GeogridOptical.cpp` prints through C++ `ostream`, so an origin arrives
as `-2.12881e+06` where the value is `-2128807.5`. A half-pixel error in the image origin displaces
every pixel index by a residual that is zero under uniform motion and grows with the velocity
gradient — invisible in any summary statistic. So `scenes.jl` rebuilds the overlap by intersecting the
two scenes with `coregister`, the same intersection `GeogridOptical.py:240-300` performs, and the
log's line is the cross-check: all 18 optical runs agree with their own log's printed origin, spacing
and dimensions.

Where no catalogue describes the scene — the four cases whose scenes are warped into a common CRS
first — the origin is recovered from the log anyway, because `gdal.Warp` is called with
`targetAlignedPixels=True` and only one multiple of the resolution renders to the six figures printed.
Checked over a ±1000-pixel search: exactly one candidate each.

**Interleaving.** Container stdout mixes streams, so one run reads
`Range: 803581  2.3295Using granule search...` and another has a parameter URL truncated mid-token to
`SPS_0Polarization hh`. Every numeric field is matched with a pattern that must consume the whole
value, and each truncated field is recovered from an independent source: the SLC reader's own line for
`dr`, the output rasters' band metadata for the nodata sentinel, and the DEM's tile for a URL. A
progress line can also overwrite a *block header* — one radar run has ` parameters: ` where the rest
have `Radar parameters: ` — so a run's path is identified by the fields inside its block.

**The clock is absent entirely.** `sensing_start` to better than a PRI is nowhere on disk: the log's
`aztime` is good to ±24 lines, and the product's `acquisition_date_img1` has microseconds but sits
1.5–3.4 s away across the eight cases, so it is a different time. `solve_sensing_start` recovers it
from `window_location.tif` band 2 — which *is* the azimuth index geogrid computed — by scanning whole
lines then eighths. Every case peaks at 96–99.7% with the nearest rival a full line away at 8–9%.

That spends the azimuth band as evidence: it can no longer corroborate the result. The range index,
the chip sizes, the mask and every float band remain independent, and those are what the radar layer
establishes.

## What the captures cannot tell you

`capture/in_*` is the ground truth for the handoff, and one thing in it is not reproducible. The
driver zeroes a grid point for two reasons: geogrid's own nodata sentinel, and the imagery marking the
point unusable. The second takes one of two branches and neither is available:

- on an ordinary run it reads the image samples, and `capture.py` takes `in_I1` at the `runAutorift`
  boundary — downstream of the Wallis or FFT prefilter, which changes which samples are zero;
- on a `wallis_fill` run it looks up the `_zeroMask` raster the fill wrote, which the capture omits.

So `captured_skip_mask` reads that mask off the capture and Layer 2 reports its size rather than
gating on it — a third of the grid on some cases, which a comparison folding it into the nodata mask
would describe wrongly. What Layer 2 does gate is that geogrid's own nodata is a subset of it, and
everything else about the derivation.

Two details of that mask were found only by measuring against the captures. Upstream autoRIFT 2.1.2
carries a comment saying `wallis_fill` disables the image-zero rule; the container's vendored
`testautoRIFT.py:337-349` substitutes the raster lookup instead, and a `wallis_fill` run zeroes 1.59
million points the upstream source predicts none of. And `chopFactor` is the largest chip size *after*
that mask has zeroed its own points: on `LC08_L1TP_060018` all 162,117 points holding 128 are zeroed,
so the factor is 4 rather than 8 and the grid is 1760×2076 rather than 1760×2072 — four columns
entering every array the correlator receives.

## Coverage, and what is not covered

| | cases | runs |
|---|---|---|
| optical, Layers 1–3 | 12 | 18 |
| radar, Layers 1–2 | 8 | 8 |
| NISAR | 2 | none — no run directory |

The two NISAR pairs are in the manifest and in `products/` but have no local run, so they have neither
`window_*.tif` nor a capture.

Producing the L1 run was attempted and did not complete. What was observed, and nothing more: the
container reached `Topo progress (block 6/6): 100%` — the end of ISCE3 geocoding — and the pixi process
was then `Killed`, with 174 GB written into the run directory. The log names no cause: no `bad_alloc`,
no `MemoryError`, no out-of-space message.

**The cause is not diagnosed.** Memory is the obvious guess and the evidence does not support it: an S1
case succeeded on the same 31 GiB Docker allocation with a 5.9 GB intermediate on a 65978 × 23857 image,
where the NISAR geocoding blocks were 3541 × 1266, and the kill landed *after* that stage rather than
inside an allocation. Disk is equally consistent — 174 GB had been written — and was not sampled during
the run. Diagnosing it means rerunning with `docker stats` and `df` sampled throughout.

So the honest statement is that this pair needs a rerun instrumented to say what it ran out of, not that
it needs a larger machine.

`test/radar_itslive_product.jl` cross-checks the L1 pair's `M11`/`M12` and `dr_to_vr_factor` against
its delivered product independently, so the radar operator is not unchecked there — that test needs
only the product and the parameter rasters, not a rerun of the pipeline.

Layer 3 covers the optical runs only. It needs a `PairGeometry` from Layer 1, and the radar geometry is
conditional on a solved clock — so running it there would report the clock's residual as a conversion
difference.

## Data this needs

Around 40 GB, none of it committed:

- `~/data/autorift/tests/golden_tests/runs/<product>/<n>/` — the geogrid output, the capture, the log,
  and the `.EOF` orbits. Produced by `AutoRIFT.jl/tools/golden/intermediate.jl`.
- `~/data/autorift/tests/golden_tests/params_cache/` — written by `params.jl`: each run's twelve
  parameter rasters windowed to its geogrid window, plus cached STAC items and raster headers. About
  2.3 GB, and offline once populated.
- `AutoRIFT.jl` beside this repository, for `manifest.json` and `tools/ab/xchg.jl`. Set
  `IPG_GOLDEN_AUTORIFT` if it lives elsewhere.

This is a tool, not part of `Pkg.test()`. Any change to `src/` that a golden finding prompts needs a
fixture test added alongside it, since the harness cannot be the only thing holding a fix in place.
