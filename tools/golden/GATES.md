# The gates, and the measurement behind each

Every gate this harness applies, what it is, and the number that set it. A tolerance is a claim
requiring a measurement; a gate below without one is a bug in this document.

Written so a reader can tell, for any red rung, whether the gate or the code is at fault.

## The rule

Bitwise is the default and a departure needs a cause identified, not merely a magnitude observed. Each
loosened gate below names the mechanism, the measurement, and — where one was tried and rejected — the
alternative that did not hold up.

Two properties are gated on *shape* rather than magnitude, because a magnitude alone cannot distinguish
them from what they must exclude:

- an integer band's disagreements must all be exactly one, whatever the count. A scattered ±1 and a
  rounding boundary produce the same count; only the bound on the magnitude separates them from a real
  index error.
- the clock scan must have one winner far above every rival a full line away. A broad plateau would
  satisfy any threshold on the winner alone while meaning the clock is not determined.

## Layer 1 — geogrid, projected path

| band group | gate | why |
|---|---|---|
| the eleven `Int32` bands | bitwise, ≤ 4 points differing by exactly one | rounding boundary, below |
| the eight `Float64` bands, same CRS | bitwise | no reprojection, so nothing to lose |
| the eight `Float64` bands, cross CRS | 1e-7, operator-normalized | transform amplification, below |
| computed-point count | equal | a coverage difference is a finding, not noise |
| geotransform | equal to the run's own output | asserted in `check_params` before any band |

**The rounding boundary.** One point across all 18 runs. `search_x` at (1132, 1489) of
`LC08_L1TP_062018` evaluates to `-34.499999762929185` — 2.4e-7 below a `.5` boundary, so a 6.9e-9
relative difference in the reference's own value rounds it the other way. That is inside what the same
case's float bands show, so it is the float tolerance reaching an integer band rather than a separate
phenomenon. `BOUNDARY_POINTS = 4` is the same order as the one observed, not a budget.

**Transform amplification.** Two mechanisms compose, and the second is why the *operator* is normalized
as a whole rather than component by component.

The kernel builds each axis unit vector from the difference of two inverse-transformed coordinates one
pixel apart, so a transform error is amplified by `|x| / spacing` — around 1e5 at ITS_LIVE scale.
FastGeoProjections and PROJ agree on the coordinates to 3.7e-15 relative for every pair the golden set
uses, but on that one-pixel difference only to 2.1e-11 for EPSG:3413↔32622 and 2.7e-9 for 3413↔32607.

`off2vy_dy` then divides by `xunit[1]`, near zero wherever the image axes are nearly perpendicular to the
grid's:

| case | `xunit[1]` | amplification | observed, self-normalized |
|---|---|---|---|
| `LC08_L1TP_009011` | +0.99457 | 1× | 1e-9 |
| `LC08_L1TP_062018` | −0.10366 | 9.6× | 1.3e-7 |
| `LC08_L1TP_060018` | −0.00301 | 332× | 4.2e-6 |

The four components of one operator share the unit vectors, so that absolute error is shared — and
normalizing each by itself reports `off2vy_dx` at 1.3e-11 and `off2vy_dy` at 4.2e-6 for one error in one
operator. Normalizing by the operator's largest component reports one number, and the bound then holds
at `test/geogrid.jl`'s own 1e-7 with no widening.

*Attribution, not inference:* substituting PROJ for FastGeoProjections drops the affected band to exactly
zero at most points and 5.6e-10 at the worst. The kernel is unchanged between the two runs.

## Layer 1 — geogrid, radar path

| band group | gate | why |
|---|---|---|
| `location_x`, `offset_x`, `search_x` | one index on ≤ 18% of computed points | `dr` printed to six figures |
| `location_y`, `offset_y`, `search_y` | one index on ≤ 6% of computed points | the azimuth residual |
| chip sizes, stable-surface mask | bitwise | from the parameter rasters, not the solve |
| `off2v*_dx` | 1e-6 | divides by `dr` |
| `off2v*_dy`, `off2v*_dr` | 1e-3 | divides by the along-track step |
| `scale_x` | 1e-7 | |
| `scale_y` | 1e-6 | |
| computed-point count | within one part in 10,000 | 14 of 6,678,195 observed |
| scene-center incidence angle | within one printed digit | the log rounds it to six figures |
| the clock scan | one winner, rivals a line away below 20% | shape, per the rule above |

**Why the index bands are not bitwise here when `REFERENCE.md`'s real-data table has range bitwise.**
The cause is the run's record, not the kernel. `dr` prints as `2.32956` — six significant figures, so
±5e-6, which is 0.14 index units at the far edge of a 66,000-sample swath. Measured over the eight runs,
as a fraction of the points the reference computed:

| band | across the eight runs |
|---|---|
| `location_x` | 6.73% … 15.06%, every disagreement exactly one |
| `location_y` | 0.35% … 4.13%, every disagreement exactly one |

Range is consistently four times worse, which is what identifies `dr` as the dominant term rather than
the clock — the azimuth index is fixed against the reference's own output and still carries the
0.0013-line residual `REFERENCE.md` documents. Separate bounds per axis, because one wide enough for
range would be six times looser than azimuth needs and would stop being a statement about azimuth.

*Rejected:* fitting `dr` rather than reading it lifts range agreement from 86.0% to a peak of 94.6%, but
the peak sits at 2.329569, which renders as `2.32957` — outside the interval the log's own `2.32956`
admits. A fitted value contradicting the printed one is absorbing some other error, so the printed value
stands. Tightening this needs the SAFE annotation, which a golden run does not keep.

**The clock.** Not a tolerance but a recovered quantity, and the one place this harness fits a parameter
to the reference's own output. `sensing_start` to better than a PRI is nowhere on disk, so
`solve_sensing_start` scans it against `window_location.tif` band 2. Every case peaks at 96–99.7% with
the nearest rival a full line away at 8–9%.

That spends the azimuth band as evidence — it can no longer corroborate the result. The range index, the
chip sizes, the mask and every float band remain independent.

## Layer 2 — the handoff

Every rung bitwise, with two exceptions that are reported rather than gated.

| rung | gate |
|---|---|
| 1 the truncation | a valid chop of the grid, read from the capture |
| 2 geogrid's nodata mask | a subset of what the driver zeroed |
| 3 the image mask's size | **reported**, not gated |
| 4 the grid convention | bitwise, `round(v) + 0.5` as `Float32` |
| 5 six pass-through arrays | bitwise, with the radar `Dy0` negation |
| 6 `ChipSize0X`, `GridSpacingX`, `ScaleChipSizeY` | equal to the capture's scalars |

**Why the image mask is reported.** The driver zeroes a point for two reasons and only one is derivable.
The image rule takes one of two branches and neither is available: on an ordinary run it reads the image
samples, and `capture.py` takes `in_I1` downstream of the prefilter that changes which samples are zero;
on a `wallis_fill` run it looks up the `_zeroMask` raster the fill wrote, which the capture omits. So
the mask is read off the capture and its size printed — a third of the grid on some cases, which a
comparison folding it into the nodata mask would describe wrongly.

**Why the truncation is read rather than derived.** `chopFactor` is the largest chip size *after* that
same mask has zeroed its own points. On `LC08_L1TP_060018` all 162,117 points holding 128 are zeroed, so
the factor is 4 rather than 8 — 1760×2076 rather than 1760×2072. What is asserted is that the capture's
shape is a valid chop: a whole number of chips each way, losing fewer than one chip's worth.

**Values, not bit patterns.** Negating a zeroed `Dy0` on the radar path gives `-0.0` where the reference
holds `0.0`. One number, two bit patterns, and a bitwise comparison would report every zeroed point.

## Layer 3 — the `PointSet`

| rung | gate |
|---|---|
| 1 the x and y conventions | one offset matches every searched point |
| 2 the search radii | bitwise, ≤ 4 boundary points |
| 2 the radius sign | no negative radius anywhere on the grid |
| 3 the priors | bitwise |
| 3 the y sign | equals `y_displacement_sign` for the coordinate system |
| 4 the chip bounds | bitwise |
| 5 the base chip extent | uniform, equal to the capture's scalar |
| 6 blocking | identical `PointSet` under a different tiling |

**The conventions are measured.** Each run's x and y are scanned over `{0, ±0.5, ±1}` and the profile
printed before any relation is asserted. All 36 scans — x and y across 18 runs — land on −0.5 with every
point matching and every other offset matching none. `AutoRIFT.jl/tools/golden/intermediate.jl` records
that reasoning about this gave the wrong answer and a scan gave the right one, and that a half-pixel
error is invisible in a median: zero residual under uniform motion, growing with the velocity gradient.

**Only searched points.** `pointset` reads a `PairGeometry`, which carries geogrid's nodata and nothing
else, so at a point the driver zeroed for an image reason the two legitimately disagree. The kept count
is printed on every convention rung so the masking cannot hide a difference silently.

## Not covered

| | why |
|---|---|
| the two NISAR pairs | no local run directory. Producing the L1 run reached the end of ISCE3 geocoding and was killed: 174 GB written, Docker capped at 31 GB of the host's 96 GB |
| Layer 3 on the radar runs | needs a `PairGeometry` whose clock is fitted, so a conversion difference and the clock's residual would be indistinguishable |
| the reference's image mask | not in the capture, per Layer 2 above |

`test/radar_itslive_product.jl` cross-checks the NISAR L1 pair's `M11`/`M12` and `dr_to_vr_factor`
against its delivered product independently, so the radar operator is not unchecked there.
