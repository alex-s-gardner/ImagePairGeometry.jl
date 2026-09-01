# Interpolating the coordinate transform

Not implemented. This records a measured option and why it is not the default, so the question does
not have to be re-investigated from scratch.

The projection library is 88.6% of a cross-CRS run (`benchmark/cost_share.jl`), and the only lever on
it is making fewer calls. The kernel makes three per grid point: one forward, and two inverses for the
point stepped one pixel along each image axis. A coordinate transform between two map projections is
smooth over a few hundred meters, so it can be evaluated on a coarse lattice and interpolated —
the approach GDAL's own `gdalwarp` takes with `GDALApproxTransformer`.

Prototyped as a bilinear lattice standing in for a `TransformPair`, so the kernel itself needs no
change. Measured on an EPSG:3413 grid at 120 m against UTM imagery at 30 m.

## What it costs and what it buys

| mode | PROJ calls/point | ns/point | speedup | integer bands |
|---|---|---|---|---|
| exact | 3 | 861 | 1.00× | bitwise |
| exact forward, interpolated inverses | 1 | 527 | **1.63×** | **bitwise** |
| fully interpolated | 0 | 94 | **9.2×** | see below |

The float bands (`off2vx`, `off2vy`, `scale_factor`) degrade smoothly with lattice spacing, in
relative terms, on a 120 m grid:

| lattice | off2vx | scale |
|---|---|---|
| 240 m | 6.0e-8 | 6.0e-7 |
| 480 m | 6.2e-7 | 6.2e-6 |
| 960 m | 3.2e-6 | 3.3e-5 |
| 7680 m | 2.8e-4 | 2.3e-3 |

## Why full interpolation is not the default

`window_location` and `window_location_y` flip by exactly one pixel once the lattice reaches 4× the
grid spacing. Measured across five configurations — EPSG:3413 and EPSG:3031 grids, UTM zones 19S/21/24,
image pixels of 10/15/30 m, grid spacing of 120/240 m — the pattern is consistent: exact at 1× and 2×,
one-pixel differences in the location bands at 4× and beyond, and no other band ever differs.

The flip is not an accuracy failure, which is what makes it disqualifying rather than tunable. At a
480 m lattice the worst positional error is **0.8 mm** — 18,000× smaller than the half-pixel margin
that rounding to a pixel index allows. The points that flip are ones whose exact fractional index sits
within that 0.8 mm of a `.5` boundary, where any perturbation tips `cround`. So "exact at 2×" is a
statement about where ties happen to fall in the scenes tested, not a margin that can be relied on: a
grid and image whose spacings put more points near a tie would flip at 2× as well.

Since `window_location` is a Tier A band — bitwise agreement with the reference, per `REFERENCE.md` —
that rules out full interpolation as a default. It remains sound as an opt-in mode for a caller who
has stated that a one-pixel search-center difference is acceptable, which for a correlator searching
±20 pixels it may well be.

## The hybrid is the interesting one

Interpolating only the *inverse* transform keeps every integer band bitwise exact, because the
location comes from the forward transform alone. It drops two of the three PROJ calls for 1.63×, and
confines the approximation to the float bands, where the error is relative and small — 6e-8 on
`off2vx` at a 240 m lattice, against a Tier B tolerance already set at 4 ULP for compiler contraction.

That asymmetry is the useful finding: the exactness requirement and the cost are on *different*
transforms.

## If this is implemented

- A lattice type satisfying the `TransformPair` interface; the kernel needs no change, which is what
  made the prototype a drop-in.
- Build the lattice per block, not per scene: a block is a few hundred points on a side, so the
  lattice is small and its extent is known from the block's own bounds. Per-task ownership, as with
  `ProjTransformFactory`, since it is built from a PROJ transform.
- Gate it against the exact path, not against the reference: assert the integer bands bitwise and
  report the float bands' ULP, the way `test/geogrid.jl` already does.
- Report the lattice spacing and the measured worst-case positional error in the result, so a
  consumer can see what it was given.
