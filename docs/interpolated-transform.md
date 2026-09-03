# Interpolating the coordinate transform

An opt-in mode. The exact path is the default, and this trades a stated amount of accuracy for
throughput on a cross-CRS run.

The projection library is around 88% of a cross-CRS run (`benchmark/cost_share.jl`), and the only
lever on it is making fewer calls. The kernel makes three per grid point: one forward, and two
inverses for the point stepped one pixel along each image axis. A coordinate transform between two
map projections is smooth over a few hundred meters, so it can be evaluated on a coarse lattice and
interpolated.

`InterpolatedTransform` wraps any transform the exact path accepts and returns a `TransformPair`
whose directions are `CoordLattice`s. A lattice is callable as `(x, y, z) -> (x′, y′, z′)`, so it
substitutes for a transform and the kernel is unchanged.

```julia
tf = InterpolatedTransform(ProjTransformFactory(3413, 32624), grid, pair;
                           lattice = 4, mode = :hybrid, interpolation = Bilinear(), window = win)
r = pairgeometry_blocked(grid, pair, source; transform = tf, window = win, ntasks = 8)
```

`lattice` is the node spacing as a multiple of the grid spacing. `interpolation` is `Bilinear()`,
`Bicubic()` or `NearestNode()`. It is an `AbstractTransformFactory`, so a blocked run calls it once
per task and each task owns its lattice and its PROJ context for that task's lifetime.

The lattice is built from the whole window's bounds, not per block, so a blocked result does not
depend on the block size — the invariant `pairgeometry_blocked` documents. Node coverage is derived
by sampling the window's perimeter through the forward transform and adding a pixel for the kernel's
one-pixel step, because a projection maps a straight window edge to a curved one and the kernel
queries the inverse before `inbounds` rejects anything.

## What it costs and what it buys

Measured on an EPSG:3413 grid at 120 m against UTM 32624 imagery at 30 m, 676,506 points, serial.

| mode | lattice | kernel | PROJ calls/point | ns/point | speedup |
|---|---|---|---|---|---|
| exact | — | — | 3 | 943 | 1.00× |
| `:hybrid` | 2 | `Bilinear` | 1 | 631 | 1.49× |
| `:hybrid` | 4 | `Bilinear` | 1 | 584 | 1.61× |
| `:hybrid` | 8 | `Bilinear` | 1 | 575 | 1.64× |
| `:full` | 2 | `Bilinear` | 0 | 296 | 3.19× |
| `:full` | 4 | `Bilinear` | 0 | 169 | 5.58× |
| `:full` | 8 | `Bilinear` | 0 | 137 | 6.90× |
| `:full` | 4 | `Bicubic` | 0 | 200 | 4.71× |
| `:full` | 8 | `Bicubic` | 0 | 168 | 5.51× |

`:hybrid` keeps one PROJ call per point, so it saturates around 1.6× however coarse the lattice —
the remaining call is the floor, and at a 4× lattice it is 57% of the run. `:full` removes that
floor, and then coarsening keeps paying.

What remains in `:full` is split between the per-point interpolation and building the lattice, which
is a fixed cost amortized over the window: at a 4× lattice the build is 28% of the run, at 8× it is
11%, and it shrinks further as a scene grows. The rest is the driver's own per-point work — reading
the input rasters, the geometry arithmetic, and writing seventeen output bands — which an identity
transform measures at 108 ns/point and which no transform work can remove.

Reproduce with `benchmark/cost_share.jl`, which also reports PROJ's share of the exact run.

## Which bands stay exact

The exactness requirement and the cost sit on *different* transforms. `location_x` and `location_y`
come from the forward transform alone, and they are the Tier A bands — bitwise agreement with the
reference, per `REFERENCE.md`. So `:hybrid` keeps the forward exact and interpolates only the two
inverses, which is where two thirds of the cost is.

| band | source | `:hybrid` | `:full` |
|---|---|---|---|
| `location_x`, `location_y` | forward only | bitwise | ±1 pixel |
| `offset_*`, `search_*` | inverse, via the velocity operator | ±1 | ±1 |
| `chip_min_*`, `chip_max_*`, `stable_surface` | neither transform | bitwise | bitwise |
| `off2vx_*`, `off2vy_*`, `scale_*` | inverse | relative, below | relative, below |

The ±1 on `offset` and `search` is not a failure of accuracy. The positional error at a 480 m
lattice is 0.8 mm, 18,000× smaller than the half-pixel margin rounding to a pixel index allows; what
changes is which side of a `.5` boundary a point falls on. On the fixtures in `test/geogrid.jl` it
first appears at 8× with `Bilinear` and not at all through 16× with `Bicubic`, so "exact at 4×"
describes where ties happen to fall in those scenes rather than a margin to rely on.
`test/interpolate.jl` asserts a bound of one, not equality.

The float bands degrade as the square of the lattice spacing, as bilinear interpolation of a smooth
function must. Worst difference over the band, normalized by the band's own maximum, on a 120 m grid
(`off2vx_dy` passes through zero, so a pointwise relative error there would report how near zero the
denominator got rather than how wrong the value is):

| lattice | `off2vx_dx` | `off2vx_dy` | `scale_x` | `scale_y` |
|---|---|---|---|---|
| 1× | 1.5e-8 | 1.7e-5 | 1.5e-7 | 1.7e-6 |
| 2× | 6.0e-8 | 6.6e-5 | 6.0e-7 | 6.8e-6 |
| 4× | 2.4e-7 | 2.7e-4 | 2.4e-6 | 2.7e-5 |
| 8× | 9.6e-7 | 1.1e-3 | 9.6e-6 | 1.1e-4 |

Tier B's bound for a reprojected float band is 1e-7 relative, which exists to absorb compiler
contraction and PROJ's platform variation on the exact path. A lattice consumes far more than that
at any spacing, so `test/interpolate.jl` asserts its own stated bounds against the exact path rather
than widening Tier B.

## Choosing a mode

`:hybrid` is the one to reach for by default: about 1.6× with the location bands bitwise, so the
search centers a correlator is handed are the same ones the exact path would give.

`:full` is worth it where a one-pixel difference in a search center is acceptable, which for a
correlator searching tens of pixels it may well be. `Bicubic` at 8× is the interesting corner —
5.51× with no location shift at all on the fixtures, against `Bilinear` at 8× which is faster at
6.90× but shifts a handful of points. Cubic convolution reproduces a quadratic exactly where
bilinear does not, so it holds a given accuracy at a coarser lattice; the extra per-point arithmetic
is cheaper than the PROJ calls a finer lattice would need to build.

`NearestNode` is not a production option. It exists to separate the error the lattice's coarseness
carries from the error the interpolation adds, by being the version that does no interpolation at
all.

## Elevation

Where the transform's horizontal result moves with elevation, the lattice is tabulated at both ends
of `zrange` and interpolated linearly between them. That is exact rather than approximate: elevation
enters a horizontal coordinate only through a datum shift, which is linear in it.

Where it does not, one level is tabulated and a query reads it directly. Which case holds is
established by probing the two `zrange` extremes at the lattice's corners and center, bitwise: a
difference of any size means the pipeline carries a vertical component. For two CRSs on one datum,
which is every ITS_LIVE projection, there is no such shift, so this halves the transform calls a
lattice costs to build. Worth having because the build is a real share of a `:full` run rather than a
setup detail: 28% at a 4× lattice on the benchmark scene.

## What does not pay

Batching through `proj_trans_generic` is 1.00×: the cost is the projection math, not the call. PROJ
gains nothing from sequential over random points, so a row-ordered sweep buys nothing either.

GDAL's `GDALApproxTransformer` is the same idea — evaluate exactly at the ends, interpolate between,
bisect where the error is too large — but it requires the points handed to it to be collinear and
ordered, for a warper working one scanline at a time. This kernel queries one point at a time, and
its two inverse queries sit off the line the forward queries trace, so there is no line to hand it.
Using it would also mean a second PROJ, whatever `GDAL_jll` bundles, where `REFERENCE.md` pins the
version because `proj_create_crs_to_crs`'s operation selection is not stable across them.
