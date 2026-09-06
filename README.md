# ImagePairGeometry

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://alex-s-gardner.github.io/ImagePairGeometry.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://alex-s-gardner.github.io/ImagePairGeometry.jl/dev/)
[![Build Status](https://github.com/alex-s-gardner/ImagePairGeometry.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/alex-s-gardner/ImagePairGeometry.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/alex-s-gardner/ImagePairGeometry.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/alex-s-gardner/ImagePairGeometry.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Geometry of a co-registered image pair on a map-projected grid.

For each point of a grid defined in projected coordinates (northing/easting), the geometry of a
pair of images relates map coordinates to image coordinates in both directions:

- forward: grid point `(x, y, z)` to pixel index in each image of the pair,
- inverse: a pixel displacement between the two images to a velocity in map coordinates.

Derived per grid point alongside that mapping: the expected displacement implied by a reference
velocity field, the search extent in pixels implied by a search-range field, chip size converted
from meters to pixels, and the scale factors relating image-space distance to ground distance.

Radar and Cartesian (e.g. optical) imagery differ in how the forward mapping is obtained — orbit
state vectors and a DEM for the former, map projection information for the latter — and in the
unit vectors that define the inverse operator.

The pixel displacement itself is estimated elsewhere, by a feature-tracking or correlation
routine; this package supplies the geometry that routine searches within and the operator that
converts its result to velocity.

## The two paths

Both coordinate systems are implemented and verified against the compiled reference.

`ProjectedCoordinate` describes an image on a map projection — optical imagery, or a geocoded radar
product. `RadarCoordinate` describes one in slant-range/azimuth, where a grid point reaches a pixel
index through orbit state vectors and a range–Doppler solve rather than an affine relation. The
per-point kernel and every output expression are shared; the two differ in how the forward mapping is
obtained and in which spacings the outputs divide by.

They differ enormously in cost. A projected point is around 19 ns under an identity transform and 990 ns
across a real reprojection, where PROJ is roughly 95% of the work. A radar point is about 5.9 µs, of
which the solve is 88% and PROJ 6% — so an ITS_LIVE-sized tile is seconds and a 5000×5000 grid a couple
of minutes, and threading rather than transform caching is the lever that matters there.

`REFERENCE.md` records the exactness standard held against the reference on each path, and every
deliberate divergence.

## Approximating the projection

This applies to the projected path. On the radar path PROJ is 6% of a point rather than 95%, so there
is little for a lattice to remove — see `benchmark/radar_scale_perf.jl`.

Where the grid and the imagery are in different CRSs, the projection library is most of the run: three
calls per grid point, and a continental grid has hundreds of millions of them. `InterpolatedTransform`
evaluates the transform on a coarse lattice and interpolates between the nodes, with a selectable
kernel (`Bilinear`, `Bicubic`, `NearestNode`) and two modes:

```julia
tf = InterpolatedTransform(fast_transform(3413, 32624), grid, pair;
                           lattice = 4, mode = :hybrid, window = win)
r = pairgeometry_blocked(grid, pair, source; transform = tf, window = win, ntasks = 8)
```

`:hybrid` interpolates only the two inverse calls, keeping the pixel-location bands bitwise identical
to the exact path, for about 1.6×. `:full` interpolates both directions for up to 6.9×, at a difference
of at most one pixel in those bands. The exact path stays the default;
[`docs/interpolated-transform.md`](docs/interpolated-transform.md) records the measured cost and
accuracy of each mode, spacing and band.

## Citing

See [`CITATION.bib`](CITATION.bib) for the relevant reference(s).
