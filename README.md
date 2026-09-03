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

## Approximating the projection

Where the grid and the imagery are in different CRSs, the projection library is most of the run: three
calls per grid point, and a continental grid has hundreds of millions of them. `InterpolatedTransform`
evaluates the transform on a coarse lattice and interpolates between the nodes, with a selectable
kernel (`Bilinear`, `Bicubic`, `NearestNode`) and two modes:

```julia
tf = InterpolatedTransform(ProjTransformFactory(3413, 32624), grid, pair;
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
