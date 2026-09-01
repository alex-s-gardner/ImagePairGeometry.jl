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

## Citing

See [`CITATION.bib`](CITATION.bib) for the relevant reference(s).
