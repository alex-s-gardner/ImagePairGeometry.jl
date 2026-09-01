"""
    ImagePairGeometry

Geometry of a co-registered image pair on a map-projected grid.

For each point of a grid defined in projected coordinates (northing/easting), the geometry of
a pair of images relates map coordinates to image coordinates in both directions:

- forward: grid point `(x, y, z)` to pixel index in each image of the pair,
- inverse: a pixel displacement between the two images to a velocity in map coordinates.

Derived per grid point alongside that mapping: the expected displacement implied by a
reference velocity field, the search extent in pixels implied by a search-range field, chip
size converted from meters to pixels, and the scale factors relating image-space distance to
ground distance.

Radar and Cartesian (e.g. optical) imagery differ in how the forward mapping is obtained —
orbit state vectors and a DEM for the former, map projection information for the latter — and
in the unit vectors that define the inverse operator.

The pixel displacement itself is estimated elsewhere, by a feature-tracking or correlation
routine; this package supplies the geometry that routine searches within and the operator that
converts its result to velocity.

`REFERENCE.md` records the reference implementation and every deliberate divergence from it.
"""
module ImagePairGeometry

using StaticArrays: SVector
using Extents: Extent

export ImageFootprint, CoregisteredPair, coregister
export ProjectedCoordinate, RadarCoordinate
export MapGrid, footprint_bounds, grid_window
export IdentityTransform, AffineTransform, TransformPair, transform_pair
export NoDataPolicy, nodata_from
export PairGeometry, GeometryInputs, GeometryParams, SearchRangeScaling
export pairgeometry, npoints, nvalid

include("kernel/vecmath.jl")
include("kernel/rounding.jl")
include("coordinates.jl")
include("transforms.jl")
include("pair.jl")
include("grid.jl")
include("kernel/searchrange.jl")
include("kernel/geometry.jl")
include("kernel/outputs.jl")
include("nodata.jl")
include("result.jl")
include("driver.jl")

end
