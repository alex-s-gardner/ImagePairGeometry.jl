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
# The radar path's own vocabulary: a `RadarCoordinate` cannot be constructed without an `Orbit`, a
# `LookSide` and an incidence angle, so these are as public as the type itself. `Ellipsoid` is here
# because `incidence_angle`'s four-argument form takes one; the keyword form defaults it.
export Ellipsoid, Orbit, LookSide, LookLeft, LookRight, incidence_angle
export MapGrid, footprint_bounds, grid_window
export IdentityTransform, AffineTransform, TransformPair, transform_pair
export NoDataPolicy, nodata_from
export PairGeometry, GeometryInputs, GeometryParams, SearchRangeScaling
export pairgeometry, pairgeometry_blocked, npoints, nvalid
export AbstractInputSource, InMemoryInputs, readblock, block_ranges
export AbstractTransformFactory
export InterpolatedTransform, CoordLattice, build_lattice, latticesize
export LatticeInterpolation, NearestNode, Bilinear, Bicubic
export proj_transform
export mapgrid, image_footprint, blocksize_from_chunks, write_geotiffs

include("kernel/vecmath.jl")
include("kernel/rounding.jl")
# The radar numerics. `ellipsoid.jl` needs the three-vector primitives above; the rest build on it in
# order. `radar/coordinate.jl` comes after `grid.jl` instead, since it adds a `footprint_bounds`
# method and needs `MapGrid`'s `DEFAULT_ZRANGE`.
include("radar/ellipsoid.jl")
include("radar/orbit.jl")
include("radar/geo2rdr.jl")
include("radar/rdr2geo.jl")
include("coordinates.jl")
include("transforms.jl")
include("pair.jl")
include("grid.jl")
include("radar/coordinate.jl")
include("kernel/searchrange.jl")
include("kernel/geometry.jl")
# After `kernel/geometry.jl`: the radar `pointgeometry` method returns the `PointGeometry` defined
# there, and uses its `DEG2RAD`.
include("radar/geometry.jl")
include("kernel/outputs.jl")
include("nodata.jl")
include("result.jl")
include("driver.jl")
include("blocks.jl")
# After `blocks.jl`: `InterpolatedTransform` subtypes `AbstractTransformFactory`, so that type must
# exist first.
include("interpolate.jl")

"""
    proj_transform(grid_crs, image_crs) -> TransformPair

A [`TransformPair`](@ref) between two CRSs, built with PROJ.

Defined when `Proj` is loaded. A threaded run wants `ProjTransformFactory` from that extension
instead, so each task builds a transform on its own PROJ context.
"""
function proj_transform end

"""
    mapgrid(dem) -> MapGrid

A [`MapGrid`](@ref) describing a raster's grid. Defined when `Rasters` is loaded.
"""
function mapgrid end

"""
    image_footprint(image) -> ImageFootprint

An [`ImageFootprint`](@ref) describing where a raster sits. Defined when `Rasters` is loaded.
"""
function image_footprint end

"""
    blocksize_from_chunks(source; floor = 256) -> NTuple{2,Int}

A block size aligned to a raster source's own chunking. Defined when `Rasters` is loaded.
"""
function blocksize_from_chunks end

"""
    write_geotiffs(dir, g::PairGeometry) -> Vector{String}

Write a result as the reference's nine GeoTIFFs. Defined when `Rasters` is loaded.
"""
function write_geotiffs end

end
