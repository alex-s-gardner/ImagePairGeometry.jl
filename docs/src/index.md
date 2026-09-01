```@meta
CurrentModule = ImagePairGeometry
```

# ImagePairGeometry

Geometry of a co-registered image pair on a map-projected grid.

For each point of a grid defined in projected coordinates (northing/easting), the geometry of a pair
of images relates map coordinates to image coordinates in both directions:

- forward: grid point `(x, y, z)` to pixel index in each image of the pair,
- inverse: a pixel displacement between the two images to a velocity in map coordinates.

Derived per grid point alongside that mapping: the expected displacement implied by a reference
velocity field, the search extent in pixels implied by a search-range field, chip size converted
from meters to pixels, and the scale factors relating image-space distance to ground distance.

The pixel displacement itself is estimated elsewhere, by a feature-tracking or correlation routine;
this package supplies the geometry that routine searches within and the operator that converts its
result to velocity.

A pure-Julia reimplementation of the geogrid module of NASA JPL's autoRIFT, the geometry layer
beneath the ITS_LIVE glacier velocity products. `REFERENCE.md` records the reference implementation,
the exactness standard held against it, and every deliberate divergence.

## Status

The projected-coordinate path is implemented and verified against the compiled reference. The radar
path is not: [`RadarCoordinate`](@ref) exists so the dispatch shape is settled, and throws.

## Walkthrough

Two images, a grid from a DEM, and the per-point geometry of the pair:

```jldoctest walkthrough
julia> using ImagePairGeometry

julia> reference = ImageFootprint(origin = (300000.0, 7800000.0), spacing = (30.0, -30.0),
                                 size = (400, 400));

julia> secondary = ImageFootprint(origin = (300600.0, 7799400.0), spacing = (30.0, -30.0),
                                  size = (400, 400));
```

[`coregister`](@ref) intersects the two footprints. The overlap — not either image alone — supplies
the origin and size that pixel indices are relative to, so the secondary image determines them as
much as the reference does:

```jldoctest walkthrough
julia> pair = coregister(reference, secondary; dt = 91 * 86400.0);

julia> pair.coordinate.origin, pair.coordinate.size
((300600.0, 7.7994e6), (380, 380))

julia> pair.reference_offset, pair.secondary_offset
((20, 20), (0, 0))
```

A grid, then the part of it the pair covers:

```jldoctest walkthrough
julia> grid = MapGrid(geotransform = (295000.0, 120.0, 0.0, 7805000.0, 0.0, -120.0),
                      size = (200, 200), crs = 32624);

julia> window = grid_window(grid, footprint_bounds(IdentityTransform(), pair.coordinate))
CartesianIndices((47:142, 47:142))
```

The inputs cover that window. Only the DEM is required; each other raster gates the outputs that
depend on it:

```jldoctest walkthrough
julia> n = size(window);

julia> inputs = GeometryInputs(dem = fill(500.0, n),
                               dhdx = fill(0.02, n), dhdy = fill(-0.01, n),
                               vx = fill(120.0, n), vy = fill(-80.0, n));

julia> result = pairgeometry(grid, pair, inputs; window);

julia> size(result), nvalid(result)
((96, 96), 9025)
```

The window is a conservative bound on the footprint, so its outermost points can fall outside the
image. Those are nodata rather than an error:

```jldoctest walkthrough
julia> result.location_x[1, 1] == -32767
true

julia> k = CartesianIndex(50, 50);   # comfortably inside

julia> result.location_x[k], result.location_y[k]
(195, 195)

julia> result.offset_x[k], result.offset_y[k]     # expected displacement, pixels
(1, 1)

julia> result.scale_x[k], result.scale_y[k]       # exactly 1: grid and image share a CRS
(1.0, 1.0)
```

The operator converts a pixel displacement to a map velocity. Here a displacement of two pixels in
x and minus one in y:

```jldoctest walkthrough
julia> vx = result.off2vx_dx[k] * 2.0 + result.off2vx_dy[k] * -1.0;

julia> vy = result.off2vy_dx[k] * 2.0 + result.off2vy_dy[k] * -1.0;

julia> round(vx, digits = 2), round(vy, digits = 2)
(240.66, -120.33)
```

### Larger grids

[`pairgeometry_blocked`](@ref) computes the same result block by block, optionally across threads.
It is bit-identical to the unblocked run at any block size and task count, because every grid point
depends only on its own inputs:

```julia
using Proj   # provides ProjTransformFactory

source = InMemoryInputs(inputs, window)
result = pairgeometry_blocked(grid, pair, source;
                              transform = ProjTransformFactory(3413, 32624),
                              blocksize = (512, 512), ntasks = 8)
```

A `ProjTransformFactory` (see [PROJ transforms](@ref)) rather than a built transform: it is called once per task, so each
task holds a PROJ transformation on a context it alone uses. A context is not safe to share between
threads.

Where the grid and the imagery share a CRS, pass [`IdentityTransform`](@ref) instead — it is exact
by construction and skips PROJ altogether, which is around 50 times faster per point.

## API

Everything exported, plus the internals the reference-fidelity notes refer to. Grouped by the file
each is defined in, which follows the layering: vector and rounding primitives, then coordinates and
transforms, then the grid, then the kernel, then the driver.

```@autodocs
Modules = [ImagePairGeometry]
Order = [:module, :type, :constant, :function, :macro]
```

## PROJ transforms

Available when `Proj` is loaded.

An extension module has no name bound in any module a `@autodocs` block can name, so it is fetched
the same way any caller would: `Base.get_extension`.

```@autodocs
Modules = [Base.get_extension(ImagePairGeometry, :ImagePairGeometryProjExt)]
Order = [:type, :function]
```

## Index

```@index
```
