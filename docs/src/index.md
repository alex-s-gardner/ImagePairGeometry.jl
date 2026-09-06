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

Both coordinate paths are implemented and verified against the compiled reference:
[`ProjectedCoordinate`](@ref) for imagery on a map projection, and [`RadarCoordinate`](@ref) for
slant-range/azimuth imagery. [`pairgeometry`](@ref) and [`pairgeometry_blocked`](@ref) take either, and
produce the reference's nine outputs on both.

The per-point kernel and the output expressions are shared. What differs is how the forward mapping is
obtained — a coordinate transform against an orbit and a range–Doppler solve — and which spacings the
outputs divide by. See [Radar geometry](radar.md).

`REFERENCE.md` records the exactness standard held on each path, including two radar float bands that
agree to 1.07e-4 rather than bitwise, and the measurements that place the cause outside this package.

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
source = InMemoryInputs(inputs, window)
result = pairgeometry_blocked(grid, pair, source;
                              transform = fast_transform(3413, 32624),
                              blocksize = (512, 512), ntasks = 8)
```

One transform object serves every task: a [`fast_transform`](@ref) pair is immutable and holds no
library state, so nothing has to be built per task.

This returns the whole window as one [`PairGeometry`](@ref), which is 108 bytes per grid point
whatever the block size. Where that does not fit — a continental grid is hundreds of gigabytes of
output — pass a `sink` and the result is written or consumed block by block instead, at one block per
task. See [Streaming the output](streaming.md).

Where the grid and the imagery share a CRS, pass [`IdentityTransform`](@ref) instead — it is exact by
construction and skips the projection entirely.

A radar pair transforms between the grid CRS and geodetic degrees rather than between two projected
CRSs, which needs no different call — `fast_transform` returns longitude/easting first by default:

```julia
result = pairgeometry_blocked(grid, pair, source;
                              transform = fast_transform(32632, 4326),
                              blocksize = (512, 512), ntasks = 8)
```

## Coordinate transforms

[`fast_transform`](@ref) returns a [`TransformPair`](@ref) evaluated by
[FastGeoProjections](https://github.com/alex-s-gardner/FastGeoProjections.jl), for the EPSG pairs it
implements natively — polar stereographic, UTM, and the geographic and geocentric systems. This is the
only transform the package provides: nothing in `src/` links a projection library, so there is no PROJ
dependency and no PROJ context to own per thread.

```julia
tf = fast_transform(3413, 32624)
r = pairgeometry_blocked(grid, pair, source; transform = tf, ntasks = 8)
```

On the projected path it is 178 ns/point against PROJ's 445, measured on the kernel's own call pattern
— and there the transform is most of the run, so that ratio is close to the run's. On the radar path
the two geometry solves dominate and the transform is around 13% of a point, so the choice is worth a
few percent either way; measured on EPSG:32632 to EPSG:4326 the native forward is in fact slightly
slower than PROJ's. Pick it for having no dependency, not for speed.

## Axis order

`always_xy` defaults to **`true`**: a coordinate pair is `(easting, northing)` or
`(longitude, latitude)`, whatever the CRS's authority order says.

For a projected-to-projected pair the flag is inert — the EPSG codes this package is used with order a
projected pair easting-first regardless — so it changes nothing the reference fixtures pin. Where it
decides something is a geographic CRS: EPSG:4326's authority order is `(latitude, longitude)`.

```julia
julia> fast_transform(32632, 4326).forward(-242500.0, 2179000.0, 500.0)
(1.9350326147365111, 19.567422331060865, 500.0)          # (lon, lat) -- the default

julia> fast_transform(32632, 4326; always_xy = false).forward(-242500.0, 2179000.0, 500.0)
(19.567422331060865, 1.9350326147365111, 500.0)          # (lat, lon) -- authority order
```

The default is not the reference's, which uses `osr.CoordinateTransformation` on SRSs from
`ImportFromEPSG` — authority order. It is chosen against the failure mode instead: the radar path maps
grid coordinates to geodetic degrees and reads the result as `(lon, lat, h)`, so an authority-order
transform swaps the two and produces geometry that is wrong everywhere *without raising anything* — the
solve converges happily against a target on the wrong side of the planet. Defaulting to the order the
kernel wants removes a silent failure whose only symptom is bad output.

Pass `always_xy = false` to reproduce authority order deliberately. If you build a `TransformPair` from
PROJ yourself, note that `Proj.Transformation` defaults the other way, so a radar pair needs
`always_xy = true` passed explicitly there.

Both CRSs must be given as EPSG codes, since FastGeoProjections resolves a transformation by code
rather than parsing a description. For a WKT or PROJ string, or a code it does not implement, build a
[`TransformPair`](@ref) from any library that does — `Proj.jl` among them — and pass that:

```julia
using Proj
tf = TransformPair(Proj.Transformation("EPSG:3413", "EPSG:32624"),
                   Proj.Transformation("EPSG:32624", "EPSG:3413"))
```

A `Proj.Transformation` wraps a context that is not safe to share between threads, so a threaded run
must pass a zero-argument factory returning a fresh pair rather than one built up front. The test
suite does this — `Proj.jl` is a test dependency, since the reference fixtures were generated through
PROJ and are asserted against it.

## Approximating the projection

Where the CRSs differ, the projection is most of a projected run: three calls per grid point.
[`InterpolatedTransform`](@ref) tabulates the transform on a coarse lattice and interpolates between
the nodes, with a selectable kernel — [`Bilinear`](@ref), [`Bicubic`](@ref) or [`NearestNode`](@ref).

```julia
tf = InterpolatedTransform(fast_transform(3413, 32624), grid, pair;
                           lattice = 4, mode = :hybrid, interpolation = Bilinear(),
                           window = window)
result = pairgeometry_blocked(grid, pair, source; transform = tf, window = window, ntasks = 8)
```

`lattice` is the node spacing as a multiple of the grid spacing. The two modes differ in which bands
they leave exact:

- `:hybrid` interpolates only the two inverse calls. The pixel-location bands come from the forward
  transform alone, so they stay bitwise identical to the exact path. Saturates around 1.6×, since
  the one remaining call is the floor.
- `:full` interpolates both directions, for up to 6.9×. The location bands can then differ by one pixel
  — the positional error is far below a pixel, but it can move a point across a rounding boundary.

It is an [`AbstractTransformFactory`](@ref), so a blocked run calls it once per task and each task owns
its own lattice. The result does not depend on the block size: the lattice is built from the whole
window's bounds.

The exact path is the default. `docs/interpolated-transform.md` records the measured cost and accuracy
of each mode, spacing and band.

## API

Everything exported, plus the internals the reference-fidelity notes refer to. Grouped by the file
each is defined in, which follows the layering: vector and rounding primitives, then coordinates and
transforms, then the grid, then the kernel, then the driver.

The radar path's own numerics — the ellipsoid, orbit interpolation, and the two geometry solves —
are on the [Radar geometry](radar.md) page; the output sink protocol is on
[Streaming the output](streaming.md).

```@autodocs
Modules = [ImagePairGeometry]
Order = [:module, :type, :constant, :function, :macro]
Pages = ["kernel/vecmath.jl", "kernel/rounding.jl", "coordinates.jl", "transforms.jl",
         "fasttransform.jl", "pair.jl", "grid.jl", "kernel/searchrange.jl", "kernel/geometry.jl",
         "kernel/outputs.jl", "nodata.jl", "result.jl", "driver.jl", "blocks.jl",
         "interpolate.jl", "ImagePairGeometry.jl"]
```

## Raster IO

Available when `Rasters`, `ArchGDAL`, `DimensionalData` and `DiskArrays` are loaded. Reads inputs a
window at a time from disk-backed rasters, and writes the reference's nine GeoTIFFs.

```@autodocs
Modules = [Base.get_extension(ImagePairGeometry, :ImagePairGeometryRastersExt)]
Order = [:type, :function]
```

## Correlator handoff

[`velocity_conversion`](@ref) gathers the per-point quantities that turn a measured pixel
displacement into a map velocity: the operator, the scale factors, the stable-surface mask, the
chip-size aspect ratio, and [`y_displacement_sign`](@ref) — the factor a y displacement needs when the
image's second axis opposes north, as a radar image's azimuth does.

Building the correlator's search grid is the correlator's own concern, so it lives there rather than
here. [`AutoRIFT`](https://github.com/alex-s-gardner/AutoRIFT.jl) adds a
`pointset(::PairGeometry)` method when both packages are loaded, which reindexes the grid to
one-based and turns a nodata point into the zero search radius it skips on. This package has no
dependency on it in either direction.

## Index

```@index
```
