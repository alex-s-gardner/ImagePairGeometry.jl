# The target grid, and which part of it the pair covers.
#
# The grid is defined by a DEM: its geotransform sets the point spacing and origin, its CRS sets
# the projection every output is expressed in. Only the part overlapping the image pair is
# computed, and finding that part is the most error-sensitive arithmetic in the package — a
# one-ULP difference in the bounding box shifts the window by a whole pixel through the
# `floor`/`ceil`, and every output moves with it.
#
# Two steps, deliberately separate because they work in different spaces:
#
#   `footprint_bounds`  image corners -> a bounding box in *grid coordinates*. Needs the
#                       coordinate transform, and is where the elevation range enters.
#   `grid_window`       that box -> a window of *grid indices*. Pure arithmetic.

"""
    MapGrid(; geotransform, size, crs = nothing)

The grid the geometry is computed on, defined by a DEM.

# Fields
- `geotransform`: GDAL's six coefficients `(x₀, dx, rx, y₀, ry, dy)`, where `(x₀, y₀)` is the
  *outer corner* of the first pixel — not its center. Rotation terms `rx`/`ry` must be zero;
  the reference assumes a north-up grid throughout and its index arithmetic is wrong otherwise.
- `size`: `(ncolumns, nrows)` of the DEM.
- `crs`: the projection, or `nothing`. Carried for output metadata; the transform used in the
  computation is passed separately, so this is not consulted during the kernel.

Grid point centers are at `x₀ + (i + 0.5) * dx`, matching the reference
(`geogridOptical.cpp:678-679`).
"""
struct MapGrid{T<:Real,C}
    geotransform::NTuple{6,T}
    size::NTuple{2,Int}
    crs::C

    function MapGrid{T,C}(geotransform, size, crs) where {T<:Real,C}
        all(>(0), size) || throw(ArgumentError("MapGrid size must be positive, got $size"))
        gt = geotransform
        (iszero(gt[3]) && iszero(gt[5])) || throw(ArgumentError(
            "MapGrid requires a north-up grid with zero rotation terms, got " *
            "geotransform[3] = $(gt[3]), geotransform[5] = $(gt[5])"))
        (!iszero(gt[2]) && !iszero(gt[6])) || throw(ArgumentError(
            "MapGrid pixel size must be nonzero, got dx = $(gt[2]), dy = $(gt[6])"))
        all(isfinite, gt) || throw(ArgumentError("MapGrid geotransform must be finite, got $gt"))
        return new{T,C}(gt, size, crs)
    end
end

function MapGrid(geotransform::NTuple{6}, size::NTuple{2,Integer}, crs = nothing)
    T = promote_type(map(typeof, geotransform)...)
    return MapGrid{T,typeof(crs)}(T.(geotransform), Int.(size), crs)
end

MapGrid(; geotransform, size, crs = nothing) = MapGrid(geotransform, size, crs)

"""
    gridspacing(g::MapGrid)

Signed grid spacing `(dx, dy)`. Geogrid reports `dx` as `gridSpacingX` in its scalar output,
where downstream code uses it to relate chip size to grid spacing.
"""
gridspacing(g::MapGrid) = (g.geotransform[2], g.geotransform[6])

"""
    gridorigin(g::MapGrid)

Outer corner `(x₀, y₀)` of the grid's first pixel. Point *centers* are half a pixel inside this.
"""
gridorigin(g::MapGrid) = (g.geotransform[1], g.geotransform[4])

"""
    DEFAULT_ZRANGE

Elevation range, in meters, bounding an image footprint where the true terrain height is unknown:
`(-200.0, 4000.0)`, the reference's default (`GeogridOptical.py:115`).

The footprint is computed at both extremes and the union taken, so the bounding box is
conservative with respect to terrain.

Elevation changes a horizontal coordinate only when the transform crosses datums, since only then
does the pipeline apply a 3D Helmert shift. Between two CRSs on one datum the horizontal result is
identical at any height, so for imagery and grids on WGS84 — UTM, EPSG:3413, EPSG:3031, hence
every ITS_LIVE projection — this range widens nothing.
"""
const DEFAULT_ZRANGE = (-200.0, 4000.0)

"""
    footprint_bounds(transform, c::ProjectedCoordinate; zrange = DEFAULT_ZRANGE) -> Extent

Bounding box, in grid coordinates, of the image described by `c`.

`transform` maps image coordinates to grid coordinates and is called as
`transform(x, y, z) -> (x′, y′, z′)`. The image's four corners are transformed at both elevations
in `zrange` — eight points — and the box is their min/max, reproducing
`GeogridOptical.determineBbox` (`GeogridOptical.py:115-155`).

Corners are *pixel centers* of the first and last pixel (`origin + (size - 1) * spacing`), not
the outer boundary, matching the reference.

Returns an `Extents.Extent{(:X, :Y)}`. The elevation range is not part of the result: it only
widens the horizontal box.

Given a [`TransformPair`](@ref) this takes the correct direction itself. Note that it is the
pair's *inverse* — the kernel's forward direction is grid to image, and this needs image to grid —
so passing `tf.forward` by hand computes a bounding box in the wrong space, which surfaces as a
window far outside the grid.
"""
function footprint_bounds(transform, c::ProjectedCoordinate; zrange = DEFAULT_ZRANGE)
    xs = (c.origin[1], c.origin[1] + (c.size[1] - 1) * c.spacing[1])
    ys = (c.origin[2], c.origin[2] + (c.size[2] - 1) * c.spacing[2])

    xmin = ymin = Inf
    xmax = ymax = -Inf
    for x in xs, y in ys, z in zrange
        gx, gy, _ = transform(Float64(x), Float64(y), Float64(z))
        xmin = min(xmin, gx); xmax = max(xmax, gx)
        ymin = min(ymin, gy); ymax = max(ymax, gy)
    end

    isfinite(xmin) && isfinite(ymin) || throw(ArgumentError(
        "footprint_bounds got a non-finite result from the coordinate transform: " *
        "X = ($xmin, $xmax), Y = ($ymin, $ymax). The image most likely falls outside the " *
        "transform's area of validity."))

    return Extent(X = (xmin, xmax), Y = (ymin, ymax))
end

# Takes the image-to-grid direction from the pair, so the caller cannot pick the wrong one.
footprint_bounds(tf::TransformPair, c::ProjectedCoordinate; zrange = DEFAULT_ZRANGE) =
    footprint_bounds(tf.inverse, c; zrange)

"""
    grid_window(g::MapGrid, bounds::Extent) -> CartesianIndices{2}

The window of grid points overlapping `bounds`, as one-based `CartesianIndices`.

Reproduces `geogridOptical.cpp:239-244`:

```c
lOff   = std::max(std::floor((ymax - geoTrans[3]) / geoTrans[5]), 0.);
lCount = std::min(std::ceil((ymin - geoTrans[3]) / geoTrans[5]), demYSize - 1.) - lOff;
```

including the `demXSize - 1.` / `demYSize - 1.` clamp, which excludes the grid's last column and
row from ever being processed. That is reproduced rather than corrected — the reference's outputs
are one pixel short of the DEM extent and matching it is the point.

The reference computes these as `double` and truncates to `int`, and can produce a negative count
for a pair that does not overlap the grid, which it then hands to GDAL as a raster size. This
throws instead; see `REFERENCE.md`.

`CartesianIndices` rather than an offset/count pair so the result composes with array views,
`DiskArrays` chunk iteration, and blocking without conversion. The reference's zero-based
`pOff`/`lOff` are recoverable as `first(window).I .- 1`.
"""
function grid_window(g::MapGrid, bounds::Extent)
    gt = g.geotransform
    x0, dx, y0, dy = gt[1], gt[2], gt[4], gt[6]
    nx, ny = g.size

    xmin, xmax = bounds.X
    ymin, ymax = bounds.Y

    # `floor`/`ceil` on Float64 then truncate, in the reference's order. Which bound maps to the
    # offset depends on the sign of the spacing: for the usual north-up grid `dy < 0`, so
    # dividing by it flips the sense and `ymax` gives the first row.
    lOff = trunc(Int, max(floor((ymax - y0) / dy), 0.0))
    lCount = trunc(Int, min(ceil((ymin - y0) / dy), ny - 1.0)) - lOff
    pOff = trunc(Int, max(floor((xmin - x0) / dx), 0.0))
    pCount = trunc(Int, min(ceil((xmax - x0) / dx), nx - 1.0)) - pOff

    (pCount > 0 && lCount > 0) || throw(ArgumentError(
        "grid window is empty: $(pCount)x$(lCount) points at offset ($pOff, $lOff) in a " *
        "$(nx)x$(ny) grid. The image pair does not overlap the grid."))

    return CartesianIndices((pOff + 1:pOff + pCount, lOff + 1:lOff + lCount))
end

"""
    window_geotransform(g::MapGrid, window::CartesianIndices{2}) -> NTuple{6}

Geotransform of the sub-grid `window` selects, for writing outputs that carry their own
georeferencing.

Matches the reference's output geotransform: the origin shifts by the window offset and the
spacing is unchanged (`geogridOptical.cpp:246-252`).
"""
function window_geotransform(g::MapGrid, window::CartesianIndices{2})
    gt = g.geotransform
    pOff, lOff = first(window).I .- 1
    return (gt[1] + pOff * gt[2], gt[2], gt[3], gt[4] + lOff * gt[6], gt[5], gt[6])
end

"""
    gridpoint_center(g::MapGrid, i::Integer, j::Integer) -> NTuple{2}

Projected coordinate of the center of grid point `(i, j)`, one-based.

The half-pixel offset matches the reference (`geogridOptical.cpp:678-679`): geometry is evaluated
at pixel centers, not corners.
"""
@inline function gridpoint_center(g::MapGrid, i::Integer, j::Integer)
    gt = g.geotransform
    return (gt[1] + (i - 1 + 0.5) * gt[2], gt[4] + (j - 1 + 0.5) * gt[6])
end
