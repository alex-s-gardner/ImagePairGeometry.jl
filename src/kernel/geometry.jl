# Per-point geometry: everything the pair's time separation does not affect.
#
# Split from the output computation because that boundary is useful three ways at once. It is
# where the coordinate transforms happen, so the identity fast path can replace this function and
# nothing else. It is where the expensive work is, so it is what a cache across pairs over one
# footprint would store. And every point is independent here, so it is the unit of parallel work.
#
# The three transform calls, in the reference's order (`geogridOptical.cpp:717-752`): forward for
# the point itself, then inverse for that result stepped one pixel along each image axis. The
# difference vectors are not normalized in place — their lengths are the physical pixel sizes in
# grid coordinates, needed for the offset conversion and the scale factors.

"""
    DEG2RAD

Radians per degree, as the reference computes it: `M_PI / 180.0` (`geogridOptical.cpp:39`).
"""
const DEG2RAD = pi / 180.0

"""
    PointGeometry

Geometry at one grid point that depends only on the grid, the image coordinate system and the
projection — not on the image pair's time separation.

# Fields
- `image_xy`: where the point falls in the image, in whatever units that path's forward mapping
  produces. On the projected path these are image *coordinates*, and [`pixel_index`](@ref) converts
  them to a rounded index. On the radar path they are already the rounded `(range, azimuth)` index,
  because range and azimuth are computed from the solve rather than from an affine relation — there is
  no intermediate coordinate to hold. So `pixel_index` applies to the projected path only; a radar
  caller reads this field directly.
- `xunit`, `yunit`: unit vectors along the image's two axes, expressed in grid coordinates.
- `xlen`, `ylen`: length in grid coordinates of a one-pixel step along each image axis. The
  physical pixel size at this point, which differs from the nominal spacing wherever the
  projection distorts.
- `normal`: upward surface normal from the slope rasters, or the zero vector where no slope is
  supplied.
"""
struct PointGeometry
    image_xy::NTuple{2,Float64}
    xunit::SVector{3,Float64}
    yunit::SVector{3,Float64}
    xlen::Float64
    ylen::Float64
    normal::SVector{3,Float64}
end

"""
    surface_normal(dhdx, dhdy) -> SVector{3,Float64}

Upward unit normal to a surface with the given slopes.

Matches `geogridOptical.cpp:756-764`: normalize `(dhdx, dhdy, -1)` then negate, giving a vector
with positive `z`. The velocity closure divides by that `z` component, so the sign convention is
load-bearing.
"""
@inline surface_normal(dhdx::Real, dhdy::Real) =
    -unitvec3(SVector{3,Float64}(Float64(dhdx), Float64(dhdy), -1.0))

"""
    NO_NORMAL

The zero vector, standing for "no slope raster supplied".

The reference sets the normal to all zeros in that case (`geogridOptical.cpp:761-764`) rather than
skipping the computations that use it, so the velocity closure divides by zero and the outputs
that depend on it are `Inf` or `NaN`. Those outputs are not written at all without a slope raster,
so the values never reach a file — but the arithmetic runs, and reproducing it keeps the code paths
aligned.
"""
const NO_NORMAL = SVector{3,Float64}(0.0, 0.0, 0.0)

"""
    pointgeometry(tf::TransformPair, gx, gy, gz, c::ProjectedCoordinate, normal) -> PointGeometry

Geometry at the grid point `(gx, gy, gz)` for an image with coordinate system `c`.

`gx`, `gy` are the point's center in grid coordinates and `gz` its elevation. `normal` comes from
[`surface_normal`](@ref), or is [`NO_NORMAL`](@ref).

Reproduces `geogridOptical.cpp:717-752`. The axis unit vectors are obtained by stepping one pixel
along each image axis, transforming back to grid coordinates, and normalizing the difference — so
they carry the projection's local rotation and scale, which is what makes the velocity conversion
valid off the projection's center.

Under [`IdentityTransform`](@ref) a specialized method computes the same values in closed form.
"""
@inline function pointgeometry(tf::TransformPair, gx::Real, gy::Real, gz::Real,
                               c::ProjectedCoordinate, normal::SVector{3,Float64})
    grid = SVector{3,Float64}(Float64(gx), Float64(gy), Float64(gz))

    ix, iy, iz = tf.forward(grid[1], grid[2], grid[3])

    dx = Float64(c.spacing[1])
    dy = Float64(c.spacing[2])

    # One pixel along the image x axis, back to grid coordinates.
    sx1, sx2, sx3 = tf.inverse(ix + dx, iy, iz)
    xdiff = SVector{3,Float64}(sx1 - grid[1], sx2 - grid[2], sx3 - grid[3])

    # One pixel along the image y axis.
    sy1, sy2, sy3 = tf.inverse(ix, iy + dy, iz)
    ydiff = SVector{3,Float64}(sy1 - grid[1], sy2 - grid[2], sy3 - grid[3])

    return PointGeometry((ix, iy), unitvec3(xdiff), unitvec3(ydiff),
                         norm3(xdiff), norm3(ydiff), normal)
end

# Identity transform: the image coordinate is the grid coordinate, so the one-pixel step lands
# exactly `spacing` away along one axis with nothing in the others. Every quantity below is exact
# — `norm3((s, 0, 0))` is `abs(s)`, and dividing by it gives `sign(s)` — so this is a shortcut in
# work only, not in precision.
@inline function pointgeometry(::TransformPair{IdentityTransform,IdentityTransform},
                               gx::Real, gy::Real, gz::Real,
                               c::ProjectedCoordinate, normal::SVector{3,Float64})
    dx = Float64(c.spacing[1])
    dy = Float64(c.spacing[2])
    return PointGeometry((Float64(gx), Float64(gy)),
                         SVector{3,Float64}(sign(dx), 0.0, 0.0),
                         SVector{3,Float64}(0.0, sign(dy), 0.0),
                         abs(dx), abs(dy), normal)
end

"""
    pixel_index(g::PointGeometry, c::ProjectedCoordinate) -> NTuple{2,Float64}

Zero-based pixel index of the point in the image, rounded to the nearest whole pixel.

Projected path only, by signature: it inverts the affine relation between image coordinates and pixel
indices, which a radar acquisition has no analogue of. There, the index comes out of the solve — see
[`PointGeometry`](@ref) on what `image_xy` holds on each path.

Matches `geogridOptical.cpp:723-724`. Uses [`cround`](@ref) — halves away from zero — because the
reference rounds here in C++, not in NumPy. The result is `Float64` so the bounds test that
follows sees exactly the value the reference tests, before any integer conversion.
"""
@inline function pixel_index(g::PointGeometry, c::ProjectedCoordinate)
    return (cround((g.image_xy[1] - Float64(c.origin[1])) / Float64(c.spacing[1])),
            cround((g.image_xy[2] - Float64(c.origin[2])) / Float64(c.spacing[2])))
end

"""
    inbounds(xind, yind, c::AbstractImageCoordinate) -> Bool

Whether a zero-based pixel index falls inside the image.

`geogridOptical.cpp:775` and `geogridRadar.cpp:1112` write the same expression against their own
size fields, so one method serves both paths through [`nsamples`](@ref) and [`nlines`](@ref).

The comparison is on the pre-conversion `Float64`. A `NaN` index — reachable when the transform is fed
a `NaN` elevation — compares false in every direction and so is reported *in* bounds, as the reference
reports it. See `REFERENCE.md`.
"""
@inline function inbounds(xind::Float64, yind::Float64, c::AbstractImageCoordinate)
    return !((xind > nsamples(c) - 1) | (xind < 0) | (yind > nlines(c) - 1) | (yind < 0))
end

"""
    cross_check(g::PointGeometry) -> Float64

Degrees by which the surface normal deviates from perpendicular to the plane of the image axes.

Matches `geogridOptical.cpp:812-814`: the angle between the normal and the normalized cross
product of the axis unit vectors, less 90°, in absolute value. The reference computes the
displacement-to-velocity operator only where this exceeds 1°, since below that the two-by-two
system is near singular and its inverse is dominated by rounding.

For realistic terrain this sits at 82–90°, far above the threshold; it approaches it only for
near-vertical surfaces.
"""
@inline function cross_check(g::PointGeometry)
    c = unitvec3(cross3(g.xunit, g.yunit))
    return abs(acos(dot3(g.normal, c)) / DEG2RAD - 90.0)
end
