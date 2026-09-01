# Per-point outputs: the reference's expressions, in its order.
#
# Everything here consumes the pair's time separation, and the arithmetic is transcribed rather
# than simplified. `dt / spacing / 365.0 / 24.0 / 3600.0` is four sequential divisions, not a
# division by a precomputed `dt / (spacing * 31_536_000)`; floating-point division is not
# associative, so the algebraically equal form gives a different last bit.
#
# The reference writes the two-by-two operator as four expressions that each repeat the same
# determinant inline (`geogridOptical.cpp:817-832`). Hoisting it is safe — an identical
# subexpression evaluated once instead of four times is bit-identical, since each evaluation was
# already deterministic — and it is the one simplification taken here.

"""
    SECONDS_PER_YEAR_TERMS

The reference's year length as the three factors it divides by in sequence: `(365.0, 24.0,
3600.0)`.

Their product, 31_536_000, is exact in `Float64`, but `x / 365.0 / 24.0 / 3600.0` and
`x / 31_536_000.0` still differ in the last bit for most `x` because each division rounds. Kept as
separate factors so the division order matches.
"""
const SECONDS_PER_YEAR_TERMS = (365.0, 24.0, 3600.0)

"""
    per_year(x) -> Float64

`x` divided by one year in seconds, in the reference's division order:
`x / 365.0 / 24.0 / 3600.0`.
"""
@inline per_year(x::Float64) =
    x / SECONDS_PER_YEAR_TERMS[1] / SECONDS_PER_YEAR_TERMS[2] / SECONDS_PER_YEAR_TERMS[3]

"""
    close_slope_parallel(vx, vy, normal) -> SVector{3,Float64}

A horizontal velocity completed to three components by requiring flow parallel to the surface.

The vertical component follows from `v · n = 0`:

    vz = -(vx * nx + vy * ny) / nz

matching `geogridOptical.cpp:766-773`. Applied to both the reference velocity and the search
range, since a search extent is a velocity bound and must lie in the same plane.

No guard on `nz`: with a slope raster the normal's `z` is positive by construction, and without
one the normal is the zero vector and the result is non-finite — which is the reference's behavior,
and those outputs are not written in that configuration.
"""
@inline function close_slope_parallel(vx::Float64, vy::Float64, normal::SVector{3,Float64})
    vz = -(vx * normal[1] + vy * normal[2]) / normal[3]
    return SVector{3,Float64}(vx, vy, vz)
end

"""
    pixel_offset(vel, g::PointGeometry, dt) -> NTuple{2,Float64}

Displacement in pixels, along each image axis, that `vel` produces over `dt`.

Matches `geogridOptical.cpp:807-808`: the velocity projected onto the axis unit vector, times the
interval, divided by the physical length of one pixel along that axis. Rounded with
[`cround`](@ref) to a whole pixel.

`vel` is in meters per year and three-component — use [`close_slope_parallel`](@ref) to build it —
and the axis lengths come from `g`, so projection distortion is accounted for.
"""
@inline function pixel_offset(vel::SVector{3,Float64}, g::PointGeometry, dt::Float64)
    return (cround(per_year(dot3(vel, g.xunit) * dt / g.xlen)),
            cround(per_year(dot3(vel, g.yunit) * dt / g.ylen)))
end

"""
    offset_to_velocity(g::PointGeometry, c::ProjectedCoordinate, dt)
        -> NTuple{4,Float64}

The two-by-two operator converting a pixel displacement to a map velocity, as
`(vx_from_dx, vx_from_dy, vy_from_dx, vy_from_dy)`.

Given a displacement `(dx, dy)` in pixels, the map velocity is

    vx = vx_from_dx * dx + vx_from_dy * dy
    vy = vy_from_dx * dx + vy_from_dy * dy

which is how the downstream correlator reconstructs velocity, after scaling the displacement by
[`scale_factors`](@ref).

Matches `geogridOptical.cpp:817-832`. The operator inverts the two-by-two system relating map
displacement to image displacement, projected into the plane perpendicular to the surface normal —
so it embeds both the projection's local geometry and the slope-parallel-flow constraint. The
shared determinant is evaluated once rather than the reference's four times, which is
bit-identical.

Valid only where [`cross_check`](@ref) exceeds one degree; the caller checks, as the reference
does, and writes nodata otherwise.
"""
@inline function offset_to_velocity(g::PointGeometry, c::ProjectedCoordinate, dt::Float64)
    n = g.normal
    xu = g.xunit
    yu = g.yunit

    # The reference spells this out inline in each of the four expressions.
    det = (n[3] * xu[1] - n[1] * xu[3]) * (n[3] * yu[2] - n[2] * yu[3]) -
          (n[3] * yu[1] - n[1] * yu[3]) * (n[3] * xu[2] - n[2] * xu[3])

    # Nominal spacing here, not the per-point axis length: the reference uses `XSize`/`YSize`,
    # and the difference between the two is what `scale_factors` carries separately.
    xden = per_year(dt / Float64(c.spacing[1]))
    yden = per_year(dt / Float64(c.spacing[2]))

    vx_dx = n[3] / xden * (n[3] * yu[2] - n[2] * yu[3]) / det
    vx_dy = -n[3] / yden * (n[3] * xu[2] - n[2] * xu[3]) / det
    vy_dx = -n[3] / xden * (n[3] * yu[1] - n[1] * yu[3]) / det
    vy_dy = n[3] / yden * (n[3] * xu[1] - n[1] * xu[3]) / det

    return (vx_dx, vx_dy, vy_dx, vy_dy)
end

"""
    scale_factors(g::PointGeometry, c::ProjectedCoordinate) -> NTuple{2,Float64}

Ratio of true ground distance to nominal pixel spacing, per axis.

Matches `geogridOptical.cpp:841-842`: the physical length of a one-pixel step in grid coordinates
divided by the nominal spacing. Exactly 1 where the grid and image share a CRS; away from that,
the map projection's local scale distortion — around 0.9996 at a UTM zone's central meridian.

The correlator multiplies its measured displacement by these before applying
[`offset_to_velocity`](@ref), which is why that operator uses nominal spacing.
"""
@inline scale_factors(g::PointGeometry, c::ProjectedCoordinate) =
    (g.xlen / abs(Float64(c.spacing[1])), g.ylen / abs(Float64(c.spacing[2])))

"""
    search_pixels(sr1, sr2, g::PointGeometry, c::ProjectedCoordinate, dt) -> NTuple{2,Float64}

Search half-extent in pixels along each image axis, from a search range already scaled and
slope-closed.

`sr1` and `sr2` are the two search vectors the reference forms: `sr2` is `sr1` with the x
component negated (`geogridOptical.cpp:713-714`). Both are projected onto each axis and the larger
magnitude taken, so the extent covers motion in either direction, then floored at one pixel — a
zero extent would mean no search at all.

Matches `geogridOptical.cpp:849-863`.
"""
@inline function search_pixels(sr1::SVector{3,Float64}, sr2::SVector{3,Float64},
                               g::PointGeometry, c::ProjectedCoordinate, dt::Float64)
    dx = Float64(c.spacing[1])
    dy = Float64(c.spacing[2])

    sx = abs(cround(per_year(dot3(sr1, g.xunit) * dt / dx)))
    sy = abs(cround(per_year(dot3(sr1, g.yunit) * dt / dy)))

    sx = max(sx, abs(cround(per_year(dot3(sr2, g.xunit) * dt / dx))))
    sy = max(sy, abs(cround(per_year(dot3(sr2, g.yunit) * dt / dy))))

    return (sx == 0 ? 1.0 : sx, sy == 0 ? 1.0 : sy)
end

"""
    chip_pixels(csx, csy, chip_size_0, pix_x, pix_y) -> NTuple{2,Float64}

A chip size in meters converted to pixels along each axis.

Matches `geogridOptical.cpp:874-875`: scaled by the ratio of the reference chip size to its
pixel equivalent. The caller converts with [`ctrunc32`](@ref) — the reference truncates here
rather than rounding, unlike every other integer output.

`pix_x`/`pix_y` come from [`chip_size_pixels`](@ref).
"""
@inline chip_pixels(csx::Float64, csy::Float64, chip_size_0::Float64,
                    pix_x::Int, pix_y::Int) =
    (csx / chip_size_0 * pix_x, csy / chip_size_0 * pix_y)

"""
    chip_size_pixels(chip_size_0, res) -> Int

The reference chip size in pixels for a pixel size of `res`, rounded up to a multiple of four.

Matches `geogridOptical.cpp:513-514`: `ceil(chip_size_0 / res / 4) * 4`. The multiple of four is
what the correlator's chip pyramid requires.

```jldoctest
julia> using ImagePairGeometry: chip_size_pixels

julia> chip_size_pixels(240.0, 30.0)
8

julia> chip_size_pixels(240.0, 10.0)
24

julia> chip_size_pixels(240.0, 15.0)
16
```
"""
@inline chip_size_pixels(chip_size_0::Real, res::Real) =
    Int(ceil(Float64(chip_size_0) / Float64(res) / 4) * 4)
