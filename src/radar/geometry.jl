# Per-point radar geometry: the forward mapping, and the two axis vectors the outputs need.
#
# Reaches the same `PointGeometry` the projected path produces, by a different route. Where that path
# transforms a grid point and steps one pixel along each image axis, this one solves for the azimuth
# time and slant range, then constructs each axis vector by stepping one pixel in *radar* coordinates
# and transforming the result back to grid coordinates:
#
#   range axis   — step `dr` along the ECEF look vector, convert to lon/lat, then to grid coordinates,
#                  and difference against the grid point (`geogridRadar.cpp:975-996`).
#   azimuth axis — advance one pulse interval, re-solve the range-Doppler equations at the new
#                  satellite position, and difference the resulting ground point the same way
#                  (`:1000-1085`).
#
# So both vectors end up in grid coordinates with the same meaning as the projected path's, which is
# what lets `outputs.jl` serve both. What does not carry over is a single nominal spacing: see
# `RadarSpacing`.

"""
    RANGE_DOPPLER_ITERATIONS

The reference's fixed iteration count for the range-Doppler solve: 10 (`geogridRadar.cpp:1038`).

As with [`GEO2RDR_ITERATIONS`](@ref) there is no convergence test, and here the residual `rdiff` is
computed (`:1070`) and never read. Contrast isce3's `rdr2geo`, which breaks on `dr < 1e-8` — this is a
separate transcription for that reason, not a call into [`rdr2geo`](@ref).
"""
const RANGE_DOPPLER_ITERATIONS = 10

"""
    RadarSpacing

The nominal spacings the radar outputs divide by, which are **not** one pair.

# Fields
- `operator`: `(dr, |da|)`, for [`offset_to_velocity`](@ref) and the azimuth [`scale_factors`](@ref)
  denominator. `da` is the along-track step in **ECEF** meters (`geogridRadar.cpp:1159`).
- `search`: `(dr, |alt|)`, for [`search_pixels`](@ref). `alt` is the same step in **grid**
  coordinates.

`alt` and `da` are one displacement expressed in two frames, so their norms differ — by the local map
scale, around 0.04% in UTM. The reference divides by `norm(da)` at `:1166-1176` and `:1191` and by
`norm(alt)` at `:1207-1216`, and that difference is reproduced rather than reconciled: it changes the
last bits of the operator relative to the search extent.

The projected path has no such split — one spacing serves every output there — which is why the
kernel takes spacings as values rather than reading them off a coordinate.
"""
struct RadarSpacing
    operator::NTuple{2,Float64}
    search::NTuple{2,Float64}
end

"""
    pointgeometry(tf::TransformPair, gx, gy, gz, c::RadarCoordinate, normal)
        -> Tuple{PointGeometry,RadarSpacing,RadarPoint}

Geometry at the grid point `(gx, gy, gz)` for the radar acquisition `c`.

`tf.forward` maps grid coordinates to geodetic degrees and `tf.inverse` back — the opposite naming
sense from the projected path's image coordinates, but the same direction of use: forward to leave the
grid, inverse to return to it.

Returns the [`PointGeometry`](@ref) the shared outputs consume, the [`RadarSpacing`](@ref) they divide
by, and the [`RadarPoint`](@ref) the range and azimuth indices come from.

Reproduces `geogridRadar.cpp:850-1085`. Three solves per point — one `geo2rdr`, then two
range-Doppler iterations for the along-track step — so this is roughly two orders of magnitude more
expensive than the projected path's three transform calls.
"""
function pointgeometry(tf::TransformPair, gx::Real, gy::Real, gz::Real,
                       c::RadarCoordinate, normal::SVector{3,Float64})
    el = Ellipsoid()
    grid = SVector{3,Float64}(Float64(gx), Float64(gy), Float64(gz))

    # Grid point to ECEF, via geodetic. The reference's `fwdTrans` produces degrees, and `llhi` holds
    # radians in (lon, lat, h) order — the swap at `:892-898` is between GDAL's (x, y) and isce3's
    # (lon, lat), which are the same order once both are read as (east, north).
    lon_d, lat_d, h = tf.forward(grid[1], grid[2], grid[3])
    xyz = lonlat_to_xyz(el, SVector{3,Float64}(lon_d * DEG2RAD, lat_d * DEG2RAD, h))

    # The zero-Doppler solve, from the scene-center initial guess the reference uses for every point.
    pm, vm = interpolate(c.orbit, orbit_midtime(c))
    p = geo2rdr(c.orbit, xyz, midtime(c), orbit_midtime(c), pm, vm)

    # Range axis: one range sample along the look direction, back to grid coordinates.
    los_ecef = unitvec3(p.look)
    range_step = xyz + los_ecef * c.dr
    xdiff = _to_grid(tf, el, range_step) - grid
    xlen = norm3(xdiff)

    # Azimuth axis: advance one pulse interval and re-solve. `tline` and `tlined` both step, as they
    # both received every Newton increment.
    satx, satv = interpolate(c.orbit, p.orbittime + 1 / c.prf)
    targ_xyz, targ_llh = _range_doppler(el, c, satx, satv, p.range, Float64(gz))

    ydiff = _to_grid(tf, targ_llh) - grid
    ylen = norm3(ydiff)

    # The along-track step in ECEF, which the operator and the azimuth scale factor divide by. Not
    # `ylen`: that is the same step in grid coordinates, and the two differ by the local map scale.
    da = norm3(targ_xyz - xyz)

    g = PointGeometry((Float64(range_index(p, c.starting_range, c.dr)),
                       Float64(azimuth_index(p, c.sensing_start, c.prf))),
                      unitvec3(xdiff), unitvec3(ydiff), xlen, ylen, normal)

    return (g, RadarSpacing((c.dr, da), (c.dr, ylen)), p)
end

# A geodetic position as a grid coordinate. Takes radians and converts, since every caller here holds
# radians and the transform wants degrees.
@inline function _to_grid(tf::TransformPair, llh::SVector{3,Float64})
    gx, gy, gz = tf.inverse(llh[1] / DEG2RAD, llh[2] / DEG2RAD, llh[3])
    return SVector{3,Float64}(gx, gy, gz)
end

# An ECEF position as a grid coordinate. Separate from the above because the range-Doppler solve
# already produces a geodetic position, and re-deriving it from ECEF would not round-trip to the same
# last bits — so that caller passes its `llh` straight to the method above.
@inline _to_grid(tf::TransformPair, el::Ellipsoid, xyz::SVector{3,Float64}) =
    _to_grid(tf, xyz_to_lonlat(el, xyz))

# The range-Doppler solve: where on the range sphere, at this satellite position, the ground sits at
# the given height.
#
# `geogridRadar.cpp:1038-1071`. Structurally isce3's `rdr2geo` against a constant-height DEM, and
# transcribed separately because it differs in three ways: a fixed 10 iterations with no convergence
# test, a residual computed and discarded, and `alpha` grouped differently — `:1046` divides only the
# second term by `dot(vhat, that)` where `Rdr2Geo.icc:94` divides the whole numerator. Those agree
# only because `dopfact` is hardcoded to zero at `:1009`, which is why both forms are kept, each at
# its own site.
function _range_doppler(el::Ellipsoid, c::RadarCoordinate, satx::SVector{3,Float64},
                        satv::SVector{3,Float64}, rngpix::Float64, height::Float64)
    vhat = unitvec3(satv)
    radius, _, _, sat_dist = nadir_sphere(el, satx)
    tcn = geodetic_tcn(satx, satv)

    dopfact = 0.0
    ndotv = dot3(tcn.nhat, vhat)
    vdott = dot3(vhat, tcn.that)

    zsch = height
    llhi = SVector{3,Float64}(0.0, 0.0, 0.0)
    targ_xyz = SVector{3,Float64}(0.0, 0.0, 0.0)

    # `a` is the satellite's geocentric distance, invariant across the loop; `a / rngpix`,
    # `rngpix / a` and the look sign are invariant with it. Hoisting them by hand is bit-identical and
    # measured no faster — LLVM already lifts them out — so the expression stays in the reference's
    # shape.
    a = sat_dist

    for _ in 1:RANGE_DOPPLER_ITERATIONS
        b = radius + zsch

        costheta = 0.5 * (a / rngpix + rngpix / a - (b / a) * (b / rngpix))
        sintheta = sqrt(1 - costheta * costheta)

        gamma = rngpix * costheta
        # geogridRadar's grouping, not isce3's. See the comment above.
        alpha = dopfact - gamma * ndotv / vdott
        beta = -looksign(c.look_side) * sqrt(rngpix * rngpix * sintheta * sintheta - alpha * alpha)

        delta = alpha * tcn.that + beta * tcn.chat + gamma * tcn.nhat
        targ_vec = satx + delta

        # Snap to the requested height, then back to ECEF. The reference overwrites `llhi[2]` in
        # place (`:1057`) and converts from the modified value.
        lonlat = xyz_to_lonlat(el, targ_vec)
        llhi = SVector{3,Float64}(lonlat[1], lonlat[2], height)
        targ_xyz = lonlat_to_xyz(el, llhi)

        zsch = norm3(targ_xyz) - radius
        # `rdiff` is computed here in the reference and never read, so it is not computed at all.
    end

    return (targ_xyz, llhi)
end
