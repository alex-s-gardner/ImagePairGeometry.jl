# Radar coordinates to ground point: the inverse of `geo2rdr`.
#
# Where `geo2rdr` solves for time given a target, this solves for the target given a time and a
# slant range. The unknown is the target's height above the local sphere: a height determines a
# look angle, which places the target on the range sphere, and the DEM at that location gives a new
# height. Iterating to a fixed point lands on the terrain.
#
# Ported from isce3's `rdr2geo` (`Rdr2Geo.icc:44-168`), not from `geogridRadar.cpp`. The reference
# calls this only through `isce3.geometry.rdr2geo` at the Python level — for the scene footprint and
# the incidence angle (`GeogridRadar.py:140-297`) — so isce3 is the thing to match. The visually
# similar loop inside `geogridRadar.cpp:1038-1071` is a *different* computation with its own
# iteration count and no convergence test; it is transcribed separately.

"""
    RDR2GEO_THRESHOLD
    RDR2GEO_MAXITER
    RDR2GEO_EXTRAITER

isce3's `Rdr2GeoParams` defaults: a range residual threshold of `1e-8` m, 25 primary iterations and
15 extra ones (`Rdr2Geo.h:14-22`).

The reference's callers do not override them (`GeogridRadar.py:169-179` passes positionally up to
the DEM and ellipsoid, leaving the tail defaulted), so these are the values in production.
"""
const RDR2GEO_THRESHOLD = 1e-8
const RDR2GEO_MAXITER = 25
const RDR2GEO_EXTRAITER = 15

"""
    LookSide

Which side of the ground track the radar looks at: [`LookLeft`](@ref) or [`LookRight`](@ref).

The reference carries this as an integer `lookSide` multiplying a square root
(`geogridRadar.cpp:1048`), where `-1` is right-looking. [`looksign`](@ref) converts.
"""
@enum LookSide LookLeft LookRight

"""
    LookLeft

Left-looking radar geometry, the `+1` branch of the reference's `lookSide`. See
[`LookSide`](@ref).
"""
LookLeft

"""
    LookRight

Right-looking radar geometry, the `-1` branch of the reference's `lookSide` — Sentinel-1's
configuration. See [`LookSide`](@ref).
"""
LookRight

"""
    looksign(side::LookSide) -> Float64

The reference's `lookSide` integer as a float: `-1.0` for [`LookRight`](@ref), `+1.0` for
[`LookLeft`](@ref).

The sign convention is inverted relative to how it reads: `geogridRadar.cpp:1048` computes
`beta = -lookSide * sqrt(...)`, so a right-looking `lookSide` of `-1` yields a *positive* `beta`
along `chat`. isce3 spells the same choice out as
`(side == LookSide::Right) ? beta : -beta` (`Rdr2Geo.icc:97-101`), which is why both forms appear
below and agree.
"""
@inline looksign(side::LookSide) = side === LookRight ? -1.0 : 1.0

"""
    rdr2geo(orbit, el, aztime, range; height, doppler = 0.0, wavelength,
            side, threshold = RDR2GEO_THRESHOLD, maxiter = RDR2GEO_MAXITER,
            extraiter = RDR2GEO_EXTRAITER) -> SVector{3,Float64}

Ground point `(lon, lat, height)` in radians and meters, imaged at azimuth time `aztime` and slant
`range`.

`height` is the terrain height above the ellipsoid, as a number or an [`AbstractHeightSource`](@ref). A
number is constant everywhere, which is all the reference's callers supply — via `DEMInterpolator(zz)` at
a scalar `zz` (`GeogridRadar.py:176`, `:289`) — and is what the footprint and incidence-angle paths pass;
that case is bitwise what it was before sources existed.

A varying source is sampled at each candidate location, which is what inverting a *reference pixel* to a
ground point over real terrain requires. The fixed-point structure is unchanged, but what convergence
means is not: see [`AbstractHeightSource`](@ref), and read [`rdr2geo_converged`](@ref)'s flag rather than
assuming it.

Ported from `Rdr2Geo.icc:44-168`. Returns the converged location; a failure to converge within
`maxiter + extraiter` is *not* an error, matching isce3 — its NaN-on-failure branch is commented out
in the source with a note that enabling it breaks tests (`Rdr2Geo.icc:153-162`), and the reference
depends on that leniency. Use [`rdr2geo_converged`](@ref) when the distinction matters.
"""
function rdr2geo(orbit::Orbit, el::Ellipsoid, aztime::Real, range::Real;
                 height, doppler::Real = 0.0, wavelength::Real,
                 side::LookSide, threshold::Real = RDR2GEO_THRESHOLD,
                 maxiter::Integer = RDR2GEO_MAXITER,
                 extraiter::Integer = RDR2GEO_EXTRAITER)
    llh, _ = rdr2geo_converged(orbit, el, aztime, range; height, doppler, wavelength, side,
                               threshold, maxiter, extraiter)
    return llh
end

# Whether `update_llh` has a real solution for a target at geocentric distance `b`, imaged at slant range
# `r` from a satellite at geocentric distance `a` — that is, whether both of its square roots have
# non-negative arguments.
#
# Two ways the geometry can fail to close, and a varying height source reaches both. The law of cosines
# gives `cos_theta` for the triangle centre-satellite-target, and `|cos_theta| > 1` means no such triangle
# exists: the range sphere and the height sphere do not intersect. Past that, `beta` takes the root of
# `(r sin_theta)^2 - alpha^2`, which goes negative when the zero-Doppler plane misses the intersection
# circle — the target is on the sphere but not at a point this geometry images.
#
# Tested before `update_llh` rather than inside it, so the arithmetic there stays exactly the reference's
# and the constant-height path is untouched. Duplicating three lines is the price of that, and it is worth
# paying: those three lines are asserted bitwise against isce3 in `test/radar_numerics.jl`.
@inline function _reachable(a::Float64, r::Float64, b::Float64, ndotv::Float64, vdott::Float64,
                            dopfact::Float64)
    cos_theta = 0.5 * (a / r + r / a - (b / a) * (b / r))
    abs(cos_theta) <= 1.0 || return false
    sin_theta = sqrt(1.0 - cos_theta * cos_theta)
    gamma = r * cos_theta
    alpha = (dopfact - gamma * ndotv) / vdott
    x = r * sin_theta
    return (x * x - alpha * alpha) >= 0.0
end

"""
    rdr2geo_converged(orbit, el, aztime, range; kw...) -> Tuple{SVector{3,Float64},Bool}

[`rdr2geo`](@ref) with the convergence flag alongside the location.

`false` means the range residual never fell below `threshold`. The location is still isce3's answer
in that case — see [`rdr2geo`](@ref) — so this is for a caller that wants to know, not a caller that
wants a different result.
"""
function rdr2geo_converged(orbit::Orbit, el::Ellipsoid, aztime::Real, range::Real;
                           height, doppler::Real = 0.0, wavelength::Real,
                           side::LookSide, threshold::Real = RDR2GEO_THRESHOLD,
                           maxiter::Integer = RDR2GEO_MAXITER,
                           extraiter::Integer = RDR2GEO_EXTRAITER)
    r = Float64(range)
    pos, vel = interpolate(orbit, aztime)
    tcn = geodetic_tcn(pos, vel)

    vmag = norm3(vel)
    vhat = unitvec3(vel)
    # The Doppler centroid as a distance along track. Zero for the zero-Doppler geometry the
    # reference uses, which makes `alpha` reduce to the pure geometric term.
    dopfact = 0.5 * Float64(wavelength) * Float64(doppler) * r / vmag

    # `a` is the satellite's geocentric distance, constant across the iteration, so it comes out of
    # `nadir_sphere` — which computes it anyway — rather than being recomputed per candidate height.
    radius, sat_height, _, a = nadir_sphere(el, pos)

    ndotv = dot3(tcn.nhat, vhat)
    vdott = dot3(vhat, tcn.that)

    # A candidate target height above the local sphere gives a look angle, hence a position on the
    # range sphere, hence a location.
    update_llh = function (h::Float64)
        b = radius + h
        cos_theta = 0.5 * (a / r + r / a - (b / a) * (b / r))
        sin_theta = sqrt(1.0 - cos_theta * cos_theta)

        gamma = r * cos_theta
        # isce3's grouping: the whole numerator over `vdott` (`Rdr2Geo.icc:94`). geogridRadar
        # divides only the second term (`:1046`); the two agree exactly when `dopfact` is zero,
        # which is the only case the reference exercises. Both forms are kept, each where it
        # belongs.
        alpha = (dopfact - gamma * ndotv) / vdott
        x = r * sin_theta
        beta = looksign(side) * -sqrt((x * x) - (alpha * alpha))

        delta = alpha * tcn.that + beta * tcn.chat + gamma * tcn.nhat
        return xyz_to_lonlat(el, pos + delta)
    end

    converged = false
    h = reference_height(height)
    llh_old = SVector{3,Float64}(0.0, 0.0, 0.0)
    llh_new = llh_old

    for i in 1:(maxiter + extraiter)
        # Near-nadir: no look angle on the range sphere reaches a target this far below the
        # satellite, so the geometry has no solution and iterating further cannot find one.
        sat_height - h >= r && break
        # The complementary failure, reachable only from a *varying* height source: a candidate so far
        # above the local sphere that no look angle on the range sphere reaches it either, which puts
        # `|cos_theta| > 1` and takes `sqrt` of a negative in `update_llh`. A constant source cannot get
        # here — `h` barely moves — so this guard costs the reference's path nothing and is not a
        # divergence from it: isce3 would fault on the same input, and faulting is not a behavior with a
        # value to reproduce. Leaves `converged` false and the last good estimate standing, which is what
        # a caller reading the flag expects.
        _reachable(a, r, radius + h, ndotv, vdott, dopfact) || break

        llh_new = update_llh(h)
        # Snap to the terrain, sampled at this candidate's own location. A constant source ignores the
        # location and returns its value — which is what makes that case converge in a handful of steps,
        # and what makes it bitwise identical to the scalar `height` this took before sources existed.
        llh_new = SVector{3,Float64}(llh_new[1], llh_new[2],
                                     height_at(height, llh_new[1], llh_new[2]))

        xyz_new = lonlat_to_xyz(el, llh_new)
        h = norm3(xyz_new) - radius

        if abs(r - norm3(pos - xyz_new)) < threshold
            converged = true
            break
        end

        # Past the primary iterations, damp by averaging with the previous estimate. isce3 does this
        # to break the oscillation that keeps a non-converging case from settling
        # (`Rdr2Geo.icc:140-147`).
        if i > maxiter
            xyz_avg = 0.5 * (lonlat_to_xyz(el, llh_old) + xyz_new)
            llh_new = xyz_to_lonlat(el, xyz_avg)
            h = norm3(xyz_avg) - radius
        end

        llh_old = llh_new
    end

    # One final evaluation, so the result sits exactly on the range sphere rather than at the last
    # DEM-snapped estimate (`Rdr2Geo.icc:164`).
    #
    # Guarded for the same reason the loop is: the height the loop exited with need not be one this
    # geometry can image, and a varying source can leave it that way. Falling back to the last snapped
    # estimate keeps the return finite and `converged` false, which is the contract a caller reads.
    _reachable(a, r, radius + h, ndotv, vdott, dopfact) || return (llh_new, false)
    return (update_llh(h), converged)
end
