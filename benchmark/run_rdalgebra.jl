# `_range_doppler`'s inner loop does xyz_to_lonlat then lonlat_to_xyz, replacing only the height,
# and uses just `zsch = norm3(targ_xyz) - radius` from the result. `llhi` and `targ_xyz` are read
# only after the final iteration.
#
# But norm3 of the ECEF point at geodetic (lat, lon, h) is independent of lon:
#   N = a / sqrt(1 - e2 sin^2 lat)
#   |xyz|^2 = (N+h)^2 cos^2(lat) + (N(1-e2)+h)^2 sin^2(lat)
# So the intermediate iterations need only the LATITUDE -- not the longitude, and not the forward
# conversion. That removes an atan2 and a full forward transform from every iteration but the last.
include(joinpath(@__DIR__, "radar_setup.jl"))
using Printf, BenchmarkTools
using ImagePairGeometry: Ellipsoid, xyz_to_lonlat, lonlat_to_xyz, norm3, dot3, unitvec3,
                         geodetic_tcn, nadir_sphere, looksign, RANGE_DOPPLER_ITERATIONS,
                         _range_doppler, semiminor, WGS84_A, WGS84_E2
const C = bench_case(128); coord = C.coord; tf = C.tf; el = Ellipsoid()
lo, la, hh = tf.forward(-242500.0, 2179000.0, 500.0)
xyz = lonlat_to_xyz(el, SVector{3,Float64}(lo*DEG2RAD, la*DEG2RAD, hh))
pm, vm = interpolate(coord.orbit, orbit_midtime(coord))
p = geo2rdr(coord.orbit, xyz, midtime(coord), orbit_midtime(coord), pm, vm)
satx, satv = interpolate(coord.orbit, p.orbittime + 1/coord.prf)

"""Geocentric radius of the point at geodetic latitude `lat` (as sin) and height `h`."""
@inline function geocentric_radius(el::Ellipsoid, slat::Float64, h::Float64)
    s2 = slat*slat
    N = el.a / sqrt(1.0 - el.e2*s2)
    re = (N + h); rp = (N*(1.0 - el.e2) + h)
    return sqrt(re*re*(1.0 - s2) + rp*rp*s2)
end

# Only the latitude is needed mid-loop, so take the cheap part of the inverse.
@inline function _sinlat(el::Ellipsoid, v::SVector{3,Float64})
    ll = xyz_to_lonlat(el, v)
    return sin(ll[2])
end

@inline function rd_algebraic(el, c, satx, satv, rngpix, height)
    vhat = unitvec3(satv); radius, _, _, a = nadir_sphere(el, satx); tcn = geodetic_tcn(satx, satv)
    ndotv = dot3(tcn.nhat, vhat); vdott = dot3(vhat, tcn.that)
    zsch = height; llhi = SVector{3,Float64}(0.,0.,0.); txyz = llhi
    for it in 1:RANGE_DOPPLER_ITERATIONS
        b = radius + zsch
        ct = 0.5*(a/rngpix + rngpix/a - (b/a)*(b/rngpix)); st = sqrt(1 - ct*ct)
        gam = rngpix*ct; alp = -gam*ndotv/vdott
        bet = -looksign(c.look_side)*sqrt(rngpix*rngpix*st*st - alp*alp)
        tv = satx + alp*tcn.that + bet*tcn.chat + gam*tcn.nhat
        if it == RANGE_DOPPLER_ITERATIONS
            # Final iteration: the caller needs the full geodetic and ECEF answer.
            ll = xyz_to_lonlat(el, tv); llhi = SVector{3,Float64}(ll[1], ll[2], height)
            txyz = lonlat_to_xyz(el, llhi); zsch = norm3(txyz) - radius
        else
            # Intermediate: only `zsch` is read, and it needs the latitude alone.
            zsch = geocentric_radius(el, _sinlat(el, tv), height) - radius
        end
    end
    (txyz, llhi)
end

println("== is the algebraic identity exact? ==")
for h in (0.0, 500.0, 4000.0)
    v = lonlat_to_xyz(el, SVector{3,Float64}(0.14, 0.34, h))
    ll = xyz_to_lonlat(el, v)
    direct = norm3(lonlat_to_xyz(el, SVector{3,Float64}(ll[1], ll[2], h)))
    viaform = geocentric_radius(el, sin(ll[2]), h)
    @printf("h=%7.1f: round trip %.9f  closed form %.9f  diff %.3e m\n", h, direct, viaform,
            abs(direct - viaform))
end

println("\n== accuracy and speed of the reduced loop ==")
a_ref = _range_doppler(el, coord, satx, satv, p.range, 500.0)
a_new = rd_algebraic(el, coord, satx, satv, p.range, 500.0)
@printf("ECEF difference vs current: %.3e m (%.2e range samples)\n",
        norm3(a_new[1]-a_ref[1]), norm3(a_new[1]-a_ref[1])/coord.dr)
t_ref = minimum(@benchmark _range_doppler($el,$coord,$satx,$satv,$(p.range),500.0)).time
t_new = minimum(@benchmark rd_algebraic($el,$coord,$satx,$satv,$(p.range),500.0)).time
@printf("current %8.4f us   reduced %8.4f us   %.3fx\n", t_ref/1e3, t_new/1e3, t_ref/t_new)
@printf("\nprojected on the whole point (%.1f%% of it): %.3fx\n",
        100*t_ref/2504.6, 1/(1 - (t_ref/2504.6)*(1 - t_new/t_ref)))
