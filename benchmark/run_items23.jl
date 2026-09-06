# Items 2 (hoist range-Doppler invariants) and 3 (hoist the scene-center guess), measured against
# the pre-change form in one process so the comparison is not across-run.
include(joinpath(@__DIR__, "radar_setup.jl"))
using Printf, BenchmarkTools
using ImagePairGeometry: Ellipsoid, xyz_to_lonlat, lonlat_to_xyz, norm3, dot3, unitvec3,
                         geodetic_tcn, nadir_sphere, looksign, RANGE_DOPPLER_ITERATIONS,
                         _range_doppler, SceneCenterStart, GeometryParams, _bind
const C = bench_case(128); coord = C.coord; tf = C.tf; el = Ellipsoid()
lo, la, hh = tf.forward(-242500.0, 2179000.0, 500.0)
xyz = lonlat_to_xyz(el, SVector{3,Float64}(lo*DEG2RAD, la*DEG2RAD, hh))
pm, vm = interpolate(coord.orbit, orbit_midtime(coord))
p = geo2rdr(coord.orbit, xyz, midtime(coord), orbit_midtime(coord), pm, vm)
satx, satv = interpolate(coord.orbit, p.orbittime + 1/coord.prf)

# The pre-item-2 form: invariants inside the loop.
@inline function rd_old(el, c, satx, satv, rngpix, height)
    vhat = unitvec3(satv); radius, _, _, a = nadir_sphere(el, satx); tcn = geodetic_tcn(satx, satv)
    ndotv = dot3(tcn.nhat, vhat); vdott = dot3(vhat, tcn.that)
    zsch = height; llhi = SVector{3,Float64}(0.,0.,0.); txyz = llhi
    for _ in 1:RANGE_DOPPLER_ITERATIONS
        b = radius + zsch
        ct = 0.5*(a/rngpix + rngpix/a - (b/a)*(b/rngpix)); st = sqrt(1 - ct*ct)
        gam = rngpix*ct; alp = -gam*ndotv/vdott
        bet = -looksign(c.look_side)*sqrt(rngpix*rngpix*st*st - alp*alp)
        tv = satx + alp*tcn.that + bet*tcn.chat + gam*tcn.nhat
        ll = xyz_to_lonlat(el, tv); llhi = SVector{3,Float64}(ll[1], ll[2], height)
        txyz = lonlat_to_xyz(el, llhi); zsch = norm3(txyz) - radius
    end
    (txyz, llhi)
end

println("== item 2: _range_doppler in isolation ==")
a_new = _range_doppler(el, coord, satx, satv, p.range, 500.0)
a_old = rd_old(el, coord, satx, satv, p.range, 500.0)
t_new = minimum(@benchmark _range_doppler($el,$coord,$satx,$satv,$(p.range),500.0)).time
t_old = minimum(@benchmark rd_old($el,$coord,$satx,$satv,$(p.range),500.0)).time
@printf("in-loop  %8.4f us\nhoisted  %8.4f us   %.3fx\n", t_old/1e3, t_new/1e3, t_old/t_new)
@printf("ECEF difference: %.3e m  (=%.2e range samples)\n\n",
        norm3(a_new[1] - a_old[1]), norm3(a_new[1]-a_old[1])/coord.dr)

println("== item 3: does binding the scene-center state actually help? ==")
unbound = SceneCenterStart()
bound = _bind(SceneCenterStart(), coord)
@printf("unbound type: %s\n  bound type: %s\n", typeof(unbound), typeof(bound))
nrm = surface_normal(0.05, -0.03)
t_u = minimum(@benchmark pointgeometry($tf,-242500.0,2179000.0,500.0,$coord,$nrm,$unbound,nothing)).time
t_b = minimum(@benchmark pointgeometry($tf,-242500.0,2179000.0,500.0,$coord,$nrm,$bound,nothing)).time
gu = pointgeometry(tf,-242500.0,2179000.0,500.0,coord,nrm,unbound,nothing)
gb = pointgeometry(tf,-242500.0,2179000.0,500.0,coord,nrm,bound,nothing)
@printf("unbound %8.4f us   bound %8.4f us   %.3fx\n", t_u/1e3, t_b/1e3, t_u/t_b)
@printf("bitwise identical: %s\n\n", gu[1].image_xy === gb[1].image_xy && gu[3].aztime === gb[3].aztime)

println("== whole kernel, repeated to separate signal from noise ==")
for rep in 1:3
    t = @belapsed pairgeometry($C.grid, $C.pair, $C.inputs; transform=$C.tf, window=$C.win,
                               nodata=nodata_from(0.0)) samples=7 evals=1
    @printf("rep %d: %6.1f ms  %6.3f us/pt\n", rep, t*1e3, t*1e6/128^2)
end
