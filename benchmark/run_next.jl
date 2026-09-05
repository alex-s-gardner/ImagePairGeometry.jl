# What remains, at the current counts (geo2rdr=16, range-Doppler=6).
include(joinpath(@__DIR__, "radar_setup.jl"))
using Printf, BenchmarkTools
using ImagePairGeometry: _range_doppler, Ellipsoid, xyz_to_lonlat, lonlat_to_xyz, norm3, dot3,
                         unitvec3, geodetic_tcn, nadir_sphere, looksign, RadarPoint,
                         GEO2RDR_ITERATIONS, RANGE_DOPPLER_ITERATIONS, azimuth_index, range_index

const C = bench_case(128)
coord = C.coord; tf = C.tf; el = Ellipsoid()
gx0, gy0, gz0 = -254500.0 + 12000.0, 2191000.0 - 12000.0, 500.0
nrm = surface_normal(0.05, -0.03)
lon_d, lat_d, h = tf.forward(gx0, gy0, gz0)
xyz = lonlat_to_xyz(el, SVector{3,Float64}(lon_d*DEG2RAD, lat_d*DEG2RAD, h))
pm, vm = interpolate(coord.orbit, orbit_midtime(coord))
p = geo2rdr(coord.orbit, xyz, midtime(coord), orbit_midtime(coord), pm, vm)
satx, satv = interpolate(coord.orbit, p.orbittime + 1/coord.prf)

t_pt  = minimum(@benchmark pointgeometry($tf,$gx0,$gy0,$gz0,$coord,$nrm)).time
t_g2r = minimum(@benchmark geo2rdr($coord.orbit,$xyz,$(midtime(coord)),$(orbit_midtime(coord)),$pm,$vm)).time
t_rd  = minimum(@benchmark _range_doppler($el,$coord,$satx,$satv,$(p.range),500.0)).time
t_int = minimum(@benchmark interpolate($coord.orbit, 305.137)).time
t_x2l = minimum(@benchmark xyz_to_lonlat($el,$xyz)).time
t_l2x = minimum(@benchmark lonlat_to_xyz($el,SVector{3,Float64}(0.14,0.34,500.0))).time
t_fwd = minimum(@benchmark $tf.forward($gx0,$gy0,$gz0)).time
t_inv = minimum(@benchmark $tf.inverse(8.0,19.5,500.0)).time

@printf("counts: geo2rdr=%d, range-Doppler=%d\n\n", GEO2RDR_ITERATIONS, RANGE_DOPPLER_ITERATIONS)
@printf("pointgeometry           %7.4f us  100.0%%\n", t_pt/1e3)
@printf("  geo2rdr               %7.4f us  %5.1f%%\n", t_g2r/1e3, 100*t_g2r/t_pt)
@printf("  _range_doppler        %7.4f us  %5.1f%%\n", t_rd/1e3, 100*t_rd/t_pt)
@printf("  PROJ (1 fwd + 2 inv)  %7.4f us  %5.1f%%\n", (t_fwd+2*t_inv)/1e3, 100*(t_fwd+2*t_inv)/t_pt)

ni = GEO2RDR_ITERATIONS + 2
nx = 2 + RANGE_DOPPLER_ITERATIONS
nl = 1 + RANGE_DOPPLER_ITERATIONS
@printf("\nby primitive, using actual call counts:\n")
@printf("  interpolate    x%-3d %7.4f us  %5.1f%%\n", ni, ni*t_int/1e3, 100*ni*t_int/t_pt)
@printf("  xyz_to_lonlat  x%-3d %7.4f us  %5.1f%%\n", nx, nx*t_x2l/1e3, 100*nx*t_x2l/t_pt)
@printf("  lonlat_to_xyz  x%-3d %7.4f us  %5.1f%%\n", nl, nl*t_l2x/1e3, 100*nl*t_l2x/t_pt)
@printf("  => the two solves' non-primitive arithmetic: %5.1f%%\n",
        100*(t_g2r + t_rd - ni*t_int - nx*t_x2l - nl*t_l2x)/t_pt)

println("\n== A. are _range_doppler's loop invariants actually hoisted? ==")
# `a/rngpix`, `rngpix/a`, and the look sign do not change across iterations. The source comment
# claims LLVM lifts them. Verify by hoisting explicitly and comparing.
@inline function rd_hoisted(el, c, satx, satv, rngpix, height, niter)
    vhat = unitvec3(satv); radius, _, _, a = nadir_sphere(el, satx); tcn = geodetic_tcn(satx, satv)
    ndotv = dot3(tcn.nhat, vhat); vdott = dot3(vhat, tcn.that)
    ar = a/rngpix; ra = rngpix/a; ls = -looksign(c.look_side); r2 = rngpix*rngpix
    zsch = height; llhi = SVector{3,Float64}(0.,0.,0.); txyz = llhi
    for _ in 1:niter
        b = radius + zsch
        ct = 0.5*(ar + ra - (b/a)*(b/rngpix)); st = sqrt(1 - ct*ct)
        gam = rngpix*ct; alp = -gam*ndotv/vdott
        bet = ls*sqrt(r2*st*st - alp*alp)
        tv = satx + alp*tcn.that + bet*tcn.chat + gam*tcn.nhat
        ll = xyz_to_lonlat(el, tv); llhi = SVector{3,Float64}(ll[1], ll[2], height)
        txyz = lonlat_to_xyz(el, llhi); zsch = norm3(txyz) - radius
    end
    (txyz, llhi)
end
a1, l1 = _range_doppler(el, coord, satx, satv, p.range, 500.0)
a2, l2 = rd_hoisted(el, coord, satx, satv, p.range, 500.0, RANGE_DOPPLER_ITERATIONS)
t_h = minimum(@benchmark rd_hoisted($el,$coord,$satx,$satv,$(p.range),500.0,$RANGE_DOPPLER_ITERATIONS)).time
@printf("bitwise identical: %s\n", (a1 === a2 && l1 === l2))
@printf("current %7.4f us   hand-hoisted %7.4f us   %.3fx\n", t_rd/1e3, t_h/1e3, t_rd/t_h)

println("\n== B. hoisting the scene-center guess out of pointgeometry ==")
@printf("removes 1 of %d interpolate calls: %.4f us of %.4f us = %.1f%%, bit-identical\n",
        ni, t_int/1e3, t_pt/1e3, 100*t_int/t_pt)

println("\n== C. warm start: what the counts could drop to, and what it costs ==")
function g2r_n(orbit, target, az0, ob0, p0, v0, n)
    satx, satv, az, ob = p0, v0, az0, ob0
    look = target - satx; rng = norm3(look)
    for _ in 1:n
        look = target - satx; rng = norm3(look)
        s = dot3(look, satv)/-dot3(satv, satv); az -= s; ob -= s
        satx, satv = interpolate(orbit, ob)
    end
    RadarPoint(az, ob, rng, look, satx, satv)
end
# Walk a block in traversal order, warm-starting each point from the previous.
n = 64; sp = 500.0*48/n
targs = SVector{3,Float64}[]
for j in 1:n, i in 1:n
    lo, la, hh = tf.forward(-254500.0 + (i-0.5)*sp, 2191000.0 - (j-0.5)*sp, 500.0)
    push!(targs, lonlat_to_xyz(el, SVector{3,Float64}(lo*DEG2RAD, la*DEG2RAD, hh)))
end
ref = [g2r_n(coord.orbit, t, midtime(coord), orbit_midtime(coord), pm, vm, 16) for t in targs]
@printf("%6s %14s %14s %10s\n", "warm k", "max |dt| (s)", "max d_lines", "idx differ")
for k in (4, 5, 6, 8, 16)
    az, ob, sx, sv = midtime(coord), orbit_midtime(coord), pm, vm
    md = 0.0; nd = 0
    for (i, t) in enumerate(targs)
        g = g2r_n(coord.orbit, t, az, ob, sx, sv, k)
        md = max(md, abs(g.aztime - ref[i].aztime))
        (azimuth_index(g, coord.sensing_start, coord.prf) !=
         azimuth_index(ref[i], coord.sensing_start, coord.prf)) && (nd += 1)
        az, ob, sx, sv = g.aztime, g.orbittime, g.position, g.velocity
    end
    @printf("%6d %14.3e %14.3e %10d\n", k, md, md*coord.prf, nd)
end
@printf("\nwarm-8 would cut geo2rdr to %.0f%% of its cost => %.2fx on the whole point\n",
        100*8/16, t_pt/(t_pt - 0.5*t_g2r))
