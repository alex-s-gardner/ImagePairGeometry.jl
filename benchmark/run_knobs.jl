include(joinpath(@__DIR__, "radar_setup.jl"))
using Printf, BenchmarkTools
using ImagePairGeometry: RadarPoint, norm3, dot3, GEO2RDR_ITERATIONS, RANGE_DOPPLER_ITERATIONS,
                         range_index, azimuth_index, Ellipsoid, geodetic_tcn, nadir_sphere,
                         looksign, unitvec3, xyz_to_lonlat, lonlat_to_xyz, cround

# geo2rdr with an arbitrary iteration count, otherwise identical.
function geo2rdr_n(orbit, target, aztime0, orbittime0, pos0, vel0, niter)
    satx, satv, aztime, orbittime = pos0, vel0, aztime0, orbittime0
    look = target - satx; rngpix = norm3(look)
    for _ in 1:niter
        look = target - satx; rngpix = norm3(look)
        step = dot3(look, satv) / -dot3(satv, satv)
        aztime -= step; orbittime -= step
        satx, satv = interpolate(orbit, orbittime)
    end
    return RadarPoint(aztime, orbittime, rngpix, look, satx, satv)
end

const C = bench_case(96)
coord = C.coord; el = Ellipsoid(); tf = C.tf
pm, vm = interpolate(coord.orbit, orbit_midtime(coord))
t_az0, t_ob0 = midtime(coord), orbit_midtime(coord)

# A spread of grid points across the window
pts = SVector{3,Float64}[]
n = 96; spacing = 500.0*48/n
for i in 1:8:n, j in 1:8:n
    gx = -254500.0 + (i - 0.5)*spacing
    gy = 2191000.0 - (j - 0.5)*spacing
    lon, lat, h = tf.forward(gx, gy, 500.0)
    push!(pts, lonlat_to_xyz(el, SVector{3,Float64}(lon*DEG2RAD, lat*DEG2RAD, h)))
end
@printf("%d probe points\n\n", length(pts))

println("== geo2rdr iteration count: cost of truncating 51 ==")
ref = [geo2rdr_n(coord.orbit, x, t_az0, t_ob0, pm, vm, 51) for x in pts]
@printf("%6s %14s %14s %12s %12s\n", "niter", "max |dt| (s)", "max d_azline", "max d_range(m)", "azidx diff")
for k in (8, 10, 12, 14, 16, 18, 20, 25, 30, 51)
    got = [geo2rdr_n(coord.orbit, x, t_az0, t_ob0, pm, vm, k) for x in pts]
    dt   = maximum(abs(g.aztime - r.aztime) for (g, r) in zip(got, ref))
    dr   = maximum(abs(g.range  - r.range)  for (g, r) in zip(got, ref))
    nidx = count(azimuth_index(g, coord.sensing_start, coord.prf) !=
                 azimuth_index(r, coord.sensing_start, coord.prf) for (g, r) in zip(got, ref))
    nrng = count(range_index(g, coord.starting_range, coord.dr) !=
                 range_index(r, coord.starting_range, coord.dr) for (g, r) in zip(got, ref))
    @printf("%6d %14.3e %14.3e %12.3e   %d azidx, %d rngidx differ\n",
            k, dt, dt*coord.prf, dr, nidx, nrng)
end

println("\n== warm start: previous point's solution as the initial guess ==")
# Walk the probe points in grid order, each starting from the previous answer.
for k in (3, 4, 5, 6, 8, 51)
    aztime, orbittime, sx, sv = t_az0, t_ob0, pm, vm
    maxdt = 0.0; nidx = 0
    for (i, x) in enumerate(pts)
        g = geo2rdr_n(coord.orbit, x, aztime, orbittime, sx, sv, k)
        maxdt = max(maxdt, abs(g.aztime - ref[i].aztime))
        azimuth_index(g, coord.sensing_start, coord.prf) !=
            azimuth_index(ref[i], coord.sensing_start, coord.prf) && (nidx += 1)
        aztime, orbittime, sx, sv = g.aztime, g.orbittime, g.position, g.velocity
    end
    @printf("warm niter=%2d: max |dt| = %.3e s (%.3e lines), %d azidx differ\n",
            k, maxdt, maxdt*coord.prf, nidx)
end

println("\n== range-Doppler solve iteration count ==")
using ImagePairGeometry: _range_doppler
function rd_n(el, c, satx, satv, rngpix, height, niter)
    vhat = unitvec3(satv); radius, _, _, a = nadir_sphere(el, satx); tcn = geodetic_tcn(satx, satv)
    ndotv = dot3(tcn.nhat, vhat); vdott = dot3(vhat, tcn.that)
    zsch = height; llhi = SVector{3,Float64}(0.,0.,0.); txyz = llhi
    for _ in 1:niter
        b = radius + zsch
        ct = 0.5*(a/rngpix + rngpix/a - (b/a)*(b/rngpix)); st = sqrt(1 - ct*ct)
        gam = rngpix*ct; alp = -gam*ndotv/vdott
        bet = -looksign(c.look_side)*sqrt(rngpix*rngpix*st*st - alp*alp)
        tv = satx + alp*tcn.that + bet*tcn.chat + gam*tcn.nhat
        ll = xyz_to_lonlat(el, tv); llhi = SVector{3,Float64}(ll[1], ll[2], height)
        txyz = lonlat_to_xyz(el, llhi); zsch = norm3(txyz) - radius
    end
    return (txyz, llhi)
end
for k in (2, 3, 4, 5, 6, 10)
    maxd = 0.0
    for (i, x) in enumerate(pts)
        p = ref[i]
        sx, sv = interpolate(coord.orbit, p.orbittime + 1/coord.prf)
        a, _ = rd_n(el, coord, sx, sv, p.range, 500.0, k)
        b, _ = rd_n(el, coord, sx, sv, p.range, 500.0, 10)
        maxd = max(maxd, norm3(a - b))
    end
    @printf("rd niter=%2d: max ECEF difference vs 10 = %.3e m\n", k, maxd)
end

println("\n== projected `interpolate` cost floor ==")
b1 = @benchmark interpolate($coord.orbit, 300.0 + 0.01)
@printf("interpolate:               %8.4f us\n", minimum(b1).time/1e3)
b2 = @benchmark geo2rdr_n($coord.orbit, $(pts[1]), $t_az0, $t_ob0, $pm, $vm, 51)
b3 = @benchmark geo2rdr_n($coord.orbit, $(pts[1]), $t_az0, $t_ob0, $pm, $vm, 16)
b4 = @benchmark geo2rdr_n($coord.orbit, $(pts[1]), $t_az0, $t_ob0, $pm, $vm, 5)
@printf("geo2rdr:          %8.4f us\n", minimum(b2).time/1e3)
@printf("geo2rdr:          %8.4f us\n", minimum(b3).time/1e3)
@printf("geo2rdr  5 iters (warm):   %8.4f us\n", minimum(b4).time/1e3)
