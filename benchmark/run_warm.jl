include(joinpath(@__DIR__, "radar_setup.jl"))
using Printf
using ImagePairGeometry: RadarPoint, norm3, dot3, azimuth_index, range_index, Ellipsoid,
                         cround

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

# True column-major traversal, as `_fill_geometry!` walks the window.
const n = 96
const spacing = 500.0*48/n
coord = bench_coord(); el = Ellipsoid(); tf = bench_tf()
pm, vm = interpolate(coord.orbit, orbit_midtime(coord))

targets = Matrix{SVector{3,Float64}}(undef, n, n)
for j in 1:n, i in 1:n
    gx = -254500.0 + (i - 0.5)*spacing
    gy = 2191000.0 - (j - 0.5)*spacing
    lon, lat, h = tf.forward(gx, gy, 500.0)
    targets[i, j] = lonlat_to_xyz(el, SVector{3,Float64}(lon*DEG2RAD, lat*DEG2RAD, h))
end

ref = [geo2rdr_n(coord.orbit, targets[k], midtime(coord), orbit_midtime(coord), pm, vm, 51)
       for k in eachindex(targets)]

println("== warm start in true traversal order, $(n)^2 = $(n*n) points, $(round(spacing,digits=1)) m spacing ==")
@printf("%6s %14s %14s %10s %10s\n", "niter", "max |dt| (s)", "max d_azline", "azidx≠", "rngidx≠")
for k in (2, 3, 4, 5, 6, 8, 12, 51)
    az, ob, sx, sv = midtime(coord), orbit_midtime(coord), pm, vm
    maxdt = 0.0; nidx = 0; nrng = 0
    for kk in eachindex(targets)
        g = geo2rdr_n(coord.orbit, targets[kk], az, ob, sx, sv, k)
        r = ref[kk]
        maxdt = max(maxdt, abs(g.aztime - r.aztime))
        azimuth_index(g, coord.sensing_start, coord.prf) !=
            azimuth_index(r, coord.sensing_start, coord.prf) && (nidx += 1)
        range_index(g, coord.starting_range, coord.dr) !=
            range_index(r, coord.starting_range, coord.dr) && (nrng += 1)
        az, ob, sx, sv = g.aztime, g.orbittime, g.position, g.velocity
    end
    @printf("%6d %14.3e %14.3e %10d %10d\n", k, maxdt, maxdt*coord.prf, nidx, nrng)
end

println("\n== cold start, for comparison at the same iteration counts ==")
for k in (5, 8, 10, 12)
    maxdt = 0.0; nidx = 0
    for kk in eachindex(targets)
        g = geo2rdr_n(coord.orbit, targets[kk], midtime(coord), orbit_midtime(coord), pm, vm, k)
        maxdt = max(maxdt, abs(g.aztime - ref[kk].aztime))
        azimuth_index(g, coord.sensing_start, coord.prf) !=
            azimuth_index(ref[kk], coord.sensing_start, coord.prf) && (nidx += 1)
    end
    @printf("cold niter=%2d: max |dt| = %.3e s (%.3e lines), %d azidx differ\n",
            k, maxdt, maxdt*coord.prf, nidx)
end
