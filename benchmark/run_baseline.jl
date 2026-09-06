include(joinpath(@__DIR__, "radar_setup.jl"))
using BenchmarkTools, Printf

const C = bench_case(128)

# Warm up + correctness anchor
r0 = pairgeometry(C.grid, C.pair, C.inputs; transform = C.tf, window = C.win,
                  nodata = nodata_from(0.0))
@printf("nvalid = %d / %d\n", nvalid(r0), npoints(r0))

println("\n== whole-kernel timing, per grid point ==")
for n in (64, 128, 256)
    c = bench_case(n)
    b = @benchmark pairgeometry($c.grid, $c.pair, $c.inputs; transform = $c.tf,
                                window = $c.win, nodata = nodata_from(0.0)) samples=3 evals=1 seconds=60
    t = minimum(b).time
    @printf("n=%4d  %8.1f ms  %7.1f us/pt  %10d allocs  %8.2f MiB\n",
            n, t/1e6, t/1e3/(n*n), minimum(b).allocs, minimum(b).memory/2^20)
end

println("\n== component timing, one point ==")
coord = C.coord; tf = C.tf
el = Ellipsoid()
gx, gy, gz = -254500.0 + 12000.0, 2191000.0 - 12000.0, 500.0
normal = surface_normal(0.05, -0.03)
lon_d, lat_d, h = tf.forward(gx, gy, gz)
xyz = lonlat_to_xyz(el, SVector{3,Float64}(lon_d*DEG2RAD, lat_d*DEG2RAD, h))
pm, vm = interpolate(coord.orbit, orbit_midtime(coord))

bp  = @benchmark pointgeometry($tf, $gx, $gy, $gz, $coord, $normal)
bg  = @benchmark geo2rdr($coord.orbit, $xyz, midtime($coord), orbit_midtime($coord), $pm, $vm)
bi  = @benchmark interpolate($coord.orbit, 300.0)
bfw = @benchmark $tf.forward($gx, $gy, $gz)
bin = @benchmark $tf.inverse(8.0, 19.5, 500.0)
bl2x= @benchmark lonlat_to_xyz($el, SVector{3,Float64}(0.14, 0.34, 500.0))
bx2l= @benchmark xyz_to_lonlat($el, $xyz)

for (name, b) in (("pointgeometry (whole)", bp), ("geo2rdr", bg),
                  ("orbit interpolate x1", bi), ("proj forward", bfw), ("proj inverse", bin),
                  ("lonlat_to_xyz", bl2x), ("xyz_to_lonlat", bx2l))
    m = minimum(b)
    @printf("%-24s %10.3f us  %6d allocs %8d B\n", name, m.time/1e3, m.allocs, m.memory)
end
