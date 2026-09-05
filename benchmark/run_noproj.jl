# The production configuration: `fast_transform` rather than PROJ. The radar path's three transform
# calls per point go through it, so this is not a small correction to the profile.
include(joinpath(@__DIR__, "radar_setup.jl"))
using Printf, BenchmarkTools
using ImagePairGeometry: fast_transform, INT_BANDS, FLOAT_BANDS, GeometryParams, WarmStart,
                         Ellipsoid, lonlat_to_xyz

const C = bench_case(128)
const FT = fast_transform(32632, 4326; always_xy = true)

println("== per-call cost of the transform pair ==")
@printf("PROJ    forward %8.4f us   inverse %8.4f us\n",
        minimum(@benchmark $C.tf.forward(-242500.0, 2179000.0, 500.0)).time/1e3,
        minimum(@benchmark $C.tf.inverse(8.0, 19.5, 500.0)).time/1e3)
@printf("native  forward %8.4f us   inverse %8.4f us\n",
        minimum(@benchmark $FT.forward(-242500.0, 2179000.0, 500.0)).time/1e3,
        minimum(@benchmark $FT.inverse(8.0, 19.5, 500.0)).time/1e3)

println("\n== whole kernel: PROJ vs native, cold and warm ==")
res = Dict{String,Any}()
for (tfname, tf) in (("PROJ", C.tf), ("native", FT))
    for (pname, p) in (("cold", GeometryParams()),
                       ("warm", GeometryParams(zero_doppler_start = WarmStart())))
        t = @belapsed pairgeometry($C.grid, $C.pair, $C.inputs; transform = $tf,
                                   window = $C.win, params = $p, nodata = nodata_from(0.0)) samples=7 evals=1
        r = pairgeometry(C.grid, C.pair, C.inputs; transform = tf, window = C.win,
                         params = p, nodata = nodata_from(0.0))
        res["$tfname/$pname"] = (t, r)
        @printf("%-8s %-5s %7.1f ms  %6.3f us/pt\n", tfname, pname, t*1e3, t*1e6/128^2)
    end
end
base = res["PROJ/cold"][1]
println()
for k in ("PROJ/cold", "PROJ/warm", "native/cold", "native/warm")
    @printf("  %-12s %.2fx vs PROJ/cold\n", k, base/res[k][1])
end

println("\n== does the native transform move any band? ==")
for k in ("native/cold", "native/warm")
    a = res[k][2]; b = res["PROJ/cold"][2]
    nd = sum(count(!=(0), Int.(getfield(a,f)) .- Int.(getfield(b,f))) for f in INT_BANDS)
    mr = 0.0
    for f in FLOAT_BANDS
        x = getfield(a,f); y = getfield(b,f)
        for i in eachindex(x,y)
            (x[i]==-32767.0||y[i]==-32767.0) && continue
            mr = max(mr, abs(x[i]-y[i])/max(abs(y[i]),1.0))
        end
    end
    @printf("  %-12s integer bands differing: %5d   float max rel: %.3e\n", k, nd, mr)
end

println("\n== remaining profile with the native transform ==")
using ImagePairGeometry: _range_doppler, geo2rdr, interpolate, orbit_midtime, midtime,
                         xyz_to_lonlat, surface_normal, DEG2RAD, pointgeometry
el = Ellipsoid(); nrm = surface_normal(0.05, -0.03)
lo, la, hh = FT.forward(-242500.0, 2179000.0, 500.0)
xyz = lonlat_to_xyz(el, SVector{3,Float64}(lo*DEG2RAD, la*DEG2RAD, hh))
pm, vm = interpolate(C.coord.orbit, orbit_midtime(C.coord))
p = geo2rdr(C.coord.orbit, xyz, midtime(C.coord), orbit_midtime(C.coord), pm, vm)
sx, sv = interpolate(C.coord.orbit, p.orbittime + 1/C.coord.prf)
t_pt = minimum(@benchmark pointgeometry($FT, -242500.0, 2179000.0, 500.0, $C.coord, $nrm)).time
for (nm, t) in (("geo2rdr", minimum(@benchmark geo2rdr($C.coord.orbit,$xyz,$(midtime(C.coord)),$(orbit_midtime(C.coord)),$pm,$vm)).time),
                ("_range_doppler", minimum(@benchmark _range_doppler($el,$C.coord,$sx,$sv,$(p.range),500.0)).time),
                ("transform x3", minimum(@benchmark $FT.forward(-242500.0,2179000.0,500.0)).time +
                                 2*minimum(@benchmark $FT.inverse(8.0,19.5,500.0)).time),
                ("interpolate x18", 18*minimum(@benchmark interpolate($C.coord.orbit, 305.137)).time),
                ("xyz_to_lonlat x8", 8*minimum(@benchmark xyz_to_lonlat($el,$xyz)).time))
    @printf("  %-18s %8.4f us  %5.1f%%\n", nm, t/1e3, 100*t/t_pt)
end
@printf("  %-18s %8.4f us  100.0%%\n", "pointgeometry", t_pt/1e3)
