include(joinpath(@__DIR__, "radar_setup.jl"))
using Profile, Printf
using ImagePairGeometry: _range_doppler, geodetic_tcn, nadir_sphere
using BenchmarkTools

const C = bench_case(192)
pairgeometry(C.grid, C.pair, C.inputs; transform = C.tf, window = C.win, nodata = nodata_from(0.0))

Profile.clear()
Profile.init(n = 10_000_000, delay = 0.0002)
@profile for _ in 1:3
    pairgeometry(C.grid, C.pair, C.inputs; transform = C.tf, window = C.win, nodata = nodata_from(0.0))
end

data, lidict = Profile.retrieve()

function backtraces(data)
    out = Vector{Vector{UInt64}}()
    cur = UInt64[]
    for d in data
        if d == 0
            isempty(cur) || push!(out, copy(cur)); empty!(cur)
        else
            push!(cur, d)
        end
    end
    return out
end

const BTS = backtraces(data)
const NTOT = length(BTS)

function flat_self(bts, lidict)
    counts = Dict{String,Int}()
    for bt in bts
        isempty(bt) && continue
        for fr in get(lidict, bt[1], [])
            key = "$(fr.func)  @ $(basename(String(fr.file))):$(fr.line)"
            counts[key] = get(counts, key, 0) + 1
            break
        end
    end
    return counts
end

let counts = flat_self(BTS, lidict)
    println("== self-time leaves, $NTOT samples ==")
    for (k, v) in sort(collect(counts); by = kv -> -kv[2])[1:min(30, end)]
        @printf("%5.1f%%  %5d  %s\n", 100v/NTOT, v, k)
    end
end

function cumulative(pat, bts, lidict)
    n = 0
    for bt in bts
        hit = false
        for ip in bt, fr in get(lidict, ip, [])
            occursin(pat, String(fr.func)) && (hit = true; break)
        end
        hit && (n += 1)
    end
    return n
end

println("\n== cumulative (inclusive) ==")
for p in ("_fill_geometry!", "pointgeometry", "geo2rdr", "_range_doppler", "interpolate",
          "xyz_to_lonlat", "lonlat_to_xyz", "GeocentricToLonLat", "LonLatToGeocentric",
          "proj_trans", "_to_grid", "geodetic_tcn", "nadir_sphere", "cbrt", "atan", "sqrt")
    @printf("%6.1f%%  %s\n", 100*cumulative(p, BTS, lidict)/NTOT, p)
end

println("\n== _range_doppler in isolation ==")
coord = C.coord; el = Ellipsoid()
gx0, gy0, gz0 = -254500.0 + 12000.0, 2191000.0 - 12000.0, 500.0
lon_d, lat_d, h = C.tf.forward(gx0, gy0, gz0)
xyz = lonlat_to_xyz(el, SVector{3,Float64}(lon_d*DEG2RAD, lat_d*DEG2RAD, h))
pm, vm = interpolate(coord.orbit, orbit_midtime(coord))
p = geo2rdr(coord.orbit, xyz, midtime(coord), orbit_midtime(coord), pm, vm)
satx, satv = interpolate(coord.orbit, p.orbittime + 1/coord.prf)
b = @benchmark _range_doppler($el, $coord, $satx, $satv, $p.range, 500.0)
@printf("_range_doppler: %8.3f us\n", minimum(b).time/1e3)
b2 = @benchmark geodetic_tcn($satx, $satv)
@printf("geodetic_tcn:              %8.3f us\n", minimum(b2).time/1e3)
