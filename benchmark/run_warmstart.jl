# Does WarmStart cost accuracy, and what does it buy? Also: is the default path unchanged?
include(joinpath(@__DIR__, "radar_setup.jl"))
using Printf, BenchmarkTools
using ImagePairGeometry: INT_BANDS, FLOAT_BANDS, SceneCenterStart, WarmStart, GeometryParams

const C = bench_case(128)
run(p) = pairgeometry(C.grid, C.pair, C.inputs; transform = C.tf, window = C.win,
                      params = p, nodata = nodata_from(0.0))
cold = run(GeometryParams())

println("== accuracy: WarmStart vs the cold default, every band, 128^2 ==")
@printf("%6s | %12s %10s | %14s | %s\n", "iters", "int pts diff", "max |d|", "max float rel", "verdict")
for k in (8, 10, 12, 16)
    w = run(GeometryParams(zero_doppler_start = WarmStart(iterations = k)))
    nd = 0; mx = 0
    for f in INT_BANDS
        d = Int.(getfield(w, f)) .- Int.(getfield(cold, f))
        nd += count(!=(0), d); mx = max(mx, maximum(abs, d; init = 0))
    end
    mr = 0.0
    for f in FLOAT_BANDS
        a = getfield(w, f); b = getfield(cold, f)
        for i in eachindex(a, b)
            (a[i] == -32767.0 || b[i] == -32767.0) && continue
            mr = max(mr, abs(a[i] - b[i])/max(abs(b[i]), 1.0))
        end
    end
    @printf("%6d | %12d %10d | %14.3e | %s\n", k, nd, mx, mr,
            nd == 0 ? "no index moved" : "INDICES MOVED")
end

println("\n== speed ==")
tc = @belapsed run(GeometryParams()) samples=5 evals=1
@printf("%-28s %7.1f ms  %6.3f us/pt  %s\n", "SceneCenterStart (default)", tc*1e3, tc*1e6/128^2, "1.00x")
for k in (8, 10)
    p = GeometryParams(zero_doppler_start = WarmStart(iterations = k))
    t = @belapsed run($p) samples=5 evals=1
    @printf("%-28s %7.1f ms  %6.3f us/pt  %.2fx\n", "WarmStart($k)", t*1e3, t*1e6/128^2, tc/t)
end

println("\n== the default path must be untouched: compare against a run with no params at all ==")
c2 = pairgeometry(C.grid, C.pair, C.inputs; transform = C.tf, window = C.win,
                  nodata = nodata_from(0.0))
same = all(getfield(cold, f) == getfield(c2, f) for f in INT_BANDS) &&
       all(reinterpret(UInt64, getfield(cold, f)) == reinterpret(UInt64, getfield(c2, f))
           for f in FLOAT_BANDS)
println("explicit GeometryParams() === default kwargs: ", same)

println("\n== allocation: the seed must not allocate per point ==")
for (nm, p) in (("cold", GeometryParams()),
                ("warm", GeometryParams(zero_doppler_start = WarmStart())))
    b = @benchmark run($p) samples=3 evals=1
    @printf("%-6s %6d allocs  %8.2f MiB\n", nm, minimum(b).allocs, minimum(b).memory/2^20)
end
