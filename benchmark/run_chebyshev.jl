# The gate on `chebyshev_orbit`: what it is worth on a whole radar window, and what it costs in every
# output band against the default Hermite path.
#
# The band comparison is the part that decides whether the option is safe to offer. A faster
# interpolant that moved a rounded pixel index would be unusable whatever its speed, so this compares
# all nineteen bands of a real window rather than the interpolant alone.
include(joinpath(@__DIR__, "radar_setup.jl"))
using BenchmarkTools, Printf, JSON3, StaticArrays
using ImagePairGeometry: norm3, starttime, stoptime, INT_BANDS, FLOAT_BANDS, GEO2RDR_ITERATIONS

ns(b) = minimum(b).time
bits(x::Float64) = reinterpret(UInt64, x)

const ORB_H = bench_orbit()
const ORB_C = chebyshev_orbit(ORB_H)

println("="^78)
println("INTERPOLANT — accuracy over the whole orbit domain")
println("="^78)
let mdp = 0.0, mdv = 0.0, bp = 0, bv = 0, n = 0
    for t in range(starttime(ORB_H), stoptime(ORB_H); length = 20001)
        a, av = interpolate(ORB_H, t)
        b, bv_ = interpolate(ORB_C, t)
        mdp = max(mdp, norm3(a - b)); mdv = max(mdv, norm3(av - bv_))
        n += 1
        all(i -> bits(a[i]) == bits(b[i]), 1:3) && (bp += 1)
        all(i -> bits(av[i]) == bits(bv_[i]), 1:3) && (bv += 1)
    end
    @printf("position  worst %.3e m    bitwise %5d/%d\n", mdp, bp, n)
    @printf("velocity  worst %.3e m/s  bitwise %5d/%d\n", mdv, bv, n)
end

# The same measurement on a real orbit, because the one above is made on an analytic circle and an
# eight-term Chebyshev series fits a circle far better than it fits a trajectory. A real orbit is
# perturbed — the NISAR granule sampled here has its radius vary by about a kilometer over 340 s, and
# reports velocities differing from its own position derivative by 0.13 m/s — and the deviation is six
# orders larger there. Still negligible in magnitude, and this is the honest figure to quote.
let path = get(ENV, "IPG_REALDATA_DIR", "")
    if isempty(path) || !isfile(joinpath(path, "reference_metadata.json"))
        println("\nreal orbit: set IPG_REALDATA_DIR to include the real-orbit comparison")
    else
        meta = JSON3.read(read(joinpath(path, "reference_metadata.json"), String))
        hx(v) = v isa Number ? Float64(v) : parse(Float64, v.hex)
        real_h = Orbit(; time = [hx(v) for v in meta.orbit.time],
                       position = [SVector{3,Float64}(hx.(p)...) for p in meta.orbit.position],
                       velocity = [SVector{3,Float64}(hx.(v)...) for v in meta.orbit.velocity])
        real_c = chebyshev_orbit(real_h)
        mdp = 0.0
        mdv = 0.0
        for t in range(starttime(real_h), stoptime(real_h); length = 20001)
            a, av = interpolate(real_h, t)
            b, bv_ = interpolate(real_c, t)
            mdp = max(mdp, norm3(a - b))
            mdv = max(mdv, norm3(av - bv_))
        end
        radii = [norm3(p) for p in real_h.position]
        @printf("\nreal orbit (%d state vectors at %g s, radius spread %.0f m)\n",
                length(real_h), real_h.spacing, maximum(radii) - minimum(radii))
        @printf("position  worst %.3e m\n", mdp)
        @printf("velocity  worst %.3e m/s\n", mdv)
    end
end

println()
println("="^78)
println("COST — one call, and a whole window")
println("="^78)
@printf("interpolate, Hermite    %7.2f ns\n", ns(@benchmark interpolate($ORB_H, 305.137)))
@printf("interpolate, Chebyshev  %7.2f ns\n", ns(@benchmark interpolate($ORB_C, 305.137)))

for n in (64, 128)
    ch = bench_case(n; orbit = ORB_H)
    cc = bench_case(n; orbit = ORB_C)
    bh = @benchmark pairgeometry($ch.grid, $ch.pair, $ch.inputs; transform = $ch.tf,
                                 window = $ch.win, nodata = nodata_from(0.0)) samples=3 evals=1 seconds=90
    bc = @benchmark pairgeometry($cc.grid, $cc.pair, $cc.inputs; transform = $cc.tf,
                                 window = $cc.win, nodata = nodata_from(0.0)) samples=3 evals=1 seconds=90
    th, tc = minimum(bh).time, minimum(bc).time
    @printf("window %3dx%-3d  Hermite %6.3f us/pt   Chebyshev %6.3f us/pt   %.3fx\n",
            n, n, th/1e3/(n*n), tc/1e3/(n*n), th/tc)
end

println()
println("="^78)
println("BANDS — every output of a 128x128 window, Chebyshev against Hermite")
println("="^78)
let
    ch = bench_case(128; orbit = ORB_H)
    cc = bench_case(128; orbit = ORB_C)
    rh = pairgeometry(ch.grid, ch.pair, ch.inputs; transform = ch.tf, window = ch.win,
                      nodata = nodata_from(0.0))
    rc = pairgeometry(cc.grid, cc.pair, cc.inputs; transform = cc.tf, window = cc.win,
                      nodata = nodata_from(0.0))
    @printf("valid points: %d of %d\n\n", nvalid(rh), npoints(rh))

    println("integer bands — a single moved index would rule the option out")
    for f in INT_BANDS
        a = getfield(rh, f); b = getfield(rc, f)
        nd = count(!=(0), a .- b)
        @printf("  %-16s %s  (%d of %d differ)\n", f, nd == 0 ? "identical" : "MOVED", nd,
                length(a))
    end

    println("\nfloat bands — relative to each band's own maximum magnitude")
    sentinel = 0.0
    for f in FLOAT_BANDS
        a = getfield(rh, f); b = getfield(rc, f)
        scale = maximum(abs, filter(!=(sentinel), a); init = 0.0)
        worst = 0.0
        nbit = 0
        for k in eachindex(a)
            (a[k] == sentinel && b[k] == sentinel) && continue
            bits(a[k]) == bits(b[k]) && (nbit += 1)
            scale > 0 && (worst = max(worst, abs(a[k] - b[k]) / scale))
        end
        @printf("  %-16s %.3e relative   bitwise %d/%d\n", f, worst, nbit, length(a))
    end
    println("\nThe radar fixture asserts the float bands against isce3 at 2e-4 relative")
    println("(`test/radar_geogrid.jl`), so that is the bound these have to sit inside.")
end
