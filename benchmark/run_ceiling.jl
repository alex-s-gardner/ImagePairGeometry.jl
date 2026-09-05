# The speed/exactness frontier for `interpolate`. Bitwise position agreement with isce3 is a tested
# contract (`radar_numerics.jl:392`), so measure what each relaxation buys and costs.
include(joinpath(@__DIR__, "radar_setup.jl"))
using Printf, BenchmarkTools, LinearAlgebra
using ImagePairGeometry: statetime, _hermite_index, norm3, starttime, stoptime

# --- variant A: cache reciprocals AND use them (multiply, not divide) ---------------------
struct RecipCache
    o::Orbit
    tn::Vector{NTuple{4,Float64}}
    sepsum::Vector{NTuple{4,Float64}}
    recip::Vector{NTuple{4,NTuple{4,Float64}}}
end
function RecipCache(o::Orbit)
    nwin = max(1, length(o) - 3)
    tn = Vector{NTuple{4,Float64}}(undef, nwin)
    ss = Vector{NTuple{4,Float64}}(undef, nwin)
    rc = Vector{NTuple{4,NTuple{4,Float64}}}(undef, nwin)
    for idx in 1:nwin
        t = ntuple(i -> statetime(o, idx+i-1), 4); tn[idx] = t
        ss[idx] = ntuple(4) do i
            s = 0.0; for j in 1:4; j == i && continue; s += 1.0/(t[i]-t[j]); end; s
        end
        rc[idx] = ntuple(i -> ntuple(j -> i == j ? 0.0 : 1.0/(t[i]-t[j]), 4), 4)
    end
    RecipCache(o, tn, ss, rc)
end
@inline function recip_interp(c::RecipCache, t::Real)
    o = c.o; tt = Float64(t); idx = _hermite_index(o, tt)
    tn = c.tn[idx]; sepsum = c.sepsum[idx]; r = c.recip[idx]
    f1 = ntuple(i -> tt - tn[i], 4)
    f0 = ntuple(i -> 1.0 - 2.0*sepsum[i]*f1[i], 4)
    h = ntuple(4) do i
        p = 1.0; for j in 1:4; j == i && continue; p *= f1[j]*r[i][j]; end; p
    end
    pos = SVector{3,Float64}(0.,0.,0.)
    for i in 1:4; pos += h[i]*h[i]*(o.position[idx+i-1]*f0[i] + o.velocity[idx+i-1]*f1[i]); end
    hdot = ntuple(4) do i
        acc = 0.0
        for j in 1:4
            j == i && continue
            pr = r[i][j]
            for k in 1:4; (k == i || k == j) && continue; pr *= f1[k]*r[i][k]; end
            acc += pr
        end
        acc
    end
    g1 = ntuple(i -> h[i] + 2.0*hdot[i]*f1[i], 4)
    g0 = ntuple(i -> 2.0*(f0[i]*hdot[i] - sepsum[i]*h[i]), 4)
    vel = SVector{3,Float64}(0.,0.,0.)
    for i in 1:4; vel += h[i]*(o.position[idx+i-1]*g0[i] + o.velocity[idx+i-1]*g1[i]); end
    (pos, vel)
end

# --- variant B: full degree-7 polynomial per interval -------------------------------------
struct PolyCache
    t0::Float64; spacing::Float64; n::Int
    cp::Vector{NTuple{8,SVector{3,Float64}}}
    cv::Vector{NTuple{7,SVector{3,Float64}}}
end
function PolyCache(o::Orbit)
    n = length(o); nwin = max(1, n-3)
    cp = Vector{NTuple{8,SVector{3,Float64}}}(undef, nwin)
    cv = Vector{NTuple{7,SVector{3,Float64}}}(undef, nwin)
    for idx in 1:nwin
        t1 = statetime(o, idx); A = zeros(8,8)
        for i in 1:4
            s = statetime(o, idx+i-1) - t1
            for k in 0:7
                A[i,k+1] = s^k; A[4+i,k+1] = k == 0 ? 0.0 : k*s^(k-1)
            end
        end
        F = lu(A)
        cols = map(d -> F \ [(i<=4 ? o.position[idx+i-1][d] : o.velocity[idx+i-5][d]) for i in 1:8], 1:3)
        cp[idx] = ntuple(k -> SVector{3,Float64}(cols[1][k], cols[2][k], cols[3][k]), 8)
        cv[idx] = ntuple(k -> k*cp[idx][k+1], 7)
    end
    PolyCache(o.t0, o.spacing, n, cp, cv)
end
@inline function poly_interp(p::PolyCache, t::Real)
    tt = Float64(t)
    idx = clamp(Int(trunc((tt-p.t0)/p.spacing)) + 1 - 2, 0, p.n-4) + 1
    s = tt - (p.t0 + (idx-1)*p.spacing)
    c = p.cp[idx]; pos = c[8]
    for k in 7:-1:1; pos = pos*s + c[k]; end
    d = p.cv[idx]; vel = d[7]
    for k in 6:-1:1; vel = vel*s + d[k]; end
    (pos, vel)
end

const RC = RecipCache(ORB)
const PC = PolyCache(ORB)

function report(name, f)
    wp = 0; wv = 0; mp = 0.0; mv = 0.0; nb = 0; nt = 0
    for t in range(starttime(ORB), stoptime(ORB); length = 50_001)
        a, av = interpolate(ORB, t); b, bv = f(t)
        nt += 1; a === b && (nb += 1)
        for k in 1:3
            wp = max(wp, abs(reinterpret(Int64,a[k]) - reinterpret(Int64,b[k])))
            wv = max(wv, abs(reinterpret(Int64,av[k]) - reinterpret(Int64,bv[k])))
        end
        mp = max(mp, norm3(a-b)); mv = max(mv, norm3(av-bv))
    end
    tb = minimum(@benchmark interpolate($ORB, 305.137)).time
    tf = minimum(@benchmark $f(305.137)).time
    @printf("%-18s %7.2fx | %6d %8d | %10.2e %10.2e | %s\n", name, tb/tf, wp, wv, mp, mv,
            nb == nt ? "BITWISE" : @sprintf("%d/%d bitwise", nb, nt))
end

println("Baseline `interpolate` = 1.00x. Position ULP budget vs isce3 is 1 (tested).")
@printf("%-18s %8s | %6s %8s | %10s %10s | %s\n",
        "variant", "speed", "pos ULP", "vel ULP", "pos (m)", "vel (m/s)", "vs current")
report("recip multiply", t -> recip_interp(RC, t))
report("degree-7 poly",  t -> poly_interp(PC, t))

println("\n== where the time actually goes in `interpolate` ==")
# Count the divisions: sepsum 12, h 12, hdot 24 -> 48 divisions per call, of which
# only the 12 in `sepsum` are cacheable without changing bits.
println("  sepsum: 12 divisions (cacheable exactly -- denominators are node separations)")
println("  h:      12 divisions (NOT cacheable -- numerator involves tt)")
println("  hdot:   24 divisions, of which 4 are pure separations (cacheable exactly)")
println("  => 16 of 48 divisions are exactly cacheable; the other 32 involve tt")
