include(joinpath(@__DIR__, "radar_setup.jl"))
using Printf, BenchmarkTools, LinearAlgebra
using ImagePairGeometry: statetime, _hermite_index, norm3, starttime, stoptime

# Per-interval degree-7 Hermite polynomial in s = t - tn1, coefficients solved once.
struct PolyOrbit
    t0::Float64
    spacing::Float64
    n::Int
    # cp[idx][k] is the coefficient of s^(k-1) for position; cv likewise for velocity.
    cp::Vector{NTuple{8,SVector{3,Float64}}}
    cv::Vector{NTuple{7,SVector{3,Float64}}}
end

function PolyOrbit(o::Orbit)
    n = length(o)
    nwin = max(1, n - 3)
    cp = Vector{NTuple{8,SVector{3,Float64}}}(undef, nwin)
    cv = Vector{NTuple{7,SVector{3,Float64}}}(undef, nwin)
    for idx in 1:nwin
        t1 = statetime(o, idx)
        # 8 constraints: position and velocity at each of 4 nodes.
        A = zeros(8, 8)
        for i in 1:4
            s = statetime(o, idx + i - 1) - t1
            for k in 0:7
                A[i, k+1] = s^k
                A[4+i, k+1] = k == 0 ? 0.0 : k * s^(k-1)
            end
        end
        F = lu(A)
        coefs = ntuple(8) do k
            SVector{3,Float64}(0.0, 0.0, 0.0)
        end
        cols = map(1:3) do d
            rhs = [ (i <= 4 ? o.position[idx+i-1][d] : o.velocity[idx+i-5][d]) for i in 1:8 ]
            F \ rhs
        end
        cp[idx] = ntuple(k -> SVector{3,Float64}(cols[1][k], cols[2][k], cols[3][k]), 8)
        cv[idx] = ntuple(k -> k * cp[idx][k+1], 7)
    end
    return PolyOrbit(o.t0, o.spacing, n, cp, cv)
end

@inline function poly_interp(p::PolyOrbit, t::Real)
    tt = Float64(t)
    search = Int(trunc((tt - p.t0) / p.spacing)) + 1
    idx = clamp(search - 2, 0, p.n - 4) + 1
    s = tt - (p.t0 + (idx - 1) * p.spacing)
    c = p.cp[idx]
    pos = c[8]
    for k in 7:-1:1; pos = pos * s + c[k]; end
    d = p.cv[idx]
    vel = d[7]
    for k in 6:-1:1; vel = vel * s + d[k]; end
    return (pos, vel)
end

const PO = PolyOrbit(ORB)

println("== accuracy of the cached polynomial vs the Hermite weights ==")
let maxdp = 0.0, maxdv = 0.0
    for t in range(starttime(ORB), stoptime(ORB); length = 20001)
        a, av = interpolate(ORB, t)
        b, bv = poly_interp(PO, t)
        maxdp = max(maxdp, norm3(a - b))
        maxdv = max(maxdv, norm3(av - bv))
    end
    @printf("max position difference: %.3e m\n", maxdp)
    @printf("max velocity difference: %.3e m/s\n", maxdv)
end

println("\n== speed ==")
b1 = @benchmark interpolate($ORB, 305.137)
b2 = @benchmark poly_interp($PO, 305.137)
@printf("Hermite weights: %8.4f us\n", minimum(b1).time/1e3)
@printf("cached poly:     %8.4f us  (%.1fx)\n", minimum(b2).time/1e3,
        minimum(b1).time/minimum(b2).time)

println("\n== reciprocal-separation cache only (multiply instead of divide) ==")
# Bit-inexact but tiny: precompute 1/(tn[i]-tn[j]) per interval.
struct RecipOrbit
    o::Orbit
    r::Vector{NTuple{4,NTuple{4,Float64}}}
    sepsum::Vector{NTuple{4,Float64}}
end
function RecipOrbit(o::Orbit)
    nwin = max(1, length(o) - 3)
    r = Vector{NTuple{4,NTuple{4,Float64}}}(undef, nwin)
    ss = Vector{NTuple{4,Float64}}(undef, nwin)
    for idx in 1:nwin
        tn = ntuple(i -> statetime(o, idx + i - 1), 4)
        r[idx] = ntuple(i -> ntuple(j -> i == j ? 0.0 : 1.0/(tn[i]-tn[j]), 4), 4)
        ss[idx] = ntuple(i -> sum(j -> j == i ? 0.0 : r[idx][i][j], 1:4), 4)
    end
    return RecipOrbit(o, r, ss)
end
@inline function recip_interp(ro::RecipOrbit, t::Real)
    o = ro.o; tt = Float64(t)
    idx = _hermite_index(o, tt)
    tn = ntuple(i -> statetime(o, idx + i - 1), 4)
    r = ro.r[idx]; sepsum = ro.sepsum[idx]
    f1 = ntuple(i -> tt - tn[i], 4)
    f0 = ntuple(i -> 1.0 - 2.0*sepsum[i]*f1[i], 4)
    h = ntuple(4) do i
        p = 1.0
        for j in 1:4; j == i && continue; p *= f1[j] * r[i][j]; end
        p
    end
    pos = SVector{3,Float64}(0.,0.,0.)
    for i in 1:4
        pos += h[i]*h[i]*(o.position[idx+i-1]*f0[i] + o.velocity[idx+i-1]*f1[i])
    end
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
    for i in 1:4
        vel += h[i]*(o.position[idx+i-1]*g0[i] + o.velocity[idx+i-1]*g1[i])
    end
    return (pos, vel)
end
const RO = RecipOrbit(ORB)
let mdp = 0.0, mdv = 0.0
    for t in range(starttime(ORB), stoptime(ORB); length = 20001)
        a, av = interpolate(ORB, t); b, bv = recip_interp(RO, t)
        mdp = max(mdp, norm3(a-b)); mdv = max(mdv, norm3(av-bv))
    end
    @printf("max position difference: %.3e m,  velocity: %.3e m/s\n", mdp, mdv)
end
b3 = @benchmark recip_interp($RO, 305.137)
@printf("recip cache:     %8.4f us  (%.1fx)\n", minimum(b3).time/1e3,
        minimum(b1).time/minimum(b3).time)

println("\n== position-only interpolation ==")
# geo2rdr needs both; but `_to_grid`/`nadir_sphere` paths differ. Measure the split.
@inline function pos_only(o::Orbit, t::Real)
    tt = Float64(t); idx = _hermite_index(o, tt)
    tn = ntuple(i -> statetime(o, idx + i - 1), 4)
    sepsum = ntuple(4) do i
        s = 0.0; for j in 1:4; j == i && continue; s += 1.0/(tn[i]-tn[j]); end; s
    end
    f1 = ntuple(i -> tt - tn[i], 4)
    f0 = ntuple(i -> 1.0 - 2.0*sepsum[i]*f1[i], 4)
    h = ntuple(4) do i
        p = 1.0; for j in 1:4; j == i && continue; p *= (tt-tn[j])/(tn[i]-tn[j]); end; p
    end
    pos = SVector{3,Float64}(0.,0.,0.)
    for i in 1:4
        pos += h[i]*h[i]*(o.position[idx+i-1]*f0[i] + o.velocity[idx+i-1]*f1[i])
    end
    pos
end
b4 = @benchmark pos_only($ORB, 305.137)
@printf("position only:   %8.4f us  (%.1fx)\n", minimum(b4).time/1e3,
        minimum(b1).time/minimum(b4).time)
