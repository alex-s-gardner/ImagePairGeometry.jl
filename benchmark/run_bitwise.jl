# Which cached quantities preserve `interpolate`'s bitwise position agreement?
# The constraint: `radar_numerics.jl:392` asserts position within 1 ULP of isce3, bitwise on 10/11
# cases. Caching a value computed in the identical order is exact; replacing a division by a
# multiplication is not.
include(joinpath(@__DIR__, "radar_setup.jl"))
using Printf, BenchmarkTools
using ImagePairGeometry: statetime, _hermite_index, norm3, starttime, stoptime

# Cache only what is bit-for-bit reproducible: the node times, the row sums `sepsum`, and the
# reciprocal separations `hdot` initializes `prod` with. Every division that involves `tt` stays.
struct ExactCache
    o::Orbit
    tn::Vector{NTuple{4,Float64}}
    sepsum::Vector{NTuple{4,Float64}}
    recip::Vector{NTuple{4,NTuple{4,Float64}}}
end
function ExactCache(o::Orbit)
    nwin = max(1, length(o) - 3)
    tn = Vector{NTuple{4,Float64}}(undef, nwin)
    ss = Vector{NTuple{4,Float64}}(undef, nwin)
    rc = Vector{NTuple{4,NTuple{4,Float64}}}(undef, nwin)
    for idx in 1:nwin
        t = ntuple(i -> statetime(o, idx + i - 1), 4)
        tn[idx] = t
        # Identical order to the loop in `interpolate`, so the sum is bit-for-bit the same.
        ss[idx] = ntuple(4) do i
            s = 0.0
            for j in 1:4
                j == i && continue
                s += 1.0 / (t[i] - t[j])
            end
            return s
        end
        rc[idx] = ntuple(i -> ntuple(j -> i == j ? 0.0 : 1.0/(t[i]-t[j]), 4), 4)
    end
    return ExactCache(o, tn, ss, rc)
end

@inline function exact_interp(c::ExactCache, t::Real)
    o = c.o; tt = Float64(t)
    (tt >= starttime(o) && tt <= stoptime(o)) || error("out of domain")
    idx = _hermite_index(o, tt)
    tn = c.tn[idx]; sepsum = c.sepsum[idx]; recip = c.recip[idx]
    f1 = ntuple(i -> tt - tn[i], 4)
    f0 = ntuple(i -> 1.0 - 2.0 * sepsum[i] * f1[i], 4)
    h = ntuple(4) do i
        p = 1.0
        for j in 1:4
            j == i && continue
            p *= (tt - tn[j]) / (tn[i] - tn[j])
        end
        return p
    end
    pos = SVector{3,Float64}(0.0, 0.0, 0.0)
    for i in 1:4
        pos += h[i]*h[i]*(o.position[idx+i-1]*f0[i] + o.velocity[idx+i-1]*f1[i])
    end
    hdot = ntuple(4) do i
        acc = 0.0
        for j in 1:4
            j == i && continue
            prod = recip[i][j]                      # cached, identical bits
            for k in 1:4
                (k == i || k == j) && continue
                prod *= (tt - tn[k]) / (tn[i] - tn[k])
            end
            acc += prod
        end
        return acc
    end
    g1 = ntuple(i -> h[i] + 2.0*hdot[i]*f1[i], 4)
    g0 = ntuple(i -> 2.0*(f0[i]*hdot[i] - sepsum[i]*h[i]), 4)
    vel = SVector{3,Float64}(0.0, 0.0, 0.0)
    for i in 1:4
        vel += h[i]*(o.position[idx+i-1]*g0[i] + o.velocity[idx+i-1]*g1[i])
    end
    return (pos, vel)
end

const EC = ExactCache(ORB)

println("== is the exact-cache variant bitwise against the current `interpolate`? ==")
nbit_p = 0; nbit_v = 0; ntot = 0; worst_p = 0; worst_v = 0
for t in range(starttime(ORB), stoptime(ORB); length = 50_001)
    a, av = interpolate(ORB, t)
    b, bv = exact_interp(EC, t)
    global ntot += 1
    a === b && (global nbit_p += 1)
    av === bv && (global nbit_v += 1)
    for k in 1:3
        global worst_p = max(worst_p, abs(reinterpret(Int64, a[k]) - reinterpret(Int64, b[k])))
        global worst_v = max(worst_v, abs(reinterpret(Int64, av[k]) - reinterpret(Int64, bv[k])))
    end
end
@printf("position: %d/%d bitwise, worst %d ULP\n", nbit_p, ntot, worst_p)
@printf("velocity: %d/%d bitwise, worst %d ULP\n", nbit_v, ntot, worst_v)

b1 = @benchmark interpolate($ORB, 305.137)
b2 = @benchmark exact_interp($EC, 305.137)
@printf("\ncurrent:     %8.4f us\n", minimum(b1).time/1e3)
@printf("exact cache: %8.4f us  (%.2fx)\n", minimum(b2).time/1e3,
        minimum(b1).time/minimum(b2).time)
