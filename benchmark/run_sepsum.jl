# The gate on storing the Hermite separation sums on the `Orbit`: bitwise agreement against forming
# them per call, and what the change is worth on one interpolation and on a whole radar point.
#
# `interp_rebuilt` below forms the sums from the node-time subtractions, which is the form the stored
# sums replace. The claim it checks is that `tn[i] - tn[j]` is exactly `(i - j) * spacing` on a
# uniform axis, so the sums are an orbit constant rather than a per-bracket one -- across brackets,
# spacings, signs of the spacing, epoch offsets, and the domain edges where the bracket clamp makes
# the interpolant one-sided.
#
# A change touching `interpolate` or `Orbit`'s spacing handling should re-run this rather than trust
# the counts in the commit message.
include(joinpath(@__DIR__, "radar_setup.jl"))
using BenchmarkTools, Printf
using ImagePairGeometry: statetime, _hermite_index, starttime, stoptime, SceneCenterStart, _sepsum

bits(x::Float64) = reinterpret(UInt64, x)
bitv(v::SVector{3,Float64}) = ntuple(i -> bits(v[i]), 3)
ns(b) = minimum(b).time

# `interpolate` as it stood before the sums moved onto the `Orbit`.
function interp_rebuilt(o::Orbit, t::Real)
    tt = Float64(t)
    (tt >= starttime(o) && tt <= stoptime(o)) || throw(ArgumentError("out of domain"))
    idx = _hermite_index(o, tt)
    tn = ntuple(i -> statetime(o, idx + i - 1), 4)
    sepsum = ntuple(4) do i
        s = 0.0
        for j in 1:4
            j == i && continue
            s += 1.0 / (tn[i] - tn[j])
        end
        return s
    end
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
        pos += h[i] * h[i] * (o.position[idx+i-1] * f0[i] + o.velocity[idx+i-1] * f1[i])
    end
    hdot = ntuple(4) do i
        acc = 0.0
        for j in 1:4
            j == i && continue
            prod = 1.0 / (tn[i] - tn[j])
            for k in 1:4
                (k == i || k == j) && continue
                prod *= (tt - tn[k]) / (tn[i] - tn[k])
            end
            acc += prod
        end
        return acc
    end
    g1 = ntuple(i -> h[i] + 2.0 * hdot[i] * f1[i], 4)
    g0 = ntuple(i -> 2.0 * (f0[i] * hdot[i] - sepsum[i] * h[i]), 4)
    vel = SVector{3,Float64}(0.0, 0.0, 0.0)
    for i in 1:4
        vel += h[i] * (o.position[idx+i-1] * g0[i] + o.velocity[idx+i-1] * g1[i])
    end
    return (pos, vel)
end

println("="^76)
println("BITWISE — stored sums vs rebuilt, across brackets, spacings and epochs")
println("="^76)
let nbad = 0, ntot = 0
    for (t0, sp) in ((0.0, 10.0), (300.0, 10.0), (1.0e7, 10.0), (0.0, 1.0), (0.0, 30.0),
                     (-500.0, 7.5), (1.23e6, 0.5))
        o = let base = bench_orbit()
            n = length(base)
            t = [t0 + (i - 1) * sp for i in 1:n]
            Orbit(; time = t, position = base.position, velocity = base.velocity)
        end
        # Every bracket, and several points across each — including the domain edges, where the
        # bracket clamp makes the interpolant one-sided.
        for k in 0:400
            tt = starttime(o) + k * (stoptime(o) - starttime(o)) / 400
            a = interpolate(o, tt)
            b = interp_rebuilt(o, tt)
            ntot += 1
            (bitv(a[1]) == bitv(b[1]) && bitv(a[2]) == bitv(b[2])) || (nbad += 1)
        end
    end
    @printf("%d of %d samples bitwise identical in position and velocity\n", ntot - nbad, ntot)
    nbad == 0 || @printf("  !! %d DIFFER\n", nbad)
end

println()
println("="^76)
println("SEPSUM — stored value vs the subtraction form it replaces")
println("="^76)
for sp in (10.0, 1.0, 30.0, 7.5, 0.5, -10.0)
    o = let base = bench_orbit()
        t = [(i - 1) * sp for i in 1:length(base)]
        Orbit(0.0, sp, base.position, base.velocity)
    end
    ok = true
    for idx in 1:(length(o) - 3)
        tn = ntuple(i -> statetime(o, idx + i - 1), 4)
        ref = ntuple(4) do i
            s = 0.0
            for j in 1:4
                j == i && continue
                s += 1.0 / (tn[i] - tn[j])
            end
            s
        end
        map(bits, ref) == map(bits, o.sepsum) || (ok = false)
    end
    @printf("spacing %+7.2f s: stored sums bitwise equal on every bracket: %s\n", sp, ok)
end

println()
println("="^76)
println("COST")
println("="^76)
const ORBIT = bench_orbit()
const TQ = 300.0
t_reb = ns(@benchmark interp_rebuilt($ORBIT, $TQ))
t_now = ns(@benchmark interpolate($ORBIT, $TQ))
@printf("interpolate, sums rebuilt  %7.2f ns\n", t_reb)
@printf("interpolate, sums stored   %7.2f ns   (%.3fx)\n", t_now, t_reb/t_now)

const C = bench_case(128)
const FT = fast_transform(32632, 4326)
const SCC = SceneCenterStart(C.coord)
const NRM = surface_normal(0.05, -0.03)
const GX, GY, GZ = -254500.0 + 12000.0, 2191000.0 - 12000.0, 500.0
@printf("pointgeometry (fast)       %7.1f ns\n",
        ns(@benchmark pointgeometry($FT, $GX, $GY, $GZ, $(C.coord), $NRM, $SCC, nothing)))

for n in (64, 128)
    c = bench_case(n)
    b = @benchmark pairgeometry($c.grid, $c.pair, $c.inputs; transform = $c.tf,
                                window = $c.win, nodata = nodata_from(0.0)) samples=3 evals=1 seconds=90
    m = minimum(b)
    @printf("whole kernel n=%3d         %7.3f us/pt  %5d allocs\n", n, m.time/1e3/(n*n), m.allocs)
end
