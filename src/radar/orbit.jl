# Satellite state vectors, and position and velocity between them.
#
# Deliberately holds no dates. isce3 stores state vector times as a `Linspace` of seconds from a
# reference epoch, and every use of the epoch in the radar path is a scalar offset — so carrying
# `DateTime` here would add date arithmetic that nothing needs and that would have to agree with
# isce3's to the microsecond. The caller reduces its times to seconds once and stays in seconds.
#
# Also holds no file parsing. `geogridRadar.cpp:333-427` scrapes an orbit EOF with string matching
# on `<UTC>` and `<X unit>` tags; that is metadata ingest, and it belongs with the SLC readers rather
# than with the geometry. An `Orbit` is built from state vectors the caller supplies.

"""
    OrbitDomainError

Thrown when an interpolation time falls outside the span of an [`Orbit`](@ref)'s state vectors.

The reference interpolates with `OrbitInterpBorderMode::Error` (`geogridRadar.cpp:434`) and exits
on failure, so extrapolation is not a behavior to reproduce. A time outside the domain means the
orbit does not cover the acquisition — the state vectors are wrong, or the time is.
"""
struct OrbitDomainError <: Exception
    t::Float64
    start::Float64
    stop::Float64
end

function Base.showerror(io::IO, e::OrbitDomainError)
    print(io, "OrbitDomainError: interpolation time ", e.t,
          " s is outside the orbit's span [", e.start, ", ", e.stop,
          "] s; the state vectors do not cover this time")
end

"""
    Orbit(; time, position, velocity)
    Orbit(t0, spacing, position, velocity)

Satellite state vectors on a uniformly spaced time axis.

`position` and `velocity` are ECEF, in meters and meters per second. Times are seconds from
whatever epoch the caller uses; the epoch itself is never stored, so every time passed to
[`interpolate`](@ref) must be on the same scale.

The keyword form takes an explicit `time` vector and derives the spacing from it, rejecting a
non-uniform axis. The positional form takes the first time and the spacing directly.

Uniform spacing is not a simplification: isce3 stores the axis as a `Linspace` and its interpolator
locates the bracketing state vectors by arithmetic on the spacing rather than by search
(`Linspace.icc:77-89`). A non-uniform axis has no representation there, and `getOrbitTime`
(`BuildOrbit.cpp:12-46`) rejects one.

At least four state vectors are required — the Hermite interpolant is built from four
(`InterpolateOrbit.icc:22`).

# Example

```jldoctest
julia> using ImagePairGeometry: Orbit, starttime, stoptime

julia> t = collect(0.0:10.0:40.0);

julia> pos = [[7.0e6, 0.0, 1.0e5 * i] for i in eachindex(t)];

julia> vel = [[0.0, 7.5e3, 1.0e4] for _ in eachindex(t)];

julia> orbit = Orbit(; time = t, position = pos, velocity = vel);

julia> length(orbit), starttime(orbit), stoptime(orbit)
(5, 0.0, 40.0)
```
"""
struct Orbit
    t0::Float64
    spacing::Float64
    position::Vector{SVector{3,Float64}}
    velocity::Vector{SVector{3,Float64}}

    function Orbit(t0::Float64, spacing::Float64, position::Vector{SVector{3,Float64}},
                   velocity::Vector{SVector{3,Float64}})
        length(position) == length(velocity) || throw(DimensionMismatch(
            "Orbit has $(length(position)) positions and $(length(velocity)) velocities; " *
            "each state vector needs both"))
        length(position) >= 4 || throw(ArgumentError(
            "Orbit needs at least 4 state vectors to form the Hermite interpolant, got " *
            "$(length(position))"))
        isfinite(spacing) && !iszero(spacing) || throw(ArgumentError(
            "Orbit state vector spacing must be finite and nonzero, got $spacing s"))
        isfinite(t0) || throw(ArgumentError("Orbit start time must be finite, got $t0 s"))
        return new(t0, spacing, position, velocity)
    end
end

function Orbit(; time, position, velocity)
    n = length(time)
    n >= 2 || throw(ArgumentError(
        "Orbit needs at least 2 times to derive a spacing, got $n"))
    t = collect(Float64, time)
    spacing = t[begin + 1] - t[begin]
    # `getOrbitTime` rejects a non-uniform axis rather than resampling (`BuildOrbit.cpp:26-40`),
    # comparing against a tolerance because its times come from date subtraction. Here the times
    # are already seconds, so the comparison is on the derived axis and the tolerance is relative
    # to the spacing.
    for i in eachindex(t)[begin:(end - 1)]
        expected = t[begin] + (i - firstindex(t)) * spacing
        isapprox(t[i], expected; atol = 1e-9 * abs(spacing)) || throw(ArgumentError(
            "Orbit state vectors must be uniformly spaced in time: state vector $i is at " *
            "$(t[i]) s, expected $expected s for a spacing of $spacing s"))
    end
    pos = [SVector{3,Float64}(p[1], p[2], p[3]) for p in position]
    vel = [SVector{3,Float64}(v[1], v[2], v[3]) for v in velocity]
    return Orbit(t[begin], spacing, pos, vel)
end

Base.length(o::Orbit) = length(o.position)

"""
    statetime(o::Orbit, i) -> Float64

Time of the `i`th state vector, one-based, computed as `t0 + (i - 1) * spacing`.

Computed rather than stored, matching `Linspace::operator[]` (`Linspace.h:56`) — so a time midway
through a long orbit carries the rounding of one multiply-add, not of an accumulated sum.
"""
@inline statetime(o::Orbit, i::Integer) = o.t0 + (i - 1) * o.spacing

"""
    starttime(o::Orbit) -> Float64
    stoptime(o::Orbit) -> Float64

First and last state vector times, in seconds on the caller's scale.
"""
starttime(o::Orbit) = statetime(o, 1)
stoptime(o::Orbit) = statetime(o, length(o))

"""
    interpolate(o::Orbit, t) -> NTuple{2,SVector{3,Float64}}

`(position, velocity)` at time `t`, by cubic Hermite interpolation over the four state vectors
bracketing it.

Throws [`OrbitDomainError`](@ref) for a time outside the orbit's span, matching the reference's
`OrbitInterpBorderMode::Error` (`geogridRadar.cpp:434`).

Transcribed from `interpolateOrbitHermite` (`InterpolateOrbit.icc:11-108`). isce3's Legendre
alternative is not ported: `OrbitInterpMethod::Hermite` is the default (`Orbit.h:198`) and the
reference never changes it.

Hermite over four points, rather than the two a cubic through positions alone would need, because
the interpolant matches position *and* velocity at all four.

Position agrees with isce3 bitwise except within half a spacing of the domain edges, where it is
within 1 ULP. Velocity agrees to about 4e-14 relative — 3e-10 m/s on an orbital speed — because the
velocity weights carry a cancellation the position weights do not: `g0` is `2 * (f0 * hdot - s * h)`,
a difference of two products each larger than their difference, so a contracted evaluation and a
strictly-rounded one diverge there. isce3 is compiled with contraction enabled and Julia rounds each
operation. Confirmed to be that and not a transcription error by an independent implementation of the
same formula, which reproduces this result rather than isce3's. See `REFERENCE.md`.
"""
function interpolate(o::Orbit, t::Real)
    tt = Float64(t)
    t_start = starttime(o)
    t_stop = stoptime(o)
    (tt >= t_start && tt <= t_stop) || throw(OrbitDomainError(tt, t_start, t_stop))

    idx = _hermite_index(o, tt)

    # The four node times, evaluated once. `statetime` is `t0 + (i - 1) * spacing`, so recomputing it
    # inside the weight loops below — around 150 times per call — costs a multiply-add each and cannot
    # give a different answer.
    tn = ntuple(i -> statetime(o, idx + i - 1), 4)

    # Reciprocal node separations, and their row sums. `f0` and `g0` each need the same sum over
    # `j != i`, so it is formed once in the same order both would have used.
    sepsum = ntuple(4) do i
        s = 0.0
        for j in 1:4
            j == i && continue
            s += 1.0 / (tn[i] - tn[j])
        end
        return s
    end

    # Time offsets to the four nodes. `f1` is the offset itself; `f0` corrects the position weight
    # for the derivative constraint at each node.
    f1 = ntuple(i -> tt - tn[i], 4)

    f0 = ntuple(i -> 1.0 - 2.0 * sepsum[i] * f1[i], 4)

    # Lagrange basis over the four nodes.
    #
    # `hdot` below recomputes these same ratios. Hoisting them into a 4×4 table is bit-identical but
    # measured 6% *slower* — twelve stack slots cost more than the divisions they save, and the
    # divisions were already in flight behind the multiply chain.
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
        pos += h[i] * h[i] * (o.position[idx + i - 1] * f0[i] + o.velocity[idx + i - 1] * f1[i])
    end

    # Derivative of the Lagrange basis, then the position and velocity weights it induces.
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
        vel += h[i] * (o.position[idx + i - 1] * g0[i] + o.velocity[idx + i - 1] * g1[i])
    end

    return (pos, vel)
end

# One-based index of the first of the four state vectors forming the interpolant.
#
# isce3 computes `orbit.time().search(t) - 2` and clamps to `[0, size - 4]`
# (`InterpolateOrbit.icc:21-22`), where `search` is
#
#     (val - first) / spacing + 1
#
# truncated toward zero by the assignment to `int`, with early returns of `0` below the first time
# and `size` above the last (`Linspace.icc:77-89`). Those early returns are unreachable here — the
# caller has already rejected an out-of-domain time — but the truncation is not, and it differs
# from `floor` for a negative quotient, which is why this is `trunc` rather than `fld`.
#
# The clamp is what keeps the interpolant well-formed near the ends: within two spacings of either
# edge there are not two state vectors on both sides, and the four chosen become one-sided rather
# than the window running off the array. So the interpolant stays cubic Hermite everywhere and
# degrades to extrapolation-within-the-window at the edges, silently. That is the reference's
# behavior, and it is why the fixture tests both domain edges specifically.
@inline function _hermite_index(o::Orbit, t::Float64)
    n = length(o)
    search = Int(trunc((t - o.t0) / o.spacing)) + 1
    idx0 = clamp(search - 2, 0, n - 4)
    return idx0 + 1
end
