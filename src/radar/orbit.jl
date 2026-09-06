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
    ChebyshevCoefficients

The orbit interpolant tabulated per bracket as a Chebyshev series, for [`chebyshev_orbit`](@ref).

# Fields
- `position`, `velocity`: eight coefficients per bracket, one entry per bracket of the orbit.
- `tmid`: the time each bracket's series is centered on.
- `half`: half-width of a bracket in seconds, `1.5 * spacing`, which maps a time into `[-1, 1]`.

A bracket's interpolant is a degree-7 polynomial — four state vectors constraining position and
velocity each — so eight coefficients represent it exactly rather than approximately. What is not exact
is recovering them: see [`chebyshev_orbit`](@ref) for the measured cost.

Built once per orbit. Adjacent grid points fall in the same bracket, so the table is read rather than
rebuilt on every one of the seventeen interpolations a grid point costs.
"""
struct ChebyshevCoefficients
    position::Vector{NTuple{8,SVector{3,Float64}}}
    velocity::Vector{NTuple{8,SVector{3,Float64}}}
    tmid::Vector{Float64}
    half::Float64
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
struct Orbit{C<:Union{Nothing,ChebyshevCoefficients}}
    t0::Float64
    spacing::Float64
    position::Vector{SVector{3,Float64}}
    velocity::Vector{SVector{3,Float64}}
    sepsum::NTuple{4,Float64}
    chebyshev::C

    function Orbit{C}(t0, spacing, position, velocity, chebyshev) where {C}
        length(position) == length(velocity) || throw(DimensionMismatch(
            "Orbit has $(length(position)) positions and $(length(velocity)) velocities; " *
            "each state vector needs both"))
        length(position) >= 4 || throw(ArgumentError(
            "Orbit needs at least 4 state vectors to form the Hermite interpolant, got " *
            "$(length(position))"))
        isfinite(spacing) && !iszero(spacing) || throw(ArgumentError(
            "Orbit state vector spacing must be finite and nonzero, got $spacing s"))
        isfinite(t0) || throw(ArgumentError("Orbit start time must be finite, got $t0 s"))
        return new{C}(t0, spacing, position, velocity, _sepsum(spacing), chebyshev)
    end
end

Orbit(t0::Float64, spacing::Float64, position::Vector{SVector{3,Float64}},
      velocity::Vector{SVector{3,Float64}}) =
    Orbit{Nothing}(t0, spacing, position, velocity, nothing)

# Row sums of the reciprocal node separations, `sum(1/(tn[i] - tn[j]) for j != i)`, which
# [`interpolate`](@ref) needs for both the position and velocity weights.
#
# A function of the spacing alone, so it is a constant of the whole orbit rather than of the bracket:
# the nodes are `t0 + (idx + i - 2) * spacing`, so `tn[i] - tn[j]` is `(i - j) * spacing` — bitwise,
# since both are one multiply-add and the subtraction of two such values on a uniform axis is exact.
# That is what makes storing this equivalent to recomputing it: the sums are identical across every
# bracket and at any epoch offset, so a query reads them instead of rebuilding them 17 times per grid
# point. Summed over `j` in increasing order, skipping `i`, which is the order the weights need.
_sepsum(spacing::Float64) = ntuple(4) do i
    s = 0.0
    for j in 1:4
        j == i && continue
        s += 1.0 / ((i - j) * spacing)
    end
    return s
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

# The Chebyshev form, for an orbit built by `chebyshev_orbit`. A separate method rather than a branch
# so the default path compiles to exactly what it did before the option existed.
function interpolate(o::Orbit{<:ChebyshevCoefficients}, t::Real)
    tt = Float64(t)
    t_start = starttime(o)
    t_stop = stoptime(o)
    (tt >= t_start && tt <= t_stop) || throw(OrbitDomainError(tt, t_start, t_stop))

    c = o.chebyshev
    idx = _hermite_index(o, tt)
    # Into `[-1, 1]` across the bracket, which is what the Chebyshev recurrence is defined on.
    x = (tt - c.tmid[idx]) / c.half
    return (_clenshaw(c.position[idx], x), _clenshaw(c.velocity[idx], x))
end

# Clenshaw summation of `sum(a[j] * T[j-1](x))`, evaluated by the Chebyshev three-term recurrence
# `T[k+1] = 2x*T[k] - T[k-1]` rather than by forming any `T[k]`.
#
# Recurses over the tuple rather than over an index, so it terminates on the type and unrolls to
# straight-line arithmetic — the shape `FastGeoProjections.sin2_series` uses. The base case is the last
# coefficient rather than a pair of zeros: starting from zeros leaves a multiply by zero and a
# subtraction of it in the unrolled result, which the fused multiply-adds the rest of the sum relies on
# do not fold away.
@inline _clen(ar, c::Tuple{Any}) = (first(c), zero(first(c)))
@inline function _clen(ar, c::Tuple)
    y1, y2 = _clen(ar, Base.tail(c))
    return (first(c) + ar * y1 - y2, y1)
end

@inline function _clenshaw(a::NTuple{N,SVector{3,Float64}}, x::Float64) where {N}
    y1, y2 = _clen(2x, Base.tail(a))
    return first(a) + x * y1 - y2
end

"""
    chebyshev_orbit(o::Orbit) -> Orbit

`o` with its interpolant tabulated per bracket as a Chebyshev series, evaluated by Clenshaw summation.

Opt-in, and **not** bitwise: this trades the position agreement [`interpolate`](@ref) documents for
throughput. The default `Orbit` is unchanged, as [`SceneCenterStart`](@ref) is the default start policy
for the same reason. Pass the result wherever an `Orbit` is taken — `interpolate` dispatches on it, so
neither solve changes.

# What it buys

Each bracket's interpolant is a degree-7 polynomial, so a cached series replaces roughly a hundred
flops of weight construction — the node times, the reciprocal separations, the Lagrange basis and its
derivative — with eight fused multiply-adds. Measured on the benchmark acquisition:

| | one call | inside `geo2rdr` | a whole radar point |
|---|---|---|---|
| Hermite weights (default) | 39.1 ns | 1008 ns | 2708 ns |
| Chebyshev + Clenshaw | 13.8 ns | 616 ns | 2316 ns |
| | 2.84× | 1.64× | **1.17×** |

The win shrinks at each level because `geo2rdr` is 37% of a point, which caps any interpolator change
near 1.6× however fast interpolation becomes.

# What it costs

Over 20001 times spanning the whole orbit domain, against the Hermite form:

| | worst difference | bitwise |
|---|---|---|
| position | 1.2e-8 m | 93 / 20001 |
| velocity | 1.4e-9 m/s | 0 / 20001 |

The error is the coefficient recovery, not the summation — Clenshaw is backward stable, but recovering
eight coefficients at Earth-radius magnitudes carries ~1e-15 relative, which is ~1e-8 m. A monomial
basis by Horner recovers position twice as accurately (6.5e-9 m, 1345/20001 bitwise) and is slower
(1.27× inside `geo2rdr`), so the faster form is also the less accurate one.

Propagated through the zero-Doppler solve this reaches an output as 1.3e-9 azimuth lines and 8.0e-10
range samples — about 1e-9 of a pixel, far below a rounding boundary — so no integer band moves. The
float off2vel bands inherit it directly and stay well inside the 2e-4 the radar fixture asserts against
isce3.

The reason this is opt-in rather than the default is the standard rather than the magnitude: the bitwise
position agreement is what `REFERENCE.md`'s exactness ladder and `test/blocks.jl`'s blocking assertions
rest on, so giving it up globally would leave every future radar change to be debugged against a
tolerance. See `REFERENCE.md`.

```julia
orbit = chebyshev_orbit(Orbit(; time = t, position = pos, velocity = vel))
coord = RadarCoordinate(; orbit, starting_range, dr, sensing_start, prf,
                        nsamples, nlines, look_side, wavelength, incidence_angle)
```
"""
function chebyshev_orbit(o::Orbit)
    n = length(o)
    nwin = n - 3
    # A bracket spans four nodes, so its midpoint sits between the second and third and its half-width
    # is one and a half spacings. `abs`, since a negative spacing would otherwise reflect the mapping.
    half = 1.5 * abs(o.spacing)
    tmid = Vector{Float64}(undef, nwin)
    pos = Vector{NTuple{8,SVector{3,Float64}}}(undef, nwin)
    vel = Vector{NTuple{8,SVector{3,Float64}}}(undef, nwin)

    # Chebyshev points of the first kind, and the coefficients by the discrete cosine transform they
    # diagonalize. Sampled from the Hermite form itself, so the series represents the interpolant this
    # package evaluates rather than a fresh fit to the state vectors.
    xs = ntuple(k -> cos(pi * (k - 0.5) / 8), 8)
    for idx in 1:nwin
        tm = statetime(o, idx) + half
        tmid[idx] = tm
        samples = ntuple(k -> interpolate(o, tm + xs[k] * half), 8)
        pos[idx] = _cheb_coefficients(ntuple(k -> samples[k][1], 8))
        vel[idx] = _cheb_coefficients(ntuple(k -> samples[k][2], 8))
    end
    return Orbit{ChebyshevCoefficients}(o.t0, o.spacing, o.position, o.velocity,
                                        ChebyshevCoefficients(pos, vel, tmid, half))
end

# Chebyshev coefficients of a function sampled at the eight points of the first kind.
@inline _cheb_coefficients(f::NTuple{8,SVector{3,Float64}}) = ntuple(8) do j
    s = SVector{3,Float64}(0.0, 0.0, 0.0)
    for k in 1:8
        s += f[k] * cos(pi * (j - 1) * (k - 0.5) / 8)
    end
    return (j == 1 ? 1.0 : 2.0) * s / 8
end

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

An orbit carrying a Chebyshev table — [`chebyshev_orbit`](@ref) — is evaluated by that table instead,
which is faster and gives up the bitwise position agreement above.
"""
function interpolate(o::Orbit{Nothing}, t::Real)
    tt = Float64(t)
    t_start = starttime(o)
    t_stop = stoptime(o)
    (tt >= t_start && tt <= t_stop) || throw(OrbitDomainError(tt, t_start, t_stop))

    idx = _hermite_index(o, tt)

    # The four node times, evaluated once. `statetime` is `t0 + (i - 1) * spacing`, so recomputing it
    # inside the weight loops below — around 150 times per call — costs a multiply-add each and cannot
    # give a different answer.
    tn = ntuple(i -> statetime(o, idx + i - 1), 4)

    # Row sums of the reciprocal node separations, which `f0` and `g0` both need. Read off the orbit
    # rather than formed here: they depend only on the spacing, so they are the same on every bracket.
    # See `_sepsum`.
    sepsum = o.sepsum

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
