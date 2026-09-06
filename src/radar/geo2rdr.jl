# Ground point to radar coordinates: the zero-Doppler azimuth time and slant range.
#
# A fixed-point iteration on the zero-Doppler condition
#
#     f(t) = (target - sat(t)) · v(t) = 0
#
# which places the target in the plane through the satellite perpendicular to its velocity. The step
# is Newton-shaped but the derivative is approximated as `-v · v`, dropping the acceleration term
# `(target - sat) · a`. That term is not negligible on an orbital arc, so convergence is *linear*
# rather than quadratic, at a rate `geo2rdr_rate` gives in closed form.
#
# Transcribed from `geogridRadar.cpp:941-972`, including one behavior a from-scratch solver would not
# have: a slant range that lags the converged time by one iteration, documented at `RadarPoint`. The
# reference's fixed 51 iterations are *not* reproduced — see `GEO2RDR_ITERATIONS`.

"""
    GEO2RDR_ITERATIONS

Fixed iteration count for the zero-Doppler solve: 16. The reference uses 51
(`geogridRadar.cpp:949`); this is the one place the radar path spends fewer iterations than it, and
the divergence is the largest single speedup on the path — about 1.6× of the whole per-point cost.

There is no convergence test and no residual threshold, matching the reference: the loop runs 16
times regardless. A threshold is the wrong instrument here because the step never reaches zero — past
convergence the iteration oscillates at the ULP of `aztime` rather than settling — so a threshold
below that floor would run to a cap on every point anyway. The floor scales with the epoch: 1e-9
azimuth lines at a `sensing_start` of 300 s, 1e-7 lines on a seconds-since-epoch clock at 1e7 s. Both
are far below a rounding boundary.

# Why 16, and when it would not be enough

The count must exceed the iterations the *worst grid point of the worst scene* needs. That number is
scene dependent, and predictable: see [`geo2rdr_rate`](@ref) for the closed form and
[`geo2rdr_iterations_needed`](@ref) to evaluate it for an acquisition in hand.

Requirement measured across synthetic acquisitions spanning real mission geometry, taking the far end
of the scene as the worst point — the reference starts every solve from scene center, so that point
is half a scene length from the initial guess:

| geometry | rate | iterations to 1e-9 s |
|---|---|---|
| 350 km (below any SAR orbit) | 0.053 | 9 |
| TerraSAR-X, ICEYE, Capella (510–570 km) | 0.076–0.083 | 10 |
| ALOS-2 (628 km) | 0.091 | 11 |
| Sentinel-1 (693 km) | 0.100 | 11–12 |
| NISAR (747 km) | 0.107 | 12 |
| RADARSAT-2 (798 km) | 0.113 | 12 |
| 1400 km (above any SAR orbit) | 0.186 | 15 |

Worst over the realistic space is 12, so 16 carries four iterations of margin — a factor of 1e-3 in
error at the worst realistic rate. Terrain height (sea level to 8000 m), incidence angle (15° to
50°), look side, grid spacing and repeat interval all move the requirement by less than one
iteration. Orbit eccentricity *lowers* it.

The bound this rests on is orbital altitude, and it is the thing to check first if real data ever
disagrees: above roughly 1000 km the requirement climbs past 16 and this constant must rise with it.
`geo2rdr_iterations_needed` answers that for a given orbit without re-deriving anything.
"""
const GEO2RDR_ITERATIONS = 16

"""
    geo2rdr_rate(satpos, target) -> Float64

The per-iteration convergence rate of [`geo2rdr`](@ref) at one geometry: the factor by which each
iteration multiplies the remaining error.

Because `fnprime` drops the acceleration term, the iteration is a fixed point with rate
`r = 1 - f'(t) / D`, which for a near-circular orbit reduces to

    r = 1 - (r_t / R) cos(ψ)

with `R` the satellite's geocentric radius, `r_t` the target's, and `ψ` the geocentric angle between
them. Since `r_t cos(ψ)` is `(satpos · target) / R`, this evaluates as `1 - (satpos · target) / R²`
without recovering either angle. Reproduces the measured rate to four significant digits over
altitudes from 350 to 1400 km and incidence angles from 15° to 50°.

`R` dominates: `r` runs from 0.053 at 350 km to 0.186 at 1400 km, and the requirement on
[`GEO2RDR_ITERATIONS`](@ref) rises with it. The closed form assumes `a = -ω²x`, exact only for a
circular orbit; a measured eccentricity of 0.05 lowers the true rate to 0.058 against a predicted
0.10, so the form is conservative rather than wrong for an eccentric orbit.
"""
@inline function geo2rdr_rate(satpos::SVector{3,Float64}, target::SVector{3,Float64})
    R = norm3(satpos)
    return 1.0 - dot3(satpos, target) / (R * R)
end

"""
    geo2rdr_iterations_needed(satpos, target; guess_error = 60.0, tol = 1e-9) -> Int

Iterations [`geo2rdr`](@ref) needs at one geometry to bring an initial guess `guess_error` seconds
from the answer within `tol` seconds of it.

`log(tol / guess_error) / log(r)` at the rate [`geo2rdr_rate`](@ref) gives, rounded up. Use it to
check that [`GEO2RDR_ITERATIONS`](@ref) covers an acquisition whose orbit sits outside the altitudes
that constant was chosen against — pass the satellite position at scene center, a target at the far
edge of the swath, and half the scene length as `guess_error`.

Returns a requirement, not a verdict: compare it against `GEO2RDR_ITERATIONS` yourself.
"""
function geo2rdr_iterations_needed(satpos::SVector{3,Float64}, target::SVector{3,Float64};
                                   guess_error::Real = 60.0, tol::Real = 1e-9)
    r = geo2rdr_rate(satpos, target)
    0 < r < 1 || throw(ArgumentError(
        "geo2rdr convergence rate is $r, outside (0, 1); the iteration does not contract at this " *
        "geometry, so no iteration count suffices"))
    return ceil(Int, log(Float64(tol) / Float64(guess_error)) / log(r))
end

"""
    RadarPoint

Where a ground point falls in radar coordinates, and the geometry at that point.

# Fields
- `aztime`: azimuth time on the *image* time scale — the one the azimuth index is measured against.
- `orbittime`: the same instant on the orbit's time scale. See [`geo2rdr`](@ref) on why both.
- `range`: slant range in meters, lagging `aztime` by one Newton iteration. See below.
- `look`: satellite-to-target vector in ECEF meters, unnormalized, whose length is `range`.
- `position`, `velocity`: satellite state at `orbittime` after the final iteration.

`range` and `look` are the reference's `rngpix` and `drpos`, and both are one iteration stale: they
are computed at the top of the loop body from the *previous* iteration's satellite position, and the
loop then updates the time and re-interpolates without recomputing them
(`geogridRadar.cpp:951-969`). So the range that produces the range index does not correspond to the
time that produces the azimuth index.

Reproducing the staleness is deliberate. The effect is sub-ULP in practice, since the final
iterations barely move the solution — but `look` also feeds the line-of-sight axis vector and the
scale factors downstream, so a "corrected" range would propagate a difference into bands that are
asserted bitwise.
"""
struct RadarPoint
    aztime::Float64
    orbittime::Float64
    range::Float64
    look::SVector{3,Float64}
    position::SVector{3,Float64}
    velocity::SVector{3,Float64}
end

"""
    geo2rdr(orbit, target, aztime0, orbittime0, pos0, vel0) -> RadarPoint

Azimuth time and slant range at which `orbit` images the ECEF point `target`.

`aztime0` and `orbittime0` are the same starting instant expressed on two time scales, and `pos0`,
`vel0` the satellite state there. Both times receive the same Newton increment each iteration; only
`orbittime0` is fed to the interpolator, and only `aztime0`'s descendant reaches the azimuth index.

Their difference is preserved to rounding rather than to the bit — each subtraction rounds to its own
exponent, so two scales separated by a large epoch offset drift by an ULP of the larger. At 1e5 s of
separation that is 1e-11 s, or 5e-9 of an azimuth line.

Carrying two scales rather than one is the reference's structure (`tline` and `tlined`,
`geogridRadar.cpp:906-911`), and it is not redundant because the two clocks have different origins.
`sensingStart` is seconds since midnight of the acquisition day, so `tline` — and the azimuth index
derived from it (`:972`) — is on that clock. `tlined` comes from an absolute UTC timestamp reduced to
seconds since the orbit's reference epoch (`:432`), which is the clock the interpolator needs. The
caller supplies both and this function preserves their difference.

Convergence is not checked, per [`GEO2RDR_ITERATIONS`](@ref), which is also the default `niter`. A
caller that starts from a better guess than the scene midpoint may pass a smaller count — see
[`WarmStart`](@ref), which is the only thing in this package that does.

A target the orbit cannot image — off the end of the state vectors — surfaces as an
[`OrbitDomainError`](@ref) from the interpolator rather than as a silent non-convergence.
"""
function geo2rdr(orbit::Orbit, target::SVector{3,Float64},
                 aztime0::Float64, orbittime0::Float64,
                 pos0::SVector{3,Float64}, vel0::SVector{3,Float64},
                 niter::Int = GEO2RDR_ITERATIONS)
    satx = pos0
    satv = vel0
    aztime = aztime0
    orbittime = orbittime0

    # Both are read after the loop, in the state its final iteration left them — one step behind the
    # times, since the loop updates the time and re-interpolates without recomputing them. See
    # `RadarPoint`. The initial values are never observed: iteration one overwrites them before any
    # read, and `GEO2RDR_ITERATIONS` is nonzero.
    look = target - satx
    rngpix = norm3(look)

    for _ in 1:niter
        look = target - satx
        rngpix = norm3(look)

        fn = dot3(look, satv)
        # The acceleration term of the true derivative is dropped, as the reference drops it.
        fnprime = -dot3(satv, satv)

        step = fn / fnprime
        aztime -= step
        orbittime -= step

        satx, satv = interpolate(orbit, orbittime)
    end

    return RadarPoint(aztime, orbittime, rngpix, look, satx, satv)
end

"""
    range_index(p::RadarPoint, starting_range, dr) -> Float64

Zero-based range sample index of a radar point, rounded to a whole sample.

Matches `geogridRadar.cpp:971`: `std::round((rngpix - startingRange) / dr)`. Uses [`cround`](@ref) —
ties away from zero — because the reference rounds here in C++.

Returned as `Float64` so the bounds test sees the value before any integer conversion, as on the
projected path. Note that the reference's bounds test at `:1108` compares the *converted* `int`,
unlike the projected path's test on the pre-conversion float; the difference is immaterial here
because the value is already integral.
"""
@inline range_index(p::RadarPoint, starting_range::Float64, dr::Float64) =
    cround((p.range - starting_range) / dr)

"""
    azimuth_index(p::RadarPoint, sensing_start, prf) -> Float64

Zero-based azimuth line index of a radar point, rounded to a whole line.

Matches `geogridRadar.cpp:972`: `std::round((tline - sensingStart) * prf)`, where `tline` is
[`RadarPoint`](@ref)'s `aztime` and `sensing_start` is on the same scale — seconds since midnight of
the acquisition day, as `GeogridRadar.py:329-332` computes it.
"""
@inline azimuth_index(p::RadarPoint, sensing_start::Float64, prf::Float64) =
    cround((p.aztime - sensing_start) * prf)
