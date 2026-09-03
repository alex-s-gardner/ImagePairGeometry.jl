# Ground point to radar coordinates: the zero-Doppler azimuth time and slant range.
#
# A fixed-point iteration on the zero-Doppler condition
#
#     f(t) = (target - sat(t)) · v(t) = 0
#
# which places the target in the plane through the satellite perpendicular to its velocity. The step
# is Newton-shaped but the derivative is approximated as `-v · v`, dropping the acceleration term
# `(target - sat) · a`. That term is not negligible on an orbital arc, so convergence is *linear*
# rather than quadratic: measured at a factor of about 11 per iteration, taking roughly 15 steps to
# reach machine precision from a scene-center start.
#
# Transcribed from `geogridRadar.cpp:941-972`, including two behaviors that a from-scratch solver
# would not have: a fixed iteration count with no convergence test, and a slant range that lags the
# converged time by one iteration. Both are documented at their sites below.

"""
    GEO2RDR_ITERATIONS

The reference's fixed iteration count for the zero-Doppler solve: 51 (`geogridRadar.cpp:949`).

There is no convergence test and no residual threshold — the loop runs 51 times regardless.

The count is not the overkill it looks like. Because the derivative drops the acceleration term the
iteration converges linearly, at a factor of about 11 per step, so from a scene-center start roughly
60 s from the answer it needs about 15 iterations to reach machine precision. Past that it does not
settle but oscillates at the 1e-12 s level, since the step at a fixed point need not round to zero.

So the count is reproduced rather than replaced with a threshold: lowering it changes the answer, and
raising it changes the last bits.
"""
const GEO2RDR_ITERATIONS = 51

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

Convergence is not checked, per [`GEO2RDR_ITERATIONS`](@ref). A target the orbit cannot image — off
the end of the state vectors — surfaces as an [`OrbitDomainError`](@ref) from the interpolator
rather than as a silent non-convergence.
"""
function geo2rdr(orbit::Orbit, target::SVector{3,Float64},
                 aztime0::Float64, orbittime0::Float64,
                 pos0::SVector{3,Float64}, vel0::SVector{3,Float64})
    satx = pos0
    satv = vel0
    aztime = aztime0
    orbittime = orbittime0

    # Declared outside the loop because they are read after it, in the state the final iteration
    # left them — one step behind the times. See `RadarPoint`.
    look = target - satx
    rngpix = norm3(look)

    for _ in 1:GEO2RDR_ITERATIONS
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
