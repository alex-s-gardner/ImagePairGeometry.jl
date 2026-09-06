# Per-point radar geometry: the forward mapping, and the two axis vectors the outputs need.
#
# Reaches the same `PointGeometry` the projected path produces, by a different route. Where that path
# transforms a grid point and steps one pixel along each image axis, this one solves for the azimuth
# time and slant range, then constructs each axis vector by stepping one pixel in *radar* coordinates
# and transforming the result back to grid coordinates:
#
#   range axis   — step `dr` along the ECEF look vector, convert to lon/lat, then to grid coordinates,
#                  and difference against the grid point (`geogridRadar.cpp:975-996`).
#   azimuth axis — advance one pulse interval, re-solve the range-Doppler equations at the new
#                  satellite position, and difference the resulting ground point the same way
#                  (`:1000-1085`).
#
# So both vectors end up in grid coordinates with the same meaning as the projected path's, which is
# what lets `outputs.jl` serve both. What does not carry over is a single nominal spacing: see
# `RadarSpacing`.

"""
    RANGE_DOPPLER_ITERATIONS

Fixed iteration count for the range-Doppler solve: 6. The reference uses 10
(`geogridRadar.cpp:1038`).

As with [`GEO2RDR_ITERATIONS`](@ref) there is no convergence test, and here the residual `rdiff` is
computed (`:1070`) and never read. Contrast isce3's `rdr2geo`, which breaks on `dr < 1e-8` — this is a
separate transcription for that reason, not a call into [`rdr2geo`](@ref).

# Why 6

This is a different fixed point from `geo2rdr`'s and converges by a different mechanism. The iterate
is `zsch`, the target's height above the sphere osculating the ellipsoid beneath the satellite.
Perturbing it moves the target along the range circle, which changes its latitude, which changes the
geocentric radius of the constant-geodetic-height surface. So the rate is governed by
`dr_t/dlat = -a f sin(2 lat)` — it vanishes at the equator and at the pole and peaks near 45°, and it
is *small*: about 6e-4, against `geo2rdr`'s 0.09. Three iterations gain nine digits.

Measured over 1320 geometries — altitudes 400 to 1000 km, latitudes 0 to 80° on a near-polar orbit,
incidence 15 to 55°, terrain height 0 and 4000 m, both look sides — the worst target displacement
from the 30-iteration answer:

| count | worst displacement | as range samples |
|---|---|---|
| 3 | 1.8e-2 m | 7.6e-3 |
| 4 | 7.7e-5 m | 3.3e-5 |
| 5 | 3.3e-7 m | 1.4e-7 |
| **6** | **7.8e-9 m** | **3.4e-9** |
| 8 | 8.3e-9 m | 3.6e-9 |
| 10 | 8.6e-9 m | 3.7e-9 |

6 is where the iteration reaches its floor: about 1e-8 m, one ULP of `norm3` at Earth radius carried
through the lon/lat round trip. Counts above 6 do not refine the answer — they land on different
noise, which is why 7, 8 and 10 are no closer to the fixed point than 6 is.

The requirement is 4, set by the range-sphere property `test/radar_geometry.jl` asserts at
`rtol = 1e-9`: over the same 1320 geometries a count of 3 violates it on 220 of them, and 4 clears it
with a worst residual of 1.9e-11. So 6 carries two iterations of margin over the requirement while
sitting on the floor.

Latitude is the governing parameter, not altitude as it is for `GEO2RDR_ITERATIONS` — and the
worst-case latitudes are the mid ones, so a polar ITS_LIVE grid and the equatorial fixture are both
easier than the 30–45° band. Terrain height changes the requirement by less than one iteration from
the Dead Sea to Everest, and altitude by less than one over the whole 400–1000 km range.
"""
const RANGE_DOPPLER_ITERATIONS = 6

"""
    RadarSpacing

The nominal spacings the radar outputs divide by, which are **not** one pair.

# Fields
- `operator`: `(dr, |da|)`, for [`offset_to_velocity`](@ref) and the azimuth [`scale_factors`](@ref)
  denominator. `da` is the along-track step in **ECEF** meters (`geogridRadar.cpp:1159`).
- `search`: `(dr, |alt|)`, for [`search_pixels`](@ref). `alt` is the same step in **grid**
  coordinates.

`alt` and `da` are one displacement expressed in two frames, so their norms differ — by the local map
scale, around 0.04% in UTM. The reference divides by `norm(da)` at `:1166-1176` and `:1191` and by
`norm(alt)` at `:1207-1216`, and that difference is reproduced rather than reconciled: it changes the
last bits of the operator relative to the search extent.

The projected path has no such split — one spacing serves every output there — which is why the
kernel takes spacings as values rather than reading them off a coordinate.
"""
struct RadarSpacing
    operator::NTuple{2,Float64}
    search::NTuple{2,Float64}
end

"""
    ZeroDopplerStart

Where [`pointgeometry`](@ref) starts the zero-Doppler solve, and how many iterations it runs.

`SceneCenterStart()` is the reference's behavior and the default: every point starts from the scene
midpoint, `GEO2RDR_ITERATIONS` iterations. [`WarmStart`](@ref) starts each point from the previous
point's answer instead, which converges in fewer iterations because adjacent grid points are a small
fraction of an azimuth line apart.
"""
abstract type ZeroDopplerStart end

"""
    SceneCenterStart()
    SceneCenterStart(c::RadarCoordinate)

Start every zero-Doppler solve from the scene midpoint, as the reference does
(`geogridRadar.cpp:906-911`). See [`ZeroDopplerStart`](@ref).

The one-argument form precomputes the satellite state at that midpoint. It is a scene constant, so
interpolating it per grid point — as a literal transcription does — repeats one orbit interpolation
across every point of the window for a value that cannot change. The per-point loop builds this form
once; the no-argument form interpolates on demand, so a direct [`pointgeometry`](@ref) call needs no
setup. Both give the same bits.
"""
struct SceneCenterStart{S} <: ZeroDopplerStart
    state::S
end

SceneCenterStart() = SceneCenterStart(nothing)

function SceneCenterStart(c::RadarCoordinate)
    pm, vm = interpolate(c.orbit, orbit_midtime(c))
    return SceneCenterStart((midtime(c), orbit_midtime(c), pm, vm))
end

"""
    WARM_START_MIN_ITERATIONS

Fewest iterations a [`WarmStart`](@ref) accepts: 8.

Set by the *float* bands, not the integer ones. No rounded index moves at any count from 5 upward, so
an integer-only check would permit 5 — but the off2vel bands, which divide by the along-track step and
so inherit the azimuth time's error directly, miss the isce3 reference by 7.3e-4 relative at 5 and
1.6e-4 at 6, against the 2e-4 `test/radar_geogrid.jl` asserts. At 8 the agreement is 1.07e-4, which is
the cold-start value: the warm start has stopped contributing any error of its own.
"""
const WARM_START_MIN_ITERATIONS = 8

"""
    WarmStart(; iterations = 8)

Start each zero-Doppler solve from the previous grid point's answer, running `iterations` iterations
instead of [`GEO2RDR_ITERATIONS`](@ref).

Adjacent grid points at ITS_LIVE spacings are a few hundredths of an azimuth line apart, so the
previous point's solution is a far better initial guess than the scene midpoint and the solve needs
roughly half as many iterations. Worth about 1.2× of the per-point radar cost.

# This trades away bit-identical blocking and threading

The result becomes a function of the order points are visited in, so a window computed in one piece
and the same window computed as blocks no longer agree bitwise — the first point of each block has no
predecessor and cold-starts, and every point after it inherits a different starting guess. That
property is what [`pairgeometry_blocked`](@ref) rests on and what `test/blocks.jl` asserts across
block sizes from `(1, 1)` upward, which is why this is opt-in and `SceneCenterStart` remains the
default.

Per-block seeding does not rescue it: a `(1, 1)` block is a cold start by construction, so no seeding
rule makes a blocked run agree with an unblocked one.

# How far the answer moves

Measured over a 64×64 block walked in the loop's own order, against the cold `GEO2RDR_ITERATIONS`
result, on the fixture's acquisition:

| `iterations` | indices moved | float bands vs isce3 | fixture gate (bound 2e-4) |
|---|---|---|---|
| 5 | 0 | 7.3e-4 | **fails** |
| 6 | 0 | 1.6e-4 | passes, little room |
| **8** | **0** | **1.07e-4** | **passes — the cold-start value** |
| 10 | 0 | 1.07e-4 | passes |

No integer band moves at any of these counts, on any of the eight fixture cases. What sets the floor
is the off2vel float bands: they divide by the along-track step, so they inherit the azimuth time's
error without a rounding step to absorb it. At 8 they recover the cold-start agreement exactly, which
is why that is both the default and the minimum — see [`WARM_START_MIN_ITERATIONS`](@ref).

# Grid spacing barely matters

The count needed depends on how far apart successive points are, which is the grid spacing — but the
dependence is far weaker than it looks, and 8 covers the whole practical range. At 10 km spacing the
gap between successive points is 645 azimuth lines, or 1.3 s of azimuth time, against the roughly 60 s
a cold start begins from: still more than an order of magnitude closer, which at the measured rate is
about four iterations' worth. Measured float-band divergence from the cold result, which is flat:

| grid spacing | `iterations = 8` | `iterations = 10` |
|---|---|---|
| 120 m (ITS_LIVE) | 2.8e-7 | 1.2e-8 |
| 500 m | 4.9e-7 | 9.3e-9 |
| 1 km | 5.3e-7 | 9.0e-9 |
| 10 km (a sparse pass) | 4.5e-7 | 8.6e-9 |

No integer band moves at any spacing at any of these counts. So a sparse pass for later interpolation
onto a finer grid — where points are deliberately far apart — is as safe as a dense one.
"""
struct WarmStart{C<:SceneCenterStart} <: ZeroDopplerStart
    iterations::Int
    # The first point of a loop has no predecessor, so it falls back to a cold solve. Held rather than
    # constructed there so the fallback shares the precomputed scene-center state.
    cold::C

    function WarmStart{C}(iterations::Int, cold::C) where {C<:SceneCenterStart}
        # 8 is the floor, set by the float bands rather than by the integer ones. No index moves at
        # any count from 5 up, but the off2vel bands drift: at 5 they miss the isce3 reference by
        # 7.3e-4 relative, outside the 2e-4 the radar fixture asserts, and at 6 by 1.6e-4 — inside
        # it with almost no room. 8 recovers the cold-start agreement of 1.07e-4 exactly.
        iterations >= WARM_START_MIN_ITERATIONS || throw(ArgumentError(
            "WarmStart iterations must be at least $WARM_START_MIN_ITERATIONS, got $iterations; " *
            "below that the off2vel float bands drift outside the 2e-4 the radar fixture asserts " *
            "against isce3, even though no integer index moves. See `WarmStart`."))
        iterations <= GEO2RDR_ITERATIONS || throw(ArgumentError(
            "WarmStart iterations is $iterations, above GEO2RDR_ITERATIONS " *
            "($(GEO2RDR_ITERATIONS)); a warm start converges in fewer iterations than a cold one, " *
            "so a larger count buys nothing and would be slower than not warm-starting"))
        return new{C}(iterations, cold)
    end
end

WarmStart(iterations::Int, cold::C = SceneCenterStart()) where {C<:SceneCenterStart} =
    WarmStart{C}(iterations, cold)

WarmStart(; iterations::Integer = 8) = WarmStart(Int(iterations))

# Rebound to a coordinate: the same policy with its cold fallback specialized to this scene. Called
# once per loop, so the first point's fallback does not re-interpolate the scene midpoint.
_bind(s::SceneCenterStart, c::RadarCoordinate) = SceneCenterStart(c)
_bind(w::WarmStart, c::RadarCoordinate) = WarmStart(w.iterations, SceneCenterStart(c))

"""
    ZeroDopplerSeed

The running state a [`WarmStart`](@ref) carries from one grid point to the next: the previous point's
azimuth time on both clocks and the satellite state there, or `nothing` before the first point.

Mutable because the per-point loop updates it in place; one instance per loop, so a threaded run's
tasks never share one. See [`WarmStart`](@ref) on why the sequence matters.
"""
mutable struct ZeroDopplerSeed
    valid::Bool
    aztime::Float64
    orbittime::Float64
    position::SVector{3,Float64}
    velocity::SVector{3,Float64}
end

ZeroDopplerSeed() = ZeroDopplerSeed(false, 0.0, 0.0, SVector{3,Float64}(0.0, 0.0, 0.0),
                                    SVector{3,Float64}(0.0, 0.0, 0.0))

"""
    pointgeometry(tf::TransformPair, gx, gy, gz, c::RadarCoordinate, normal)
        -> Tuple{PointGeometry,RadarSpacing,RadarPoint}

Geometry at the grid point `(gx, gy, gz)` for the radar acquisition `c`.

`tf.forward` maps grid coordinates to geodetic degrees and `tf.inverse` back — the opposite naming
sense from the projected path's image coordinates, but the same direction of use: forward to leave the
grid, inverse to return to it.

Returns the [`PointGeometry`](@ref) the shared outputs consume, the [`RadarSpacing`](@ref) they divide
by, and the [`RadarPoint`](@ref) the range and azimuth indices come from.

Reproduces `geogridRadar.cpp:850-1085`. Three solves per point — one `geo2rdr`, then two
range-Doppler iterations for the along-track step — so this is roughly two orders of magnitude more
expensive than the projected path's three transform calls.
"""
function pointgeometry(tf::TransformPair, gx::Real, gy::Real, gz::Real,
                       c::RadarCoordinate, normal::SVector{3,Float64},
                       start::ZeroDopplerStart = SceneCenterStart(),
                       seed::Union{Nothing,ZeroDopplerSeed} = nothing)
    el = Ellipsoid()
    grid = SVector{3,Float64}(Float64(gx), Float64(gy), Float64(gz))

    # Grid point to ECEF, via geodetic. The reference's `fwdTrans` produces degrees, and `llhi` holds
    # radians in (lon, lat, h) order — the swap at `:892-898` is between GDAL's (x, y) and isce3's
    # (lon, lat), which are the same order once both are read as (east, north).
    lon_d, lat_d, h = tf.forward(grid[1], grid[2], grid[3])
    xyz = lonlat_to_xyz(el, SVector{3,Float64}(lon_d * DEG2RAD, lat_d * DEG2RAD, h))

    # The zero-Doppler solve. `SceneCenterStart` is the reference's: the same scene-midpoint guess for
    # every point. `WarmStart` begins from the previous point's answer and runs fewer iterations.
    p = _solve_zero_doppler(start, seed, c, xyz)
    _record!(seed, p)

    # Range axis: one range sample along the look direction, back to grid coordinates.
    los_ecef = unitvec3(p.look)
    range_step = xyz + los_ecef * c.dr
    xdiff = _to_grid(tf, el, range_step) - grid
    xlen = norm3(xdiff)

    # Azimuth axis: advance one pulse interval and re-solve. `tline` and `tlined` both step, as they
    # both received every Newton increment.
    satx, satv = interpolate(c.orbit, p.orbittime + 1 / c.prf)
    targ_xyz, targ_llh = _range_doppler(el, c, satx, satv, p.range, Float64(gz))

    ydiff = _to_grid(tf, targ_llh) - grid
    ylen = norm3(ydiff)

    # The along-track step in ECEF, which the operator and the azimuth scale factor divide by. Not
    # `ylen`: that is the same step in grid coordinates, and the two differ by the local map scale.
    da = norm3(targ_xyz - xyz)

    g = PointGeometry((Float64(range_index(p, c.starting_range, c.dr)),
                       Float64(azimuth_index(p, c.sensing_start, c.prf))),
                      unitvec3(xdiff), unitvec3(ydiff), xlen, ylen, normal)

    return (g, RadarSpacing((c.dr, da), (c.dr, ylen)), p)
end

# The zero-Doppler solve under each start policy. Two methods rather than a branch so the cold path
# compiles to exactly what it did before this option existed.
@inline function _solve_zero_doppler(::SceneCenterStart{Nothing}, ::Union{Nothing,ZeroDopplerSeed},
                                     c::RadarCoordinate, xyz::SVector{3,Float64})
    pm, vm = interpolate(c.orbit, orbit_midtime(c))
    return geo2rdr(c.orbit, xyz, midtime(c), orbit_midtime(c), pm, vm)
end

@inline function _solve_zero_doppler(s::SceneCenterStart{<:Tuple}, ::Union{Nothing,ZeroDopplerSeed},
                                     c::RadarCoordinate, xyz::SVector{3,Float64})
    az0, ob0, pm, vm = s.state
    return geo2rdr(c.orbit, xyz, az0, ob0, pm, vm)
end

@inline function _solve_zero_doppler(w::WarmStart, seed::Union{Nothing,ZeroDopplerSeed},
                                     c::RadarCoordinate, xyz::SVector{3,Float64})
    # No predecessor yet — the first point of the loop, or a caller that passed no seed. Falls back to
    # a full cold solve, so the sequence starts from the same place the reference would.
    if seed === nothing || !seed.valid
        return _solve_zero_doppler(w.cold, seed, c, xyz)
    end
    return geo2rdr(c.orbit, xyz, seed.aztime, seed.orbittime, seed.position, seed.velocity,
                   w.iterations)
end

# Carry this point's answer forward. A no-op without a seed, which is the cold path.
@inline _record!(::Nothing, ::RadarPoint) = nothing
@inline function _record!(seed::ZeroDopplerSeed, p::RadarPoint)
    seed.valid = true
    seed.aztime = p.aztime
    seed.orbittime = p.orbittime
    seed.position = p.position
    seed.velocity = p.velocity
    return nothing
end

# A geodetic position as a grid coordinate. Takes radians and converts, since every caller here holds
# radians and the transform wants degrees.
@inline function _to_grid(tf::TransformPair, llh::SVector{3,Float64})
    gx, gy, gz = tf.inverse(llh[1] / DEG2RAD, llh[2] / DEG2RAD, llh[3])
    return SVector{3,Float64}(gx, gy, gz)
end

# An ECEF position as a grid coordinate. Separate from the above because the range-Doppler solve
# already produces a geodetic position, and re-deriving it from ECEF would not round-trip to the same
# last bits — so that caller passes its `llh` straight to the method above.
@inline _to_grid(tf::TransformPair, el::Ellipsoid, xyz::SVector{3,Float64}) =
    _to_grid(tf, xyz_to_lonlat(el, xyz))

# The range-Doppler solve: where on the range sphere, at this satellite position, the ground sits at
# the given height.
#
# `geogridRadar.cpp:1038-1071`. Structurally isce3's `rdr2geo` against a constant-height DEM, and
# transcribed separately because it differs in three ways: a fixed 10 iterations with no convergence
# test, a residual computed and discarded, and `alpha` grouped differently — `:1046` divides only the
# second term by `dot(vhat, that)` where `Rdr2Geo.icc:94` divides the whole numerator. Those agree
# only because `dopfact` is hardcoded to zero at `:1009`, which is why both forms are kept, each at
# its own site.
# Geocentric radius of the point at geodetic latitude `slat` — given as its sine — and height `h`.
#
# The same quantity as `norm3(lonlat_to_xyz(el, (lon, lat, h)))`, and independent of `lon`: the
# ellipsoid is a surface of revolution, so rotating a point about the polar axis does not change its
# distance from the center. Two expressions for one value, so they round differently: measured within
# 2 ULP on WGS84 and 3 on a deliberately eccentric ellipsoid, so 2.8e-9 m at Earth radius. That is the
# magnitude of this solve's own fixed-point floor, and 1.3e-9 of a range sample.
@inline function _geocentric_radius(el::Ellipsoid, slat::Float64, h::Float64)
    s2 = slat * slat
    N = el.a / sqrt(1.0 - el.e2 * s2)
    re = N + h
    rp = N * (1.0 - el.e2) + h
    return sqrt(re * re * (1.0 - s2) + rp * rp * s2)
end

function _range_doppler(el::Ellipsoid, c::RadarCoordinate, satx::SVector{3,Float64},
                        satv::SVector{3,Float64}, rngpix::Float64, height::Float64)
    vhat = unitvec3(satv)
    radius, _, _, sat_dist = nadir_sphere(el, satx)
    tcn = geodetic_tcn(satx, satv)

    dopfact = 0.0
    ndotv = dot3(tcn.nhat, vhat)
    vdott = dot3(vhat, tcn.that)

    zsch = height
    llhi = SVector{3,Float64}(0.0, 0.0, 0.0)
    targ_xyz = SVector{3,Float64}(0.0, 0.0, 0.0)

    a = sat_dist

    # `a` is the satellite's geocentric distance, invariant across the loop; `a / rngpix`,
    # `rngpix / a` and the look sign are invariant with it. Hoisting them by hand is *not*
    # value-preserving — it moves the ECEF result 8.7e-10 m — and measures no faster once this function
    # is inlined into the per-point loop, though it does look like a 1.06x win benchmarked in
    # isolation. So the expression stays in the reference's shape. See `benchmark/run_items23.jl`.
    for it in 1:RANGE_DOPPLER_ITERATIONS
        b = radius + zsch

        costheta = 0.5 * (a / rngpix + rngpix / a - (b / a) * (b / rngpix))
        sintheta = sqrt(1 - costheta * costheta)

        gamma = rngpix * costheta
        # geogridRadar's grouping, not isce3's. See the comment above.
        alpha = dopfact - gamma * ndotv / vdott
        beta = -looksign(c.look_side) * sqrt(rngpix * rngpix * sintheta * sintheta - alpha * alpha)

        delta = alpha * tcn.that + beta * tcn.chat + gamma * tcn.nhat
        targ_vec = satx + delta

        # Snap to the requested height, then back to ECEF. The reference overwrites `llhi[2]` in
        # place (`:1057`) and converts from the modified value.
        lonlat = xyz_to_lonlat(el, targ_vec)
        llhi = SVector{3,Float64}(lonlat[1], lonlat[2], height)
        # The loop reads only `zsch` from this, and `zsch` needs the *norm* of the snapped position,
        # which `_geocentric_radius` gives from the latitude alone. So the forward conversion runs once
        # rather than per iteration; `targ_xyz` is read only after the loop. Worth 1.03x of a point.
        if it == RANGE_DOPPLER_ITERATIONS
            targ_xyz = lonlat_to_xyz(el, llhi)
        end

        zsch = _geocentric_radius(el, sin(lonlat[2]), height) - radius
        # `rdiff` is computed here in the reference and never read, so it is not computed at all.
    end

    return (targ_xyz, llhi)
end
