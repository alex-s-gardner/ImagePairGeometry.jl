# The radar `pixel_offset` method: where a ground point falls in the secondary acquisition.
#
# Two solves already present in this package, composed:
#
#   grid point --lonlat_to_xyz--> ECEF --geo2rdr(orbit 1)--> (t1, r1)
#                                      --geo2rdr(orbit 2)--> (t2, r2)
#
# and the offset is the difference of the two, each expressed in its own acquisition's pixel units. isce3
# calls the composition `rdr2rdr` and reaches it through `rdr2geo` then `geo2rdr`; here the ground point is
# already in hand — the caller passes a grid point, as the kernel does — so the `rdr2geo` half is not
# needed and both acquisitions are solved directly. That is fewer solves and no dependence on `rdr2geo`'s
# fixed-point iteration, and it is the same answer: `rdr2geo(t1, r1)` returns the point `geo2rdr` was
# given, to the 1.9e-9 m `REFERENCE.md` records.
#
# Both differences are taken in *pixels*, not in meters and seconds, because the two acquisitions need not
# share a range spacing or a PRF, and two bursts of one subswath do not share a sensing start either.

"""
    RDR2RDR_ITERATIONS

Iterations the secondary's zero-Doppler solve runs: 16, the same as [`GEO2RDR_ITERATIONS`](@ref).

The reference acquisition's solve starts from the scene midpoint and needs the full count. The
secondary's could start from the reference's answer — a repeat pass images the same ground within a few
lines — and converge in fewer, as [`WarmStart`](@ref) does for adjacent grid points. That is not done
here, and the reason is that the saving does not pay for what it costs.

`pixel_offset` is not on the per-point critical path. The offset field is evaluated on a coarse lattice —
a few thousand points for a whole subswath against the millions `pairgeometry` visits — so halving one of
its two solves saves microseconds per pair, where the equivalent choice inside the kernel was worth 1.6×
of a run. Against that, a warm start would make the offset depend on which point was solved before it,
which is the property [`WarmStart`](@ref) documents as costing blocking invariance. A lattice built in one
piece and the same lattice built in blocks would then disagree, and the field is meant to be a pure
function of position.

So both solves are cold and independent, and a lattice node's value depends on nothing but its own
coordinates.
"""
const RDR2RDR_ITERATIONS = GEO2RDR_ITERATIONS

# The radar method under `pixel_offset`, declared in `misregistration.jl`.
#
# `tf` is not taken here: the caller passes a grid point and this needs an ECEF one, and on the radar path
# the grid-to-geodetic transform is `pair`-independent state the caller already holds. So the geodetic
# conversion happens in `_pixel_offset`'s caller — see `pixel_offset`'s `transform` keyword.
function _radar_offset(ref::RadarCoordinate, sec::RadarCoordinate, xyz::SVector{3,Float64})
    el = Ellipsoid()

    # Each acquisition solved from its own scene midpoint, as the kernel solves the reference's. Both are
    # cold; see `RDR2RDR_ITERATIONS`.
    p1 = _solve_from_midpoint(ref, xyz)
    p2 = _solve_from_midpoint(sec, xyz)

    # Unrounded, unlike `range_index`/`azimuth_index`: a resampler needs the fractional part, and the
    # rounded indices the kernel reports have already discarded it.
    samp1 = (p1.range - ref.starting_range) / ref.dr
    samp2 = (p2.range - sec.starting_range) / sec.dr
    line1 = (p1.aztime - ref.sensing_start) * ref.prf
    line2 = (p2.aztime - sec.sensing_start) * sec.prf

    return (samp2 - samp1, line2 - line1)
end

# One acquisition's zero-Doppler solve, from its own scene midpoint on both clocks. The same starting
# state `SceneCenterStart` builds, without the per-loop caching that policy exists to provide.
@inline function _solve_from_midpoint(c::RadarCoordinate, xyz::SVector{3,Float64})
    pm, vm = interpolate(c.orbit, orbit_midtime(c))
    return geo2rdr(c.orbit, xyz, midtime(c), orbit_midtime(c), pm, vm, RDR2RDR_ITERATIONS)
end

"""
    height_sensitivity(pair::CoregisteredPair, x, y; heights = DEFAULT_ZRANGE, transform) -> NTuple{2,Float64}

Change in [`pixel_offset`](@ref) per meter of elevation, as `(dsample_dh, dline_dh)`.

Evaluated as a difference across `heights`, two `pixel_offset` calls, at the grid point `(x, y)`. Scene
center is the point to pass.

# What it is for

This is the measure of whether a pixel shift describes the relationship between the two acquisitions at
all. For a near-repeat pass the two orbits are close, their look directions at a given ground point nearly
agree, and the offset barely depends on elevation — 1.5e-4 samples per meter on the 24-day Sentinel-1 pair
in `docs/src/coregistration.md`, so 0.3 samples across 2 km of relief. As the orbits separate the offset
becomes increasingly a parallax term: a DEM stops being optional, then the two acquisitions stop seeing
the same scene at all, since layover and shadow differ between look directions that differ enough.

Baseline in meters is the wrong instrument for that, which is why this exists. What matters is how much
of the offset is height-dependent, and that depends on the incidence angle as well as on the separation —
so it is measured rather than inferred. Returns the number, not a verdict, as
[`geo2rdr_iterations_needed`](@ref) does: what threshold matters depends on what the caller is doing, and
an interferometric use tolerates far less than amplitude tracking.

`transform` is as [`pixel_offset`](@ref) takes it.
"""
function height_sensitivity(pair::CoregisteredPair, x::Real, y::Real;
                            heights = DEFAULT_ZRANGE, transform = IdentityTransform())
    z0, z1 = Float64(heights[1]), Float64(heights[2])
    z1 == z0 && throw(ArgumentError(
        "height_sensitivity needs two distinct heights, got $heights"))
    s0, l0 = pixel_offset(pair, x, y, z0; transform)
    s1, l1 = pixel_offset(pair, x, y, z1; transform)
    dz = z1 - z0
    return ((s1 - s0) / dz, (l1 - l0) / dz)
end
