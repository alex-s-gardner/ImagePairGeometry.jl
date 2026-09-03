# Per-point cost of the radar numerics, by stage.
#
# The radar forward mapping is far more expensive than the projected one: where that path costs three
# transform calls, this one runs a 51-iteration solve, each iteration interpolating the orbit. So the
# cost is dominated by `interpolate`, and the ratio below is what any caching or lattice scheme has to
# beat.

using ImagePairGeometry
using ImagePairGeometry: Ellipsoid, Orbit, interpolate, geo2rdr, rdr2geo,
                         lonlat_to_xyz, xyz_to_lonlat, geodetic_tcn, nadir_sphere,
                         LookRight, GEO2RDR_ITERATIONS
using BenchmarkTools
using StaticArrays: SVector

const EL = Ellipsoid()
const WVL = 0.05546576

"""
A near-polar circular orbit.

The same parameters `test/reference/gen_radar_numerics.py` generates its fixture orbit from, so a
timing here is measured on the geometry the correctness tests assert against. Duplicated rather than
shared because the fixture side is a Python generator and the committed JSON is its output — reading
that JSON here would make the benchmark depend on the test fixtures.
"""
function testorbit(; n = 25, spacing = 10.0)
    R = 7.0e6
    w = sqrt(3.986004418e14 / R^3)
    inc = deg2rad(98.0)
    t = [(i - 1) * spacing for i in 1:n]
    pos = [SVector{3,Float64}(R * cos(w * ti), R * sin(w * ti) * cos(inc),
                              R * sin(w * ti) * sin(inc)) for ti in t]
    vel = [SVector{3,Float64}(-R * w * sin(w * ti), R * w * cos(w * ti) * cos(inc),
                              R * w * cos(w * ti) * sin(inc)) for ti in t]
    return Orbit(t[1], spacing, pos, vel)
end

const ORB = testorbit()
const TMID = 120.0
const PM, VM = interpolate(ORB, TMID)
const LLH = SVector{3,Float64}(-0.0866, 0.0533, 500.0)
const TARGET = lonlat_to_xyz(EL, LLH)

println("Per-point cost of the radar numerics (minimum of a benchmark sample)\n")
println(rpad("stage", 34), lpad("ns", 12), lpad("bytes", 8))

"""One table row: the stage's minimum time and its allocation."""
row(label, b, bytes) =
    println(rpad(label, 34), lpad(round(minimum(b).time; digits = 1), 12), lpad(bytes, 8))

# Each `@benchmark` interpolates its arguments with `$` so the benchmarked call sees runtime values.
# Collecting the stages into a table of closures instead lets the compiler fold each call to a literal
# and report 1 ns for arithmetic that cannot be that fast — which is why these are written out rather
# than looped over.
const T_INTERP, T_SOLVE = let
    row("lonlat_to_xyz", (@benchmark lonlat_to_xyz($EL, $LLH)),
        @allocated(lonlat_to_xyz(EL, LLH)))

    row("xyz_to_lonlat", (@benchmark xyz_to_lonlat($EL, $TARGET)),
        @allocated(xyz_to_lonlat(EL, TARGET)))

    row("geodetic_tcn", (@benchmark geodetic_tcn($PM, $VM)),
        @allocated(geodetic_tcn(PM, VM)))

    row("nadir_sphere", (@benchmark nadir_sphere($EL, $PM)),
        @allocated(nadir_sphere(EL, PM)))

    interp = @benchmark interpolate($ORB, $(122.5))
    row("interpolate (one orbit eval)", interp, @allocated(interpolate(ORB, 122.5)))

    solve = @benchmark geo2rdr($ORB, $TARGET, $TMID, $TMID, $PM, $VM)
    row("geo2rdr (51 iterations)", solve, @allocated(geo2rdr(ORB, TARGET, TMID, TMID, PM, VM)))

    row("rdr2geo", (@benchmark rdr2geo($ORB, $EL, $(120.0), $(8.5e5); height = $(500.0),
                                       wavelength = $WVL, side = $LookRight)),
        @allocated(rdr2geo(ORB, EL, 120.0, 8.5e5; height = 500.0, wavelength = WVL,
                           side = LookRight)))

    (minimum(interp).time, minimum(solve).time)
end

# What fraction of `geo2rdr` is orbit interpolation? That is the number that decides whether the radar
# path is worth caching across pairs over one footprint. Both timings come from the table above rather
# than from a second measurement of the same two calls.
println("\ngeo2rdr: ", GEO2RDR_ITERATIONS, " orbit evaluations account for ",
        round(100 * GEO2RDR_ITERATIONS * T_INTERP / T_SOLVE; digits = 1), "% of the solve")
