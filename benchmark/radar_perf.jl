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

"""A near-polar circular orbit, as the fixtures use."""
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

# Every argument is interpolated with `$` so the benchmarked call sees runtime values. Passing a
# closure over `const` globals instead lets the compiler fold the whole call to a literal and report
# 1 ns for arithmetic that cannot be that fast.
let
    b = @benchmark lonlat_to_xyz($EL, $LLH)
    println(rpad("lonlat_to_xyz", 34), lpad(round(minimum(b).time; digits = 1), 12),
            lpad(@allocated(lonlat_to_xyz(EL, LLH)), 8))

    b = @benchmark xyz_to_lonlat($EL, $TARGET)
    println(rpad("xyz_to_lonlat", 34), lpad(round(minimum(b).time; digits = 1), 12),
            lpad(@allocated(xyz_to_lonlat(EL, TARGET)), 8))

    b = @benchmark geodetic_tcn($PM, $VM)
    println(rpad("geodetic_tcn", 34), lpad(round(minimum(b).time; digits = 1), 12),
            lpad(@allocated(geodetic_tcn(PM, VM)), 8))

    b = @benchmark nadir_sphere($EL, $PM)
    println(rpad("nadir_sphere", 34), lpad(round(minimum(b).time; digits = 1), 12),
            lpad(@allocated(nadir_sphere(EL, PM)), 8))

    b = @benchmark interpolate($ORB, $(122.5))
    println(rpad("interpolate (one orbit eval)", 34), lpad(round(minimum(b).time; digits = 1), 12),
            lpad(@allocated(interpolate(ORB, 122.5)), 8))

    b = @benchmark geo2rdr($ORB, $TARGET, $TMID, $TMID, $PM, $VM)
    println(rpad("geo2rdr (51 iterations)", 34), lpad(round(minimum(b).time; digits = 1), 12),
            lpad(@allocated(geo2rdr(ORB, TARGET, TMID, TMID, PM, VM)), 8))

    b = @benchmark rdr2geo($ORB, $EL, $(120.0), $(8.5e5); height = $(500.0),
                           wavelength = $WVL, side = $LookRight)
    println(rpad("rdr2geo", 34), lpad(round(minimum(b).time; digits = 1), 12),
            lpad(@allocated(rdr2geo(ORB, EL, 120.0, 8.5e5; height = 500.0, wavelength = WVL,
                                    side = LookRight)), 8))
end

# What fraction of `geo2rdr` is orbit interpolation? That is the number that decides whether the
# radar path is worth caching across pairs over one footprint.
let
    interp = minimum(@benchmark interpolate($ORB, 122.5)).time
    solve = minimum(@benchmark geo2rdr($ORB, $TARGET, $TMID, $TMID, $PM, $VM)).time
    share = GEO2RDR_ITERATIONS * interp / solve
    println("\ngeo2rdr: ", GEO2RDR_ITERATIONS, " orbit evaluations account for ",
            round(100 * share; digits = 1), "% of the solve")
end
