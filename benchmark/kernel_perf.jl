# Per-point cost of the kernel, by transform.
#
# The arithmetic is a few tens of nanoseconds; a real map projection is an order of magnitude more.
# That ratio is why the identity path (grid and image sharing a CRS) is a dispatch rather than a
# branch, and why blocking and threading target the transform rather than the arithmetic.

using ImagePairGeometry
using ImagePairGeometry: pointgeometry, surface_normal, close_slope_parallel, pixel_offset,
                         offset_to_velocity, scale_factors, cross_check, transform_pair,
                         IdentityTransform, AffineTransform, TransformPair, pixel_index, inbounds
using BenchmarkTools
using Proj

const C = ProjectedCoordinate(origin = (300000.0, 7800000.0), spacing = (30.0, -30.0),
                              size = (6000, 6000))
const N = surface_normal(0.02, -0.05)
const DT = 91 * 86400.0

"""One complete per-point computation: geometry, index, offset, operator, scale, degeneracy."""
function fullpoint(tf, gx, gy, gz, c, n, dt)
    g = pointgeometry(tf, gx, gy, gz, c, n)
    xi, yi = pixel_index(g, c)
    inbounds(xi, yi, c) || return (0.0,)
    v = close_slope_parallel(300.0, -120.0, n)
    dx, dy = pixel_offset(v, g, dt)
    o = offset_to_velocity(g, c, dt)
    sf = scale_factors(g, c)
    return (xi, yi, dx, dy, o..., sf..., cross_check(g))
end

const CASES = [
    "identity" => transform_pair(IdentityTransform()),
    "affine" => transform_pair(AffineTransform(a = 1.0001, b = 0.0002, c = 13.0,
                                               d = -0.0002, e = 0.9999, f = -7.0)),
    "proj same CRS (noop)" => TransformPair(Proj.Transformation("EPSG:32624", "EPSG:32624"),
                                            Proj.Transformation("EPSG:32624", "EPSG:32624")),
    "proj UTM24N<->3413" => TransformPair(Proj.Transformation("EPSG:32624", "EPSG:3413"),
                                          Proj.Transformation("EPSG:3413", "EPSG:32624")),
]

println("Per-point cost, full computation (minimum of a benchmark sample)\n")
println(rpad("transform", 24), lpad("ns/point", 10), lpad("bytes", 8))
for (name, tf) in CASES
    b = @benchmark fullpoint($tf, 300150.0, 7799850.0, 500.0, $C, $N, $DT)
    alloc = @allocated fullpoint(tf, 300150.0, 7799850.0, 500.0, C, N, DT)
    println(rpad(name, 24), lpad(round(minimum(b).time; digits = 1), 10), lpad(alloc, 8))
end
