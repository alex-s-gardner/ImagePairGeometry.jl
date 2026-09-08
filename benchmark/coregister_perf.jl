# Cost of the coregistration path: the offset field, the lattice that makes it affordable, and the
# resampler.
#
# The claims this measures are the ones `docs/src/coregistration.md` rests on. Two of them:
#
# A `pixel_offset` is two zero-Doppler solves, so it costs about what two radar grid points cost, and
# evaluating one per sample of a Sentinel-1 subswath is minutes. That is the number that makes
# `LatticeOffsetField` necessary rather than an optimization.
#
# The resampler reads a 9x9 chip per output sample and sinc-interpolates it, which is 64 taps of complex
# arithmetic. So its cost per sample is fixed and its throughput is what decides whether resampling a
# whole subswath is practical.

using ImagePairGeometry
using ImagePairGeometry: CoregisteredPair, RadarCoordinate, Orbit, LookRight, incidence_angle,
                         MapGrid, OffsetField, LatticeOffsetField, pixel_offset, fit_offset,
                         height_sensitivity, SincKernel, sinc_interpolate, ResampledSLC,
                         Ellipsoid, rdr2geo, SINC_ONE
using BenchmarkTools
using Random
using StaticArrays: SVector

"""
Two acquisitions a repeat cycle apart, on near-polar circular orbits.

Analytic rather than read from the committed fixture for the reason `radar_perf.jl` gives: the fixture is
a Python generator's output, and reading it here would make a benchmark depend on the test data. The
second orbit is displaced by a small along-track and cross-track baseline, which is what produces an
offset of the magnitude a real repeat pass has.
"""
function testpair(; n = 41, spacing = 10.0, dt = 24 * 86400.0)
    R = 7.0e6
    w = sqrt(3.986004418e14 / R^3)
    inc = deg2rad(98.0)
    t = [(i - 1) * spacing for i in 1:n]

    function orb(dr, dcross)
        pos = [SVector((R + dr) * cos(w * ti),
                       (R + dr) * sin(w * ti) * cos(inc) + dcross,
                       (R + dr) * sin(w * ti) * sin(inc)) for ti in t]
        vel = [SVector(-(R + dr) * w * sin(w * ti),
                       (R + dr) * w * cos(w * ti) * cos(inc),
                       (R + dr) * w * cos(w * ti) * sin(inc)) for ti in t]
        return Orbit(; time = t, position = pos, velocity = vel)
    end

    function coord(o)
        kw = (; orbit = o, starting_range = 8.0e5, dr = 2.3295621147, sensing_start = 100.0,
              prf = 486.4863103, nsamples = 24845, nlines = 12244, look_side = LookRight,
              wavelength = 0.05546576)
        return RadarCoordinate(; kw..., incidence_angle = incidence_angle(; kw...))
    end

    ref = coord(orb(0.0, 0.0))
    sec = coord(orb(120.0, 350.0))
    return CoregisteredPair(ref; dt, secondary = sec)
end

const PAIR = testpair()
const EL = Ellipsoid()

# A grid at the reference's scene center, in lon/lat degrees under the identity transform — the same
# convention the radar tests use to isolate the geometry from PROJ.
function testgrid(npoints)
    c = PAIR.coordinate
    llh = rdr2geo(c.orbit, EL, c.sensing_start + 0.5 * c.nlines / c.prf,
                  c.starting_range + 0.5 * c.nsamples * c.dr;
                  height = 0.0, wavelength = c.wavelength, side = c.look_side)
    lon, lat = rad2deg(llh[1]), rad2deg(llh[2])
    step = 0.002
    half = step * npoints / 2
    return (MapGrid(geotransform = (lon - half, step, 0.0, lat + half, 0.0, -step),
                    size = (npoints, npoints)),
            CartesianIndices((1:npoints, 1:npoints)), lon, lat)
end

const GRID, WIN, LON, LAT = testgrid(64)

println("=== one pixel_offset ===")
b = @benchmark pixel_offset($PAIR, $LON, $LAT, 0.0)
one_offset = minimum(b).time
println("  ", BenchmarkTools.prettytime(one_offset), "  ",
        BenchmarkTools.prettymemory(minimum(b).memory))

# What that means at the scale a real acquisition has. A Sentinel-1 IW subswath mosaic is about
# 1500 x 22000 samples; NISAR's is larger. This is the number `LatticeOffsetField` exists to avoid.
for (name, npix) in (("S1 IW subswath", 1500 * 22000), ("NISAR RSLC swath", 24845 * 12244))
    secs = one_offset * 1e-9 * npix
    println("  ", rpad(name, 18), lpad(npix, 12), " samples -> ",
            round(secs; digits = 1), " s exact, ",
            round(npix * 16 / 1e6; digits = 0), " MB if materialized")
end

println()
println("=== the field, exact against lattice ===")
exact = OffsetField(PAIR, GRID, WIN; dem = 500.0)
println("  exact 64x64 collect: ",
        BenchmarkTools.prettytime(minimum(@benchmark collect($exact)).time))
for L in (4, 16, 64)
    lat_f = LatticeOffsetField(PAIR, GRID, WIN; dem = 500.0, lattice = L)
    build = @benchmark LatticeOffsetField($PAIR, $GRID, $WIN; dem = 500.0, lattice = $L)
    read = @benchmark collect($lat_f)
    println("  lattice=", lpad(L, 3),
            "  build ", rpad(BenchmarkTools.prettytime(minimum(build).time), 11),
            "  read ", rpad(BenchmarkTools.prettytime(minimum(read).time), 11),
            "  nodes ", prod(ImagePairGeometry.latticesize(lat_f.lattice)))
end

println()
println("=== the polynomial fit ===")
field = (x, y, z) -> pixel_offset(PAIR, x, y, z)
for o in 1:3
    f = fit_offset(field, GRID, WIN; heights = (0.0, 1000.0, 2000.0), order = o)
    println("  order ", o, "  residual ", round(f.residual[1]; sigdigits = 3),
            " sample  eval ",
            BenchmarkTools.prettytime(minimum(@benchmark $f($LON, $LAT, 500.0)).time))
end
# Fitting itself is once per pair, so its cost is amortized over the whole scene.
fitb = @benchmark fit_offset($field, $GRID, $WIN; heights = (0.0, 1000.0, 2000.0), target = 0.01)
println("  selecting an order over 64x64x3 samples: ",
        BenchmarkTools.prettytime(minimum(fitb).time))

println()
println("=== height sensitivity ===")
hs = height_sensitivity(PAIR, LON, LAT)
println("  ", round(hs[1]; sigdigits = 3), " sample/m, ", round(hs[2]; sigdigits = 3), " line/m")
println("  over 2 km of relief: ", round(hs[1] * 2000; sigdigits = 3), " samples")
println("  cost: ", BenchmarkTools.prettytime(minimum(@benchmark height_sensitivity($PAIR, $LON, $LAT)).time))

println()
println("=== the sinc kernel ===")
kb = @benchmark SincKernel()
println("  table build: ", BenchmarkTools.prettytime(minimum(kb).time), "  ",
        BenchmarkTools.prettymemory(minimum(kb).memory), " (once per run)")
Random.seed!(9)
chip = ComplexF32.(randn(SINC_ONE, SINC_ONE) .+ 1im .* randn(SINC_ONE, SINC_ONE))
K = SincKernel()
ib = @benchmark sinc_interpolate($K, $chip, 5.25, 5.5)
println("  one interpolation (9x9 chip, 64 taps): ",
        BenchmarkTools.prettytime(minimum(ib).time))

println()
println("=== the resampler ===")
big = ComplexF32.(randn(2048, 2048) .+ 1im .* randn(2048, 2048))
off = fill((3.25, -2.5), 512, 512)
rs = ResampledSLC(big, off; kernel = K)
sb = @benchmark $rs[100, 100]
println("  one output sample: ", BenchmarkTools.prettytime(minimum(sb).time), "  ",
        BenchmarkTools.prettymemory(minimum(sb).memory))
for w in (64, 256)
    wb = @benchmark $rs[1:$w, 1:$w]
    t = minimum(wb).time
    println("  ", lpad(w, 4), "x", w, " window: ",
            rpad(BenchmarkTools.prettytime(t), 11),
            " = ", round(t / (w * w); digits = 1), " ns/sample, ",
            round(w * w / (t * 1e-9) / 1e6; digits = 2), " Msamples/s")
end
# What that throughput means for a whole subswath, which is the question a pipeline asks.
persample = minimum(sb).time * 1e-9
for (name, npix) in (("S1 IW subswath", 1500 * 22000), ("NISAR RSLC swath", 24845 * 12244))
    println("  ", rpad(name, 18), " whole: ", round(persample * npix; digits = 1),
            " s single-threaded")
end
