# What a radar run costs at scale, and where the time goes.
#
# The projected path's cost is dominated by PROJ — three calls per point, which is why
# `InterpolatedTransform` exists. The radar path's is dominated by the solve instead, and this measures
# the ratio that decides whether the same trick is worth applying here.

using ImagePairGeometry
using ImagePairGeometry: Ellipsoid, Orbit, RadarCoordinate, LookRight, incidence_angle,
                         CoregisteredPair, MapGrid, GeometryInputs, InMemoryInputs,
                         TransformPair, footprint_bounds, grid_window, pairgeometry,
                         pairgeometry_blocked, pointgeometry, surface_normal, nodata_from,
                         interpolate, rdr2geo, geo2rdr, orbit_midtime, midtime, _range_doppler
using BenchmarkTools
using Proj
using StaticArrays: SVector

const EL = Ellipsoid()
const DR = 2.329562114715323
const SR = 8.0e5
const PRF = 486.4863103
const WVL = 0.05546576

function scene_orbit()
    R = 7.0e6
    w = sqrt(3.986004418e14 / R^3)
    inc = deg2rad(98.0)
    t = [(i - 1) * 10.0 for i in 1:61]
    pos = [SVector{3,Float64}(R * cos(w * ti), R * sin(w * ti) * cos(inc),
                              R * sin(w * ti) * sin(inc)) for ti in t]
    vel = [SVector{3,Float64}(-R * w * sin(w * ti), R * w * cos(w * ti) * cos(inc),
                              R * w * cos(w * ti) * sin(inc)) for ti in t]
    return Orbit(t[1], 10.0, pos, vel)
end

const ORB = scene_orbit()
const RKW = (starting_range = SR, dr = DR, sensing_start = 300.0, prf = PRF,
             nsamples = 10000, nlines = 8000, look_side = LookRight, wavelength = WVL)
const IA = incidence_angle(; orbit = ORB, RKW...)
const RC = RadarCoordinate(; orbit = ORB, incidence_angle = IA, RKW...)
const PAIR = CoregisteredPair(RC; dt = 6 * 86400.0)

"""A fresh PROJ pair. A thunk, since a threaded run needs one context per task."""
projpair() = TransformPair(Proj.Transformation("EPSG:32632", "EPSG:4326"; always_xy = true),
                           Proj.Transformation("EPSG:4326", "EPSG:32632"; always_xy = true))

"""A grid of `n`×`n` points at `spacing` m, centred on the swath."""
function scene(n, spacing)
    ext = footprint_bounds(projpair(), RC)
    cx = 0.5 * (ext.X[1] + ext.X[2])
    cy = 0.5 * (ext.Y[1] + ext.Y[2])
    gt = (cx - n * spacing / 2, spacing, 0.0, cy + n * spacing / 2, 0.0, -spacing)
    grid = MapGrid(geotransform = gt, size = (n, n), crs = 32632)
    win = CartesianIndices((1:n, 1:n))
    sz = (n, n)
    inputs = GeometryInputs(dem = fill(500.0, sz),
                            dhdx = fill(0.02, sz), dhdy = fill(-0.01, sz),
                            vx = fill(120.0, sz), vy = fill(-80.0, sz))
    return (grid, win, inputs)
end

println("Radar cost at scale\n")

# Per-point cost, and its breakdown. `$`-interpolated so nothing folds to a literal.
let
    llh = rdr2geo(ORB, EL, 305.0, SR + 4000 * DR; height = 500.0, wavelength = WVL,
                  side = LookRight)
    lx, ly, lz = rad2deg(llh[1]), rad2deg(llh[2]), llh[3]
    gxx, gyy = projpair().inverse(lx, ly)
    tf = projpair()
    nrm = surface_normal(0.02, -0.01)
    pm, vm = interpolate(ORB, orbit_midtime(RC))
    tgt = ImagePairGeometry.lonlat_to_xyz(EL, SVector{3,Float64}(llh[1], llh[2], llh[3]))
    satx, satv = interpolate(ORB, 305.0)

    point = minimum(@benchmark pointgeometry($tf, $gxx, $gyy, $lz, $RC, $nrm)).time
    solve = minimum(@benchmark geo2rdr($ORB, $tgt, $(midtime(RC)), $(orbit_midtime(RC)),
                                       $pm, $vm)).time
    rdop = minimum(@benchmark _range_doppler($EL, $RC, $satx, $satv, $(SR + 4000 * DR),
                                             $(500.0))).time
    proj = minimum(@benchmark $tf.forward($gxx, $gyy, $lz)).time

    println(rpad("stage", 30), lpad("ns", 10), lpad("share", 9))
    for (nm, v) in (("pointgeometry (whole)", point), ("  geo2rdr", solve),
                    ("  range-Doppler solve", rdop), ("  one PROJ call", proj))
        println(rpad(nm, 30), lpad(round(v; digits = 1), 10),
                lpad(startswith(nm, " ") ? "$(round(100v / point; digits = 1))%" : "", 9))
    end
    println("\nPROJ is ", round(100 * 3 * proj / point; digits = 2),
            "% of a radar point (three calls), against ~95% of a projected one.")
    println("So the lattice trick that wins on the projected path has almost nothing to")
    println("take here: the solve is the cost, and it cannot be interpolated away without")
    println("interpolating the answer itself.")
end

# Whole-window cost, and what threading buys. The per-point figure from the first grid is kept for the
# extrapolation at the end rather than measured a second time.
per_point = Ref(0.0)
println("\n", rpad("grid", 14), lpad("points", 9), lpad("serial ms", 12), lpad("ms/point", 11))
for (n, spacing) in ((48, 500.0), (96, 250.0))
    grid, win, inputs = scene(n, spacing)
    r = pairgeometry(grid, PAIR, inputs; transform = projpair(), window = win,
                     nodata = nodata_from(0.0))
    run1() = pairgeometry(grid, PAIR, inputs; transform = projpair(), window = win,
                          nodata = nodata_from(0.0))
    t = minimum(@benchmark $run1() samples = 5 evals = 1).time
    n == 48 && (per_point[] = t / length(win))
    println(rpad("$(n)x$(n) @ $(Int(spacing))m", 14), lpad(length(win), 9),
            lpad(round(t / 1e6; digits = 1), 12),
            lpad(round(t / length(win) / 1e3; digits = 2), 11))
end

let
    n, spacing = 96, 250.0
    grid, win, inputs = scene(n, spacing)
    src = InMemoryInputs(inputs, win)
    println("\n", rpad("ntasks", 8), lpad("ms", 10), lpad("speedup", 10),
            "   (blocksize 32x32, ", Threads.nthreads(), " threads available)")
    base = 0.0
    for nt in (1, 2, 4)
        nt > Threads.nthreads() && continue
        runb() = pairgeometry_blocked(grid, PAIR, src; transform = projpair, window = win,
                                      blocksize = (32, 32), ntasks = nt,
                                      nodata = nodata_from(0.0))
        t = minimum(@benchmark $runb() samples = 5 evals = 1).time
        nt == 1 && (base = t)
        println(rpad(nt, 8), lpad(round(t / 1e6; digits = 1), 10),
                lpad(round(base / t; digits = 2), 10))
    end
end

# What a production-scale run would cost, from the measured per-point figure.
let
    per = per_point[]
    println("\nAt ", round(per / 1e3; digits = 2), " us/point, one thread:")
    for (nm, npts) in ("a 5000x5000 grid" => 25_000_000,
                       "an ITS_LIVE tile, 915x915 at 120 m" => 915^2)
        println("  ", rpad(nm, 38), round(per * npts / 1e9 / 60; digits = 1), " min")
    end
    println("\nSo a production tile is seconds and a continental grid is minutes, both")
    println("without a lattice. Threading is the lever that matters here.")
end

# PROJ's atexit teardown aborts when transformations built in this process are still reachable, so the
# process is ended before it runs. Nothing above depends on cleanup.
exit(0)
