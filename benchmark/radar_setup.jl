# Shared setup for radar-path profiling: the fixture's acquisition on a grid large enough
# that per-point cost dominates.
using ImagePairGeometry
using ImagePairGeometry: Orbit, RadarCoordinate, LookRight, TransformPair, MapGrid,
                         CoregisteredPair, GeometryInputs, pairgeometry, pairgeometry_blocked,
                         InMemoryInputs, footprint_bounds, grid_window, nodata_from,
                         pointgeometry, geo2rdr, interpolate, orbit_midtime, midtime,
                         Ellipsoid, lonlat_to_xyz, xyz_to_lonlat, DEG2RAD, surface_normal,
                         _resolve_transform, _fill_geometry!, allocate_geometry,
                         window_geotransform, GeometryParams
using StaticArrays: SVector

const R, W, INC, SP, NSV = 7.0e6, 0.001078007612872506, 1.710422666954443, 10.0, 61

function bench_orbit()
    t = [(i - 1) * SP for i in 1:NSV]
    pos = [SVector{3,Float64}(R*cos(W*ti), R*sin(W*ti)*cos(INC), R*sin(W*ti)*sin(INC)) for ti in t]
    vel = [SVector{3,Float64}(-R*W*sin(W*ti), R*W*cos(W*ti)*cos(INC), R*W*cos(W*ti)*sin(INC)) for ti in t]
    return Orbit(t[1], SP, pos, vel)
end

const ORB = bench_orbit()

bench_coord(orbit = ORB) = RadarCoordinate(; orbit, starting_range = 800000.0,
    dr = 2.329562114715323, sensing_start = 300.0, prf = 486.4863103,
    nsamples = 10000, nlines = 8000, look_side = LookRight,
    wavelength = 0.05546576, incidence_angle = 0.7371595365886898)

# The grid CRS to geodetic degrees, which is the direction the radar path reads as (lon, lat, h).
bench_tf() = fast_transform(32632, 4326)

"""A grid of `n`×`n` points at 120 m over the fixture's swath, with every input band."""
function bench_case(n::Int = 256; orbit = ORB)
    coord = bench_coord(orbit)
    pair = CoregisteredPair(coord; dt = 518400.0)
    # The fixture's 48x48 window at 500 m; refine the spacing to reach n points inside the swath.
    spacing = 500.0 * 48 / n
    grid = MapGrid(geotransform = (-254500.0, spacing, 0.0, 2191000.0, 0.0, -spacing),
                   size = (n, n), crs = 32632)
    tf = bench_tf()
    win = CartesianIndices((1:n, 1:n))
    sz = (n, n)
    inputs = GeometryInputs(
        dem = fill(500.0, sz),
        dhdx = fill(0.05, sz), dhdy = fill(-0.03, sz),
        vx = fill(20.0, sz), vy = fill(-15.0, sz),
        srx = fill(10.0, sz), sry = fill(10.0, sz),
        csminx = fill(120.0, sz), csminy = fill(120.0, sz),
        csmaxx = fill(480.0, sz), csmaxy = fill(480.0, sz),
        ssm = fill(1.0, sz))
    return (; grid, pair, coord, tf, win, inputs)
end
