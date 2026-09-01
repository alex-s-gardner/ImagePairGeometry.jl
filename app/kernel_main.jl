# A trim-compilable entry point exercising the kernel: no IO, no PROJ, no Rasters.
using ImagePairGeometry
using ImagePairGeometry: nodata_from

function (@main)(_)
    grid = MapGrid(geotransform = (0.0, 120.0, 0.0, 12000.0, 0.0, -120.0), size = (100, 100))
    fp = ImageFootprint(origin = (60.0, 11940.0), spacing = (30.0, -30.0), size = (396, 396))
    pair = coregister(fp, fp; dt = 91 * 86400.0)
    win = grid_window(grid, footprint_bounds(IdentityTransform(), pair.coordinate))
    n = size(win)
    inputs = GeometryInputs(dem = fill(500.0, n), dhdx = fill(0.02, n), dhdy = fill(-0.01, n),
                            vx = fill(120.0, n), vy = fill(-80.0, n))
    r = pairgeometry(grid, pair, inputs; window = win, nodata = nodata_from(-32767.0))
    Core.println(Core.stdout, "valid points: ", nvalid(r))
    return 0
end
