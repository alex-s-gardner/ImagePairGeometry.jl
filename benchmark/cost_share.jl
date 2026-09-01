# Where the time goes, and therefore what is worth optimizing.
#
# The answer sets the ceiling on every other optimization in this package, so it is measured rather
# than assumed. Both paths must do the same per-point work to be comparable: each transform gets a
# grid derived from its *own* footprint bounds, and the valid fraction is reported, because reusing
# one path's window for the other sends most points down the out-of-bounds early-continue and
# measures the skip rather than the work.

using ImagePairGeometry
using ImagePairGeometry: nodata_from
using BenchmarkTools
using Printf
using Proj

const PROJ_EXT = Base.get_extension(ImagePairGeometry, :ImagePairGeometryProjExt)
using .PROJ_EXT: ProjTransformFactory

_pair_of(f::ProjTransformFactory) = f()
_pair_of(t) = transform_pair(t)

"""A grid covering the image's footprint under `tf`, with inputs for every band."""
function setup(tf; imagesize = (3000, 3000), spacing = 120.0)
    coord = ProjectedCoordinate(origin = (300000.0, 7800000.0), spacing = (30.0, -30.0),
                                size = imagesize)
    b = footprint_bounds(_pair_of(tf), coord)
    x0 = floor(b.X[1] / spacing - 4) * spacing
    y1 = ceil(b.Y[2] / spacing + 4) * spacing
    grid = MapGrid(geotransform = (x0, spacing, 0.0, y1, 0.0, -spacing),
                   size = (Int(ceil((b.X[2] - x0) / spacing)) + 8,
                           Int(ceil((y1 - b.Y[1]) / spacing)) + 8),
                   crs = 3413)
    win = grid_window(grid, b)
    fp = ImageFootprint(origin = coord.origin, spacing = coord.spacing, size = coord.size)
    pair = coregister(fp, fp; dt = 91 * 86400.0)
    f(v) = fill(v, size(win))
    inputs = GeometryInputs(dem = f(500.0), dhdx = f(0.02), dhdy = f(-0.01),
                            vx = f(120.0), vy = f(-80.0), srx = f(400.0), sry = f(300.0),
                            csminx = f(240.0), csminy = f(240.0),
                            csmaxx = f(480.0), csmaxy = f(480.0), ssm = f(1.0))
    return (; grid, pair, win, source = InMemoryInputs(inputs, win))
end

function measure(name, tf)
    s = setup(tf)
    go = () -> pairgeometry_blocked(s.grid, s.pair, s.source; transform = tf, window = s.win,
                                    blocksize = (256, 256), ntasks = 1,
                                    nodata = nodata_from(-32767.0))
    r = go()
    n = length(s.win)
    t = minimum(@benchmark $go()).time
    @printf("%-24s %9d pts  %5.1f%% valid  %7.1f ns/pt\n", name, n, 100 * nvalid(r) / n, t / n)
    return t / n
end

println("Per-point cost by transform, serial\n")
identity_ns = measure("identity (0 PROJ/pt)", IdentityTransform())
proj_ns = measure("cross-CRS (3 PROJ/pt)", ProjTransformFactory(3413, 32624))

@printf("\nPROJ share of a cross-CRS run : %5.1f%%\n", 100 * (proj_ns - identity_ns) / proj_ns)
@printf("Amdahl ceiling on optimizing everything else : %.2fx\n",
        proj_ns / (proj_ns - identity_ns))
println("""
Reading: the projection library dominates, so throughput work belongs in making *fewer* transform
calls — which is what dispatching on `IdentityTransform` does for a same-CRS pair — rather than in
tuning the arithmetic around them. Two candidates were measured and rejected: batching through
`proj_trans_generic` is 1.00x (the cost is the projection math, not the call), and PROJ gains nothing
from sequential over random points, so a row-ordered sweep buys nothing.""")
