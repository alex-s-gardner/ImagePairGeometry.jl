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

_pair_of(f::Function) = f()
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

# `scene` is given for a lattice transform: `InterpolatedTransform` is built *for* a grid and window,
# so deriving a fresh scene from it here would size the lattice against one grid and measure it on
# another. Every mode below therefore runs on the exact path's own scene, which is also what makes the
# per-point times comparable.
function measure(name, tf; scene = nothing)
    s = scene === nothing ? setup(tf) : scene
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
identity_ns = measure("identity (0 calls/pt)", IdentityTransform())
cross_ns = measure("cross-CRS (3 calls/pt)", fast_transform(3413, 32624))

# The lattice modes, against the exact cross-CRS run above. `:hybrid` keeps one call per point, so its
# speedup saturates; `:full` keeps paying as the lattice coarsens. Accuracy per mode and spacing is in
# `docs/interpolated-transform.md`.
println()
let factory = fast_transform(3413, 32624), s = setup(factory)
    for mode in (:hybrid, :full), lattice in (2, 4, 8), kern in (Bilinear(), Bicubic())
        tf = InterpolatedTransform(factory, s.grid, s.pair; lattice, mode,
                                   interpolation = kern, window = s.win)
        ns = measure("$mode lattice=$lattice $(nameof(typeof(kern)))", tf; scene = s)
        @printf("%-24s %.2fx vs exact\n", "", cross_ns / ns)
    end
end

@printf("\ntransform share of a cross-CRS run : %5.1f%%\n",
        100 * (cross_ns - identity_ns) / cross_ns)
@printf("Amdahl ceiling on optimizing everything else : %.2fx\n",
        cross_ns / (cross_ns - identity_ns))
println("""
Reading: the transform dominates, so throughput work belongs in making *fewer* transform calls —
which is what dispatching on `IdentityTransform` does for a same-CRS pair — rather than in tuning the
arithmetic around them. Two candidates were measured against a PROJ-backed transform and rejected,
and neither depends on the backend: batching through `proj_trans_generic` was 1.00x, because the cost
is the projection math rather than the call, and a row-ordered sweep buys nothing over random points.

Interpolating the transform on a coarse lattice is the approach that pays, and `InterpolatedTransform`
is it. `:hybrid` keeps the forward transform exact, so the location bands stay bitwise, and saturates
around 1.6x because the one remaining call is the floor. `:full` removes that floor and keeps paying as
the lattice coarsens, at a one-pixel difference in the location bands. Accuracy per mode, spacing and
band is in `docs/interpolated-transform.md`; the exact path remains the default.""")
