# Throughput and peak memory of the blocked driver.
#
# Blocking exists to bound memory, and threading to use the cores a cross-CRS run needs: at hundreds
# of nanoseconds per point, a continental grid is hours of single-threaded work.
#
# `fast_transform` is immutable and holds no library state, so one object serves every task and this
# passes it directly rather than a factory. A transform that must be built per task — a PROJ pipeline,
# whose context PROJ documents as usable from one thread at a time — is what the factory form exists
# for, and `test/blocks.jl` exercises that path.

using ImagePairGeometry
using ImagePairGeometry: nodata_from
using Printf

const COORD = ProjectedCoordinate(origin = (300000.0, 7800000.0), spacing = (30.0, -30.0),
                                  size = (8000, 8000))
const TF = fast_transform(3413, 32624)

# The grid is derived from the footprint so the window is square: a window narrower than a block in
# one direction would make the block-size comparison meaningless.
const GRID = let probe = TF
    b = footprint_bounds(probe, COORD)
    sp = 120.0
    x0 = floor(b.X[1] / sp - 4) * sp
    y1 = ceil(b.Y[2] / sp + 4) * sp
    nx = Int(ceil((b.X[2] - x0) / sp)) + 8
    ny = Int(ceil((y1 - b.Y[1]) / sp)) + 8
    MapGrid(geotransform = (x0, sp, 0.0, y1, 0.0, -sp), size = (nx, ny), crs = 3413)
end
const PAIR = coregister(ImageFootprint(origin = COORD.origin, spacing = COORD.spacing,
                                       size = COORD.size),
                        ImageFootprint(origin = COORD.origin, spacing = COORD.spacing,
                                       size = COORD.size); dt = 91 * 86400.0)

const WIN = grid_window(GRID, footprint_bounds(TF, COORD))

function make_inputs(win)
    sz = size(win)
    f(v) = fill(v, sz)
    GeometryInputs(dem = f(500.0), dhdx = f(0.02), dhdy = f(-0.01),
                   vx = f(120.0), vy = f(-80.0), srx = f(400.0), sry = f(300.0),
                   csminx = f(240.0), csminy = f(240.0), csmaxx = f(480.0), csmaxy = f(480.0),
                   ssm = f(1.0))
end

const INPUTS = make_inputs(WIN)
const SRC = InMemoryInputs(INPUTS, WIN)
const NPTS = length(WIN)

@printf("grid window: %d x %d = %d points, EPSG:3413 grid against UTM 24N imagery\n\n",
        size(WIN)..., NPTS)

# Warm up so the numbers are not compilation.
pairgeometry_blocked(GRID, PAIR, SRC; transform = TF, window = WIN,
                     blocksize = (256, 256), ntasks = 1, nodata = nodata_from(-32767.0))

@printf("%-10s %-8s %12s %12s %10s\n", "blocksize", "ntasks", "seconds", "ns/point", "speedup")
base = Ref(0.0)
for bs in ((128, 128), (256, 256), (512, 512))
    for nt in (1, 2, 4, min(8, Threads.nthreads()))
        nt > Threads.nthreads() && continue
        t = @elapsed pairgeometry_blocked(GRID, PAIR, SRC; transform = TF, window = WIN,
                                          blocksize = bs, ntasks = nt,
                                          nodata = nodata_from(-32767.0))
        bs == (128, 128) && nt == 1 && (base[] = t)
        @printf("%-10s %-8d %12.3f %12.1f %9.2fx\n", string(bs), nt, t, 1e9 * t / NPTS,
                base[] / t)
    end
end

@printf("\nthreads available: %d\n", Threads.nthreads())
