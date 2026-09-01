# Peak memory against block size.
#
# Blocking exists to bound memory rather than to go faster, so this is the measurement that says
# whether it works. The source here synthesizes each block on demand — standing in for a disk-backed
# read — so a small block genuinely never holds the whole grid, which a source viewing resident
# arrays could not show.
using ImagePairGeometry
using ImagePairGeometry: nodata_from, AbstractInputSource
using Proj, Printf
const ProjExt = Base.get_extension(ImagePairGeometry, :ImagePairGeometryProjExt)
using .ProjExt: ProjTransformFactory

const COORD = ProjectedCoordinate(origin=(300000.0,7800000.0), spacing=(30.0,-30.0), size=(8000,8000))
const GRID = let p = ProjTransformFactory(3413,32624)()
    b = footprint_bounds(p, COORD); sp=120.0
    x0=floor(b.X[1]/sp-4)*sp; y1=ceil(b.Y[2]/sp+4)*sp
    nx=Int(ceil((b.X[2]-x0)/sp))+8; ny=Int(ceil((y1-b.Y[1])/sp))+8
    MapGrid(geotransform=(x0,sp,0.0,y1,0.0,-sp), size=(nx,ny), crs=3413)
end
const FAC = ProjTransformFactory(3413,32624)
const WIN = grid_window(GRID, footprint_bounds(FAC(), COORD))
const FP = ImageFootprint(origin=COORD.origin, spacing=COORD.spacing, size=COORD.size)
const PAIR = coregister(FP, FP; dt=91*86400.0)

# Stands in for a disk source: synthesizes each block on demand, never holding the whole grid.
struct Synth <: AbstractInputSource end
function ImagePairGeometry.readblock(::Synth, b::CartesianIndices{2})
    sz = size(b); f(v) = fill(v, sz)
    GeometryInputs(dem=f(500.0), dhdx=f(0.02), dhdy=f(-0.01), vx=f(120.0), vy=f(-80.0),
        srx=f(400.0), sry=f(300.0), csminx=f(240.0), csminy=f(240.0),
        csmaxx=f(480.0), csmaxy=f(480.0), ssm=f(1.0))
end
rss() = Sys.maxrss() / 2^20
@printf("window %d x %d = %.2f M points\n", size(WIN)..., length(WIN)/1e6)
@printf("11 input rasters materialized would be %.0f MiB\n", 11*length(WIN)*8/2^20)
pairgeometry_blocked(GRID, PAIR, Synth(); transform=FAC, window=WIN, blocksize=(256,256), ntasks=1, nodata=nodata_from(-32767.0))
GC.gc(); base = rss()
@printf("\n%-12s %12s\n", "blocksize", "peak RSS MiB")
for bs in ((128,128),(256,256),(512,512),(2048,2048))
    GC.gc()
    pairgeometry_blocked(GRID, PAIR, Synth(); transform=FAC, window=WIN, blocksize=bs, ntasks=1, nodata=nodata_from(-32767.0))
    @printf("%-12s %12.0f\n", string(bs), rss())
end
