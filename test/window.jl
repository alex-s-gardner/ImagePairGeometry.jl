# Gate: `footprint_bounds` and `grid_window` match the reference.
#
# The window itself — `pOff`, `lOff`, `pCount`, `lCount` — is compared exactly in every case. It
# feeds every later output, and one point of shift displaces all of them.
#
# The bounding box behind it is compared bitwise only where no reprojection happens. PROJ does not
# promise bit-identical results across platforms: the same PROJ 9.8.1 returns an easting differing
# by 2 ULP on x86-64 Linux and Windows from what it returns on aarch64 macOS, since the projection
# formulae compile to different orderings and libm's transcendentals differ. So cross-CRS bounds get
# a tolerance, and the exact window assertion is what actually guards the arithmetic: the bounds sit
# far from a `floor`/`ceil` boundary, so a few ULP cannot move a window edge, while a genuine error
# in the transcription would move it by whole points.

using ImagePairGeometry
import GeoFormatTypes as GFT
import GeoInterface
using ImagePairGeometry: gridspacing, gridorigin, window_geotransform, gridpoint_center,
                         DEFAULT_ZRANGE
using Extents: Extent
using JSON3
using Proj
using Test

const WFIX = JSON3.read(read(joinpath(@__DIR__, "reference", "window.json"), String))

bits(x::Float64) = reinterpret(UInt64, x)

"""Transform from image CRS to grid CRS, built the way the reference builds it.

`always_xy = false`: the reference uses `osr.CoordinateTransformation` on SRSs from
`ImportFromEPSG`, i.e. authority axis order. For projected-to-projected this is the same as
`always_xy = true`, but matching the reference's construction is what is being tested.
"""
crs_transform(from::Int, to::Int) = Proj.Transformation("EPSG:$from", "EPSG:$to")

image_coord(d) = ProjectedCoordinate(
    origin = (Float64(d.origin[1]), Float64(d.origin[2])),
    spacing = (Float64(d.spacing[1]), Float64(d.spacing[2])),
    size = (Int(d.size[1]), Int(d.size[2])))

grid_of(d) = MapGrid(
    geotransform = ntuple(i -> Float64(d.geotransform[i]), 6),
    size = (Int(d.size[1]), Int(d.size[2])),
    crs = Int(d.epsg))

@testset "provenance" begin
    p = WFIX.provenance
    @test p.proj_version == "9.8.1"
    @test collect(p.zrange) == [-200, 4000]
    @test DEFAULT_ZRANGE == (-200.0, 4000.0)
end

@testset "matches reference: $(c.name)" for c in WFIX.cases
    coord = image_coord(c.image)
    grid = grid_of(c.dem)
    tf = crs_transform(Int(c.image.epsg), Int(c.dem.epsg))

    bounds = footprint_bounds(tf, coord)
    e = c.expect

    xlim = parse.(Float64, collect(e.xlim_hex))
    ylim = parse.(Float64, collect(e.ylim_hex))
    got = (bounds.X..., bounds.Y...)
    want = (xlim..., ylim...)

    if Int(c.image.epsg) == Int(c.dem.epsg)
        # No reprojection: the bounds are the image corners rearranged, so they are reproducible
        # bit for bit and any difference is an arithmetic error.
        @test all(bits.(got) .== bits.(want))
    else
        # A reprojection, so PROJ's last bits are platform-dependent. Bounded relative to the
        # magnitude rather than bitwise, at a tolerance far below one grid step.
        @test all(abs.(got .- want) .<= 1e-9 .* max.(abs.(want), 1.0))
    end

    # What the bounds are actually for. `grid_window` floors and ceils them, so this is the
    # assertion that a transcription error cannot pass — and it holds on every platform, which is
    # the evidence that the tolerance above is not hiding one.
    window = grid_window(grid, bounds)
    pOff, lOff = first(window).I .- 1
    @test pOff == Int(e.pOff)
    @test lOff == Int(e.lOff)
    @test size(window) == (Int(e.pCount), Int(e.lCount))
end

@testset "rejects the empty window the reference passes to GDAL: $(c.name)" for c in WFIX.fail_cases
    coord = image_coord(c.image)
    grid = grid_of(c.dem)
    tf = crs_transform(Int(c.image.epsg), Int(c.dem.epsg))
    bounds = footprint_bounds(tf, coord)

    # The fixture records the reference computing a negative count here.
    @test Int(c.expect.pCount) < 0 || Int(c.expect.lCount) < 0
    @test_throws "does not overlap the grid" grid_window(grid, bounds)
end

@testset "identity transform gives the image's own bounds" begin
    # With no reprojection the bounds are the corner pixel centers exactly, and the elevation
    # range cannot perturb them.
    coord = ProjectedCoordinate(origin = (300000.0, 7800000.0), spacing = (30.0, -30.0),
                                size = (100, 200))
    id = (x, y, z) -> (x, y, z)
    b = footprint_bounds(id, coord)
    @test b.X == (300000.0, 300000.0 + 99 * 30.0)
    @test b.Y == (7800000.0 - 199 * 30.0, 7800000.0)

    # Widening zrange changes nothing under an identity transform.
    @test footprint_bounds(id, coord; zrange = (-5000.0, 9000.0)) == b
end

@testset "zrange affects the box only under a datum shift" begin
    # Elevation changes a horizontal coordinate only when the transform includes a 3D Helmert
    # datum shift. Between two CRSs on the same datum the pipeline passes z through and the
    # horizontal result is bit-identical at any height — so for every ITS_LIVE projection (UTM on
    # WGS84, EPSG:3413, EPSG:3031) `zrange` is inert, and the reference's `[-200, 4000]` widens
    # nothing. Asserted rather than noted, because a PROJ change that made z matter here would
    # silently move every window.
    coord = ProjectedCoordinate(origin = (300000.0, 7800000.0), spacing = (30.0, -30.0),
                                size = (2000, 2000))
    for (from, to) in ((32624, 3413), (32719, 3031), (32624, 4326))
        tf = crs_transform(from, to)
        flat = footprint_bounds(tf, coord; zrange = (0.0, 0.0))
        ref = footprint_bounds(tf, coord; zrange = DEFAULT_ZRANGE)
        wider = footprint_bounds(tf, coord; zrange = (-9000.0, 20000.0))
        @test bits(ref.X[1]) == bits(flat.X[1]) && bits(ref.X[2]) == bits(flat.X[2])
        @test bits(ref.Y[1]) == bits(flat.Y[1]) && bits(ref.Y[2]) == bits(flat.Y[2])
        @test wider == ref
    end

    # Across datums it does matter, which is why the elevation range is carried at all.
    shifted = ProjectedCoordinate(origin = (400000.0, 400000.0), spacing = (30.0, -30.0),
                                  size = (500, 500))
    tf = crs_transform(27700, 3413)     # OSGB36 to WGS84 polar stereographic
    flat = footprint_bounds(tf, shifted; zrange = (0.0, 0.0))
    ref = footprint_bounds(tf, shifted; zrange = DEFAULT_ZRANGE)
    @test ref != flat
    @test ref.X[1] <= flat.X[1] && ref.X[2] >= flat.X[2]
    @test ref.Y[1] <= flat.Y[1] && ref.Y[2] >= flat.Y[2]
end

@testset "window excludes the grid's last row and column" begin
    # The reference clamps to `demXSize - 1.`, so the final point is never processed. Reproduced
    # deliberately; see REFERENCE.md.
    grid = MapGrid(geotransform = (0.0, 1.0, 0.0, 100.0, 0.0, -1.0), size = (100, 100))
    huge = Extent(X = (-1e6, 1e6), Y = (-1e6, 1e6))
    w = grid_window(grid, huge)
    @test first(w).I == (1, 1)
    @test last(w).I == (99, 99)     # not (100, 100)
    @test size(w) == (99, 99)
end

@testset "window geometry helpers" begin
    grid = MapGrid(geotransform = (1000.0, 10.0, 0.0, 5000.0, 0.0, -10.0), size = (50, 60),
                   crs = 32624)
    @test gridorigin(grid) == (1000.0, 5000.0)
    @test gridspacing(grid) == (10.0, -10.0)

    # Point centers sit half a pixel inside the origin.
    @test gridpoint_center(grid, 1, 1) == (1005.0, 4995.0)
    @test gridpoint_center(grid, 2, 3) == (1015.0, 4975.0)

    w = CartesianIndices((6:20, 11:30))
    gt = window_geotransform(grid, w)
    @test gt == (1050.0, 10.0, 0.0, 4900.0, 0.0, -10.0)
    # The window's own origin reproduces the same first point center.
    @test (gt[1] + 0.5 * gt[2], gt[4] + 0.5 * gt[6]) == gridpoint_center(grid, 6, 11)
end

@testset "MapGrid validation" begin
    @test_throws "size must be positive" MapGrid((0.0, 1.0, 0.0, 0.0, 0.0, -1.0), (0, 5))
    @test_throws "zero rotation" MapGrid((0.0, 1.0, 0.5, 0.0, 0.0, -1.0), (5, 5))
    @test_throws "zero rotation" MapGrid((0.0, 1.0, 0.0, 0.0, 0.5, -1.0), (5, 5))
    @test_throws "pixel size must be nonzero" MapGrid((0.0, 0.0, 0.0, 0.0, 0.0, -1.0), (5, 5))
    @test_throws "must be finite" MapGrid((NaN, 1.0, 0.0, 0.0, 0.0, -1.0), (5, 5))
end

@testset "non-finite transform result fails fast" begin
    coord = ProjectedCoordinate(origin = (0.0, 0.0), spacing = (1.0, -1.0), size = (10, 10))
    @test_throws "outside the transform's area of validity" footprint_bounds(
        (x, y, z) -> (NaN, NaN, NaN), coord)
end

@testset "type stable" begin
    coord = ProjectedCoordinate(origin = (0.0, 100.0), spacing = (1.0, -1.0), size = (10, 10))
    grid = MapGrid(geotransform = (-5.0, 1.0, 0.0, 105.0, 0.0, -1.0), size = (30, 30))
    id = (x, y, z) -> (x, y, z)
    @test @inferred(footprint_bounds(id, coord)) isa Extent
    @test @inferred(grid_window(grid, footprint_bounds(id, coord))) isa CartesianIndices{2}
end

@testset "the CRS follows the GeoJulia convention" begin
    # `GeoInterface.crs` rather than the field, and a `GeoFormatTypes.GeoFormat` rather than a bare
    # integer — so a consumer reads it the same way here as from a `Raster`, and the Rasters extension
    # has one type to write rather than a conversion of its own.
    gt = (0.0, 10.0, 0.0, 100.0, 0.0, -10.0)
    g = MapGrid(geotransform = gt, size = (10, 10), crs = 32624)
    @test GeoInterface.crs(g) === GFT.EPSG(32624)
    @test GeoInterface.crs(g) isa GFT.GeoFormat

    # An EPSG integer and the `GeoFormat` it denotes are the same thing to the constructor.
    @test GeoInterface.crs(MapGrid(geotransform = gt, size = (10, 10), crs = GFT.EPSG(32624))) ===
          GeoInterface.crs(g)

    # A WKT or PROJ string is already a `GeoFormat` and passes through untouched.
    wkt = GFT.WellKnownText(GFT.CRS(), "PROJCS[\"dummy\"]")
    @test GeoInterface.crs(MapGrid(geotransform = gt, size = (10, 10), crs = wkt)) === wkt

    @test GeoInterface.crs(MapGrid(geotransform = gt, size = (10, 10))) === nothing

    # Anything else is refused rather than stored and misinterpreted downstream.
    @test_throws "crs must be a GeoFormatTypes.GeoFormat" MapGrid(geotransform = gt,
                                                                  size = (10, 10),
                                                                  crs = "EPSG:32624")

    # A result inherits its grid's, and reads back the same way.
    a = ImageFootprint(origin = (5.0, 95.0), spacing = (5.0, -5.0), size = (16, 16))
    pair = coregister(a, a; dt = 86400.0)
    win = grid_window(g, footprint_bounds(IdentityTransform(), pair.coordinate))
    r = pairgeometry(g, pair, GeometryInputs(dem = zeros(size(win))); window = win)
    @test GeoInterface.crs(r) === GeoInterface.crs(g)
end
