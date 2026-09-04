# Gate: the raster path is the same computation, read from and written to disk.
#
# Two properties matter here and neither is about GDAL. Reading blocks from disk must give the same
# result as computing from resident arrays — otherwise the memory bound costs correctness. And the
# nine written files must carry the band order, types and nodata the reference's consumers expect,
# since a downstream reader built against its output is the reason the names are fixed.

using ImagePairGeometry
using ImagePairGeometry: nodata_from, REFERENCE_FILES
using Rasters
using ArchGDAL
using DimensionalData
using DimensionalData.Lookups: Intervals, Start
using DiskArrays
using Test

# An extension's exports are not brought in by `using ImagePairGeometry`, so its names are reached
# through the module itself — the same way a caller would.
const RA_EXT = Base.get_extension(ImagePairGeometry, :ImagePairGeometryRastersExt)
using .RA_EXT: RasterInputs

"""A raster on the grid `gt` describes, written to `path` so it is disk-backed when read back.

Built through `Rasters.write` rather than raw ArchGDAL calls: writing a dataset with
`ArchGDAL.create` from inside a function loses the geotransform, and a fixture that silently lands
on the default `(0, 1, 0, 0, 0, 1)` would make every geometry assertion here meaningless.

Lookups are `Intervals(Start())`, matching what GDAL produces. `Start` is the cell's *low* edge in
coordinate order, so on the usual north-up axis — negative Y step — the first coordinate is one step
below the geotransform origin rather than equal to it. Building the range from the origin itself
shifts the file by a pixel, which is the same trap `_edge_offset` handles on the reading side.
"""
function write_input(path, arr, gt, epsg)
    nx, ny = size(arr)
    x = X(range(gt[1] + (gt[2] < 0 ? gt[2] : 0.0); step = gt[2], length = nx);
          sampling = Intervals(Start()))
    y = Y(range(gt[4] + (gt[6] < 0 ? gt[6] : 0.0); step = gt[6], length = ny);
          sampling = Intervals(Start()))
    # The return of `Rasters.write` is not the path on every backend, so the path is returned
    # explicitly rather than threaded through.
    Rasters.write(path, Raster(arr, (x, y); crs = Rasters.EPSG(epsg)); force = true)
    isfile(path) || error("write_input did not produce $path")
    return path
end

# A radar coordinate, only ever used to select the three-band output layout — the geometry is never
# computed against it here, so the orbit need only be well-formed.
const RADAR_COORD = let
    R = 7.0e6
    w = sqrt(3.986004418e14 / R^3)
    t = [(i - 1) * 10.0 for i in 1:8]
    pos = [ImagePairGeometry.SVector{3,Float64}(R * cos(w * ti), 0.0, R * sin(w * ti)) for ti in t]
    vel = [ImagePairGeometry.SVector{3,Float64}(-R * w * sin(w * ti), 0.0, R * w * cos(w * ti))
           for ti in t]
    orbit = ImagePairGeometry.Orbit(t[1], 10.0, pos, vel)
    RadarCoordinate(; orbit, starting_range = 8.0e5, dr = 2.33, sensing_start = 10.0, prf = 486.0,
                    nsamples = 1000, nlines = 1000, look_side = LookRight, wavelength = 0.055,
                    incidence_angle = deg2rad(40))
end

const GT = (295000.0, 120.0, 0.0, 7805000.0, 0.0, -120.0)
const EPSG = 32624
const NGRID = 120

"""Inputs written as GeoTIFFs in a fresh directory, with the grid they are on."""
function raster_case(dir)
    isdir(dir) || mkpath(dir)
    n = (NGRID, NGRID)
    # Varied rather than constant, so a transposed read or an off-by-one window shows up.
    xx = [i for i in 1:n[1], _ in 1:n[2]]
    yy = [j for _ in 1:n[1], j in 1:n[2]]
    fields = Dict(
        "dem" => 500.0 .+ 40.0 .* sin.(xx ./ 7) .+ 30.0 .* cos.(yy ./ 5),
        "dhdx" => 0.03 .* cos.(xx ./ 7),
        "dhdy" => -0.02 .* sin.(yy ./ 5),
        "vx" => 120.0 .* sin.(xx ./ 11) .+ 40.0,
        "vy" => -90.0 .* cos.(yy ./ 9),
        "srx" => 200.0 .+ 150.0 .* abs.(sin.(xx ./ 6)),
        "sry" => 150.0 .+ 100.0 .* abs.(cos.(yy ./ 8)),
        "csminx" => fill(240.0, n), "csminy" => fill(360.0, n),
        "csmaxx" => fill(480.0, n), "csmaxy" => fill(720.0, n),
        "ssm" => Float64.((xx .+ yy) .% 3 .== 0),
    )
    paths = Dict(k => write_input(joinpath(dir, k * ".tif"), v, GT, EPSG) for (k, v) in fields)
    return fields, paths
end

load(p) = Raster(p; lazy = true)

@testset "mapgrid and footprint from rasters" begin
    mktempdir() do dir
        _, paths = raster_case(dir)
        dem = load(paths["dem"])

        grid = mapgrid(dem)
        # A GeoTIFF read back is `Intervals{Start}`: its lookups are pixel edges, so they equal the
        # geotransform origin with no adjustment. `mapgrid` reads the locus rather than assuming it,
        # which the in-memory cases below exercise.
        @test grid.geotransform[1] ≈ GT[1]
        @test grid.geotransform[4] ≈ GT[4]
        @test grid.geotransform[2] ≈ GT[2]
        @test grid.geotransform[6] ≈ GT[6]
        @test grid.size == (NGRID, NGRID)

        fp = ImagePairGeometry.image_footprint(dem)
        # A footprint's origin is the first pixel *center*, half a pixel inside the edge.
        @test fp.origin[1] ≈ GT[1] + GT[2] / 2
        @test fp.origin[2] ≈ GT[4] + GT[6] / 2
        @test fp.spacing == (GT[2], GT[6])
        @test fp.size == (NGRID, NGRID)
    end
end

@testset "reading from disk equals computing in memory" begin
    mktempdir() do dir
        fields, paths = raster_case(dir)
        grid = mapgrid(load(paths["dem"]))

        img = ImageFootprint(origin = (300000.0, 7800000.0), spacing = (30.0, -30.0),
                             size = (400, 400))
        pair = coregister(img, img; dt = 91 * 86400.0)
        win = grid_window(grid, footprint_bounds(IdentityTransform(), pair.coordinate))

        mem = pairgeometry(grid, pair,
                           GeometryInputs(dem = fields["dem"][win], dhdx = fields["dhdx"][win],
                                          dhdy = fields["dhdy"][win], vx = fields["vx"][win],
                                          vy = fields["vy"][win], srx = fields["srx"][win],
                                          sry = fields["sry"][win],
                                          csminx = fields["csminx"][win],
                                          csminy = fields["csminy"][win],
                                          csmaxx = fields["csmaxx"][win],
                                          csmaxy = fields["csmaxy"][win], ssm = fields["ssm"][win]);
                           window = win, nodata = nodata_from(-32767.0))

        src = RasterInputs(dem = load(paths["dem"]), dhdx = load(paths["dhdx"]),
                           dhdy = load(paths["dhdy"]), vx = load(paths["vx"]),
                           vy = load(paths["vy"]), srx = load(paths["srx"]),
                           sry = load(paths["sry"]), csminx = load(paths["csminx"]),
                           csminy = load(paths["csminy"]), csmaxx = load(paths["csmaxx"]),
                           csmaxy = load(paths["csmaxy"]), ssm = load(paths["ssm"]))

        # Several block sizes: the read must be window-correct, not merely correct in one tiling.
        for bs in ((16, 16), (37, 11), (512, 512))
            disk = pairgeometry_blocked(grid, pair, src; transform = IdentityTransform(),
                                        window = win, blocksize = bs, ntasks = 1,
                                        nodata = nodata_from(-32767.0))
            @testset "blocksize $bs" begin
                for f in ImagePairGeometry.INT_BANDS
                    @test getfield(mem, f) == getfield(disk, f)
                end
                for f in ImagePairGeometry.FLOAT_BANDS
                    @test reinterpret(UInt64, getfield(mem, f)) ==
                          reinterpret(UInt64, getfield(disk, f))
                end
            end
        end
    end
end

@testset "write_geotiffs round-trips" begin
    mktempdir() do dir
        fields, paths = raster_case(dir)
        grid = mapgrid(load(paths["dem"]))
        img = ImageFootprint(origin = (300000.0, 7800000.0), spacing = (30.0, -30.0),
                             size = (400, 400))
        pair = coregister(img, img; dt = 91 * 86400.0)
        win = grid_window(grid, footprint_bounds(IdentityTransform(), pair.coordinate))
        src = RasterInputs(dem = load(paths["dem"]), dhdx = load(paths["dhdx"]),
                           dhdy = load(paths["dhdy"]), vx = load(paths["vx"]),
                           vy = load(paths["vy"]), srx = load(paths["srx"]),
                           sry = load(paths["sry"]), csminx = load(paths["csminx"]),
                           csminy = load(paths["csminy"]), csmaxx = load(paths["csmaxx"]),
                           csmaxy = load(paths["csmaxy"]), ssm = load(paths["ssm"]))
        r = pairgeometry_blocked(grid, pair, src; transform = IdentityTransform(), window = win,
                                 blocksize = (32, 32), nodata = nodata_from(-32767.0))

        out = mktempdir()
        written = write_geotiffs(out, r)
        @test length(written) == 9      # every band is supported by these inputs
        @test Set(basename.(written)) == Set(first.(REFERENCE_FILES))

        # The projected path writes *two*-band off2vel files. `PairGeometry` carries the radar path's
        # third band as a field, so the layout has to be chosen rather than taken from the struct —
        # and getting it wrong is silent, since a consumer indexes bands positionally.
        @test ImagePairGeometry.reference_files(r) === REFERENCE_FILES
        for f in ("window_rdr_off2vel_x_vec.tif", "window_rdr_off2vel_y_vec.tif")
            ArchGDAL.read(joinpath(out, f)) do ds
                @test ArchGDAL.nraster(ds) == 2
            end
        end

        # A result computed for a radar coordinate selects the three-band layout, so the same writer
        # produces the reference's radar output without being told which path it is on. The layout
        # comes from the coordinate the result carries, not from which bands happen to be filled —
        # inferring it from `off2vx_dr` misclassified a radar result computed without a slope raster,
        # since the branch that fills that band is skipped entirely.
        radar = ImagePairGeometry.PairGeometry(
            (getfield(r, f) for f in ImagePairGeometry.INT_BANDS)...,
            (getfield(r, f) for f in ImagePairGeometry.FLOAT_BANDS)...,
            r.geotransform, r.crs, r.window, r.nodata, RADAR_COORD)
        @test ImagePairGeometry.reference_files(radar) === ImagePairGeometry.RADAR_REFERENCE_FILES
        # And the discriminator really is the coordinate: these bands are all sentinel.
        @test all(==(radar.nodata.output), radar.off2vx_dr)
        rout = mktempdir()
        write_geotiffs(rout, radar)
        for f in ("window_rdr_off2vel_x_vec.tif", "window_rdr_off2vel_y_vec.tif")
            ArchGDAL.read(joinpath(rout, f)) do ds
                @test ArchGDAL.nraster(ds) == 3
            end
        end

        for (filename, bandfields) in REFERENCE_FILES
            path = joinpath(out, filename)
            @test isfile(path)
            ArchGDAL.read(path) do ds
                @test ArchGDAL.width(ds) == size(r)[1]
                @test ArchGDAL.height(ds) == size(r)[2]
                @test ArchGDAL.nraster(ds) == length(bandfields)
                @test collect(ArchGDAL.getgeotransform(ds)) ≈ collect(r.geotransform)

                for (i, f) in enumerate(bandfields)
                    band = ArchGDAL.getband(ds, i)
                    @test ArchGDAL.getnodatavalue(band) == -32767.0
                    want = getfield(r, f)
                    got = ArchGDAL.read(band)
                    @test size(got) == size(want)
                    if eltype(want) === Int32
                        @test Int32.(got) == want
                    else
                        # Float64 on disk is exact, so a round-trip must be bitwise.
                        @test reinterpret(UInt64, Float64.(got)) == reinterpret(UInt64, want)
                    end
                end
            end
        end
    end
end

@testset "an unsupported band writes no file" begin
    # The reference writes nothing for an output its inputs did not support, and a consumer reads the
    # file's absence as "not computed" rather than "computed and empty".
    mktempdir() do dir
        _, paths = raster_case(dir)
        grid = mapgrid(load(paths["dem"]))
        img = ImageFootprint(origin = (300000.0, 7800000.0), spacing = (30.0, -30.0),
                             size = (400, 400))
        pair = coregister(img, img; dt = 91 * 86400.0)
        win = grid_window(grid, footprint_bounds(IdentityTransform(), pair.coordinate))
        bare = pairgeometry_blocked(grid, pair, RasterInputs(dem = load(paths["dem"]));
                                    transform = IdentityTransform(), window = win,
                                    blocksize = (32, 32), nodata = nodata_from(-32767.0))
        out = mktempdir()
        written = write_geotiffs(out, bare)
        @test basename.(written) == ["window_location.tif"]
        @test !isfile(joinpath(out, "window_offset.tif"))
        @test !isfile(joinpath(out, "window_scale_factor.tif"))
    end
end

@testset "blocksize_from_chunks" begin
    mktempdir() do dir
        _, paths = raster_case(dir)
        src = RasterInputs(dem = load(paths["dem"]))
        bs = blocksize_from_chunks(src)
        # Never larger than the grid, and at least the floor even for a striped file — which is what
        # ArchGDAL writes by default, reporting chunks one row tall.
        @test all(bs .<= (NGRID, NGRID))
        @test all(bs .>= 1)
        # Whatever it returns must still give the same answer, which is the only property that
        # matters about a block size.
        grid = mapgrid(load(paths["dem"]))
        img = ImageFootprint(origin = (300000.0, 7800000.0), spacing = (30.0, -30.0),
                             size = (400, 400))
        pair = coregister(img, img; dt = 91 * 86400.0)
        win = grid_window(grid, footprint_bounds(IdentityTransform(), pair.coordinate))
        a = pairgeometry_blocked(grid, pair, src; transform = IdentityTransform(), window = win,
                                 blocksize = bs, nodata = nodata_from(-32767.0))
        b = pairgeometry_blocked(grid, pair, src; transform = IdentityTransform(), window = win,
                                 blocksize = (8, 8), nodata = nodata_from(-32767.0))
        @test a.location_x == b.location_x
    end

    # An in-memory raster has no chunking to align to.
    @test blocksize_from_chunks(RasterInputs(dem = Raster(rand(10, 10), (X(1:10), Y(1:10))))) ==
          ImagePairGeometry.DEFAULT_BLOCKSIZE
end

@testset "mismatched input sizes are refused" begin
    mktempdir() do dir
        _, paths = raster_case(dir)
        small = write_input(joinpath(dir, "small.tif"), fill(1.0, 10, 10), GT, EPSG)
        @test_throws DimensionMismatch RasterInputs(dem = load(paths["dem"]),
                                                    dhdx = load(small), dhdy = load(small))
    end
end

@testset "the geotransform survives a round-trip in both axis directions" begin
    # `Intervals{Start}` names the *low* edge of a cell in coordinate order, so on a north-up raster —
    # negative Y step, reverse-ordered axis — it is the edge furthest from the geotransform origin.
    # Reading it as "no adjustment" gets X right and Y wrong by exactly one pixel, which a
    # same-CRS run would then carry into every output.
    mktempdir() do dir
        for gt in ((295000.0, 120.0, 0.0, 7805000.0, 0.0, -120.0),   # north-up, the usual case
                   (295000.0, 120.0, 0.0, 7800000.0, 0.0, 120.0),    # south-up
                   (-2000000.0, 240.0, 0.0, 1500000.0, 0.0, -240.0)) # polar stereographic
            path = write_input(joinpath(dir, "gt.tif"), fill(1.0, 8, 6), gt, EPSG)
            grid = mapgrid(Raster(path; lazy = true))
            @test collect(grid.geotransform) ≈ collect(gt)
            @test grid.size == (8, 6)

            # And a footprint's origin is the first pixel center, half a step inside the origin.
            fp = ImagePairGeometry.image_footprint(Raster(path; lazy = true))
            @test fp.origin[1] ≈ gt[1] + gt[2] / 2
            @test fp.origin[2] ≈ gt[4] + gt[6] / 2
        end
    end
end

@testset "a file's geotransform is read, not inferred from its sampling" begin
    # Rasters sets a lookup's sampling from the `AREA_OR_POINT` tag — `Points` for `Point`,
    # `Intervals{Start}` for `Area` — while reporting the same coordinates either way, namely the
    # geotransform's. So the tag must not change the grid: both the ITS_LIVE parameter rasters and
    # the Landsat scenes carry `AREA_OR_POINT=Point`, and inferring "point means center" from that
    # takes pixel edges for centers and shifts the whole grid half a pixel.
    gt = (-309247.5, 120.0, 0.0, -2084872.5, 0.0, -120.0)
    mktempdir() do dir
        results = Dict{String,Any}()
        for tag in ("Area", "Point")
            # Written through the same helper as every other case here, then retagged in place, so
            # the two files differ in exactly the one metadata item under test.
            path = write_input(joinpath(dir, "aop_$tag.tif"), fill(1.0, 6, 5), gt, EPSG)
            let ds = ArchGDAL.GDAL.gdalopen(path, ArchGDAL.GDAL.GA_Update)
                ArchGDAL.GDAL.gdalsetmetadataitem(ds, "AREA_OR_POINT", tag, C_NULL)
                ArchGDAL.GDAL.gdalclose(ds)
            end
            r = Raster(path; lazy = true)
            grid = mapgrid(r)
            fp = ImagePairGeometry.image_footprint(r)
            @test collect(grid.geotransform) ≈ collect(gt)
            @test fp.origin[1] ≈ gt[1] + gt[2] / 2
            @test fp.origin[2] ≈ gt[4] + gt[6] / 2
            results[tag] = (; grid, fp)
        end
        @test results["Area"].grid.geotransform == results["Point"].grid.geotransform
        @test results["Area"].fp.origin == results["Point"].fp.origin
    end
end

@testset "an in-memory raster's locus is read, not assumed" begin
    # A raster built in memory is `Points` sampling, whose coordinates are cell centers — so the
    # geotransform origin is half a pixel outside them.
    r = Raster(rand(4, 3), (X(10.0:2.0:16.0), Y(100.0:-5.0:90.0)))
    grid = mapgrid(r)
    @test grid.geotransform[1] ≈ 10.0 - 1.0      # center 10.0, step 2.0
    @test grid.geotransform[2] ≈ 2.0
    @test grid.geotransform[6] ≈ -5.0
    # A point lookup's coordinates are already centers, so a footprint takes them unchanged.
    fp = ImagePairGeometry.image_footprint(r)
    @test fp.origin[1] ≈ 10.0
    @test fp.origin[2] ≈ 100.0
end

@testset "an irregular axis is refused" begin
    # Without a constant step there is no single pixel size, and the reference's index arithmetic
    # assumes one.
    r = Raster(rand(5, 5), (X(Rasters.Sampled([1.0, 2.0, 4.0, 8.0, 16.0];
                                               span = Rasters.Irregular(1.0, 16.0),
                                               sampling = Intervals(Start()))), Y(1.0:5.0)))
    @test_throws "not regularly spaced" mapgrid(r)
    @test_throws "not regularly spaced" ImagePairGeometry.image_footprint(r)
end
