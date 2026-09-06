# The radar path against the compiled reference, on a real acquisition and a real parameter grid.
#
# `radar_geogrid.jl` asserts the same standard on a synthetic circular orbit and synthetic inputs.
# What this adds is everything synthetic data cannot exercise: a perturbed real orbit, a real
# acquisition's range and azimuth scales, and the ITS_LIVE 120 m rasters over Jakobshavn Isbræ with
# their real nodata pattern.
#
# The reference outputs are produced by `test/reference/gen_radar_realdata.py`, which is not committed
# output — the nine GeoTIFFs are ~500 MB. So this needs that script run first and its directory named:
#
#     micromamba run -n geogrid-ref python test/reference/gen_radar_realdata.py \
#         --reference <ref.json> --secondary <sec.json> \
#         --params ~/data/autorift/tests/params --outdir <dir>
#     IPG_REALDATA_DIR=<dir> julia --project=. -e 'using TestEnv; TestEnv.activate();
#                                                  include("test/radar_realdata.jl")'
#
# The exactness standard is the one `REFERENCE.md` sets and `radar_geogrid.jl` already applies, reused
# rather than restated: Tier A bitwise on the integer bands except the azimuth-dependent ones, Tier B
# relative on the floats.

using ImagePairGeometry
using ImagePairGeometry: INT_BANDS, FLOAT_BANDS, reference_files, nodata_from, mapgrid,
                         fast_transform, Orbit, RadarCoordinate, CoregisteredPair,
                         LookLeft, LookRight, incidence_angle
using ArchGDAL
using Dates
using JSON3
using Rasters
using StaticArrays
using Test

# An extension's exports are not brought in by `using ImagePairGeometry`, so its names are reached
# through the module itself — the same way a caller would.
const RA_EXT = Base.get_extension(ImagePairGeometry, :ImagePairGeometryRastersExt)
using .RA_EXT: RasterInputs

const REALDATA_DIR = get(ENV, "IPG_REALDATA_DIR", "")
const PARAMS_DIR = get(ENV, "IPG_PARAMS_DIR",
                       expanduser("~/data/autorift/tests/params"))

# The azimuth-dependent bands, tolerant by exactly the margin `radar_geogrid.jl` documents: the
# reference's azimuth line index carries a ~0.0013-line systematic offset, which flips a point already
# that close to a `std::round` boundary.
const TOLERANT_BANDS = (:location_y, :search_y, :offset_x, :offset_y, :search_x)

hexparse(v) = v isa Number ? Float64(v) : parse(Float64, v.hex)

"""
    band_nodata(path) -> Union{Nothing,Float64}

The nodata value a raster's first band declares, or `nothing` where it declares none.

Read from the file rather than assumed: the radar kernel takes its one missing-value sentinel from the
reference-velocity band, so this value decides which points are computed.
"""
function band_nodata(path::AbstractString)
    return ArchGDAL.read(path) do ds
        v = ArchGDAL.getnodatavalue(ArchGDAL.getband(ds, 1))
        v === nothing ? nothing : Float64(v)
    end
end

"""
    read_reference_band(dir, file, band) -> Matrix

One band of a reference GeoTIFF, as written, with no masking applied.
"""
function read_reference_band(dir, file, band)
    return ArchGDAL.read(joinpath(dir, file)) do ds
        permutedims(ArchGDAL.read(ArchGDAL.getband(ds, band)), (1, 2))
    end
end

# The epoch of a CF units string like "seconds since 2025-10-28T00:00:00".
cf_epoch(units::AbstractString) =
    DateTime(replace(strip(split(units, "since")[2]), " " => "T"))

"""
    coordinate_from_metadata(ref, sec) -> (RadarCoordinate, dt)

The reference acquisition as a coordinate, and the pair's interval in seconds.

Radar geometry comes from image 1 alone (`testGeogrid.py:427-470`); the secondary contributes only its
sensing time. The interval spans both epochs, since two acquisitions weeks apart carry different ones.
"""
function coordinate_from_metadata(ref, sec)
    g, orb = ref.geometry, ref.orbit
    epoch = cf_epoch(orb.epoch)

    time = [hexparse(v) for v in orb.time]
    position = [SVector{3,Float64}(hexparse.(p)...) for p in orb.position]
    velocity = [SVector{3,Float64}(hexparse.(v)...) for v in orb.velocity]
    orbit = Orbit(; time, position, velocity)

    # Seconds from the epoch's own midnight to the epoch: the azimuth index is measured against
    # seconds-since-midnight and the orbit against its epoch, so this is the constant between them.
    offset = (epoch - DateTime(Date(epoch))).value / 1000

    look = lowercase(ref.identification.look_direction) == "left" ? LookLeft : LookRight
    kwargs = (; orbit, starting_range = hexparse(g.starting_range),
              dr = hexparse(g.range_pixel_spacing), sensing_start = hexparse(g.sensing_start),
              prf = hexparse(g.prf), nsamples = g.nsamples, nlines = g.nlines,
              look_side = look, wavelength = hexparse(g.wavelength), orbit_epoch_offset = offset)
    coord = RadarCoordinate(; kwargs..., incidence_angle = incidence_angle(; kwargs...))

    sec_epoch = cf_epoch(sec.orbit.epoch)
    dt = (sec_epoch - epoch).value / 1000 +
         hexparse(sec.geometry.sensing_start) - hexparse(g.sensing_start)
    return coord, dt
end

"""
    realdata_case(dir, params) -> NamedTuple

The grid, pair and inputs of the reference run recorded in `dir/run.json`.

The window comes from the reference's own offsets and counts, so the two sides are band-aligned even
where the footprint's last bits could move an edge — the same choice `radar_geogrid.jl` makes.
"""
function realdata_case(dir::AbstractString, params::AbstractString)
    run = JSON3.read(read(joinpath(dir, "run.json"), String))
    rad = run.radar

    ref_meta = JSON3.read(read(joinpath(dir, "reference_metadata.json"), String))
    sec_meta = JSON3.read(read(joinpath(dir, "secondary_metadata.json"), String))

    # Built from the harvested metadata rather than through SARDatasets.jl, so this package's test
    # suite gains no dependency on a reader. That package's own tests assert its bridge produces the
    # same values.
    coord, dt = coordinate_from_metadata(ref_meta, sec_meta)
    pair = CoregisteredPair(coord; dt)

    # The reference computes the incidence angle before the kernel and stores it, so agreement on it
    # is a precondition of comparing any band rather than something the comparison can absorb.
    want_ia = hexparse(rad.incidence_angle)
    @test coord.incidence_angle ≈ want_ia rtol = 1e-9

    dem = Raster(joinpath(params, "NPS_0120m_h.tif"); lazy = true, missingval = nothing)
    # `mapgrid` carries the DEM's CRS as WKT, which `fast_transform` cannot resolve — it looks a
    # transformation up by code rather than parsing a description. The reference was given the EPSG
    # code, so the grid is rebuilt on the same code it used.
    from_file = mapgrid(dem)
    grid = MapGrid(geotransform = from_file.geotransform, size = from_file.size,
                   crs = Int(run.grid.epsg))

    w = run.window
    win = CartesianIndices((w.pOff + 1:(w.pOff + w.pCount), w.lOff + 1:(w.lOff + w.lCount)))

    param(name) = Raster(joinpath(params, "NPS_0120m_$name.tif"); lazy = true, missingval = nothing)
    src = RasterInputs(dem = dem,
                       dhdx = param("dhdx"), dhdy = param("dhdy"),
                       vx = param("vx"), vy = param("vy"),
                       srx = param("vxSearchRange"), sry = param("vySearchRange"),
                       csminx = param("xMinChipSize"), csminy = param("yMinChipSize"),
                       csmaxx = param("xMaxChipSize"), csmaxy = param("yMaxChipSize"),
                       ssm = param("StableSurface"))

    return (; run, grid, pair, coord, win, src, dem)
end

if isempty(REALDATA_DIR)
    @info """skipping the real-data radar comparison; set IPG_REALDATA_DIR to a directory produced by
             test/reference/gen_radar_realdata.py"""
elseif !isfile(joinpath(REALDATA_DIR, "run.json"))
    error("IPG_REALDATA_DIR=$REALDATA_DIR has no run.json; run " *
          "test/reference/gen_radar_realdata.py into it first")
else
    case = realdata_case(REALDATA_DIR, PARAMS_DIR)

    @testset "the window matches the reference" begin
        w = case.run.window
        @test size(case.win) == (w.pCount, w.lCount)
        # The grid the DEM defines must be the grid the reference read, or every band is offset.
        @test case.grid.size == size(case.dem)
        @test case.grid.geotransform[2] == case.run.grid.spacing
    end

    @testset "the ground pixel sizes match" begin
        # `X_res`/`Y_res` are what the chip-size conversion divides by, so a disagreement here would
        # move every chip-size band without being a chip-size bug.
        @test ImagePairGeometry.xsize(case.coord) ≈ case.run.resolution.X_res rtol = 1e-5
        @test ImagePairGeometry.ysize(case.coord) ≈ case.run.resolution.Y_res rtol = 1e-5
    end

    # The radar path reads its one missing-value sentinel from the **vx** band
    # (`geogridRadar.cpp:509-512`). The ITS_LIVE rasters declare that band's nodata as -32767, where
    # the synthetic fixture in `radar_geogrid.jl` declares one on the DEM only and so is read as 0.0.
    # This is the difference real inputs make: with 0.0 here, every point whose reference velocity is
    # exactly zero — ice-free ground, most of the scene — is treated as missing.
    nodata = nodata_from(band_nodata(joinpath(PARAMS_DIR, "NPS_0120m_vx.tif")))

    # One run, reused by both tiers: the kernel over 4.7M points is not something to repeat per band.
    result = pairgeometry_blocked(case.grid, case.pair, case.src;
                                  transform = () -> fast_transform(case.grid.crs, 4326),
                                  window = case.win, ntasks = 8, nodata)

    @testset "Tier A: the integer bands" begin
        worst = Dict{String,Tuple{Int,Int}}()
        for (file, fields) in reference_files(result)
            fields[1] in INT_BANDS || continue
            isfile(joinpath(REALDATA_DIR, file)) || continue
            for (bi, f) in enumerate(fields)
                want = Int32.(read_reference_band(REALDATA_DIR, file, bi))
                got = getfield(result, f)
                @test size(got) == size(want)

                d = Int.(got) .- Int.(want)
                nbad = count(!=(0), d)
                worst["$file/$f"] = (nbad, isempty(d) ? 0 : maximum(abs, d))

                if f in TOLERANT_BANDS
                    @test maximum(abs, d) <= 1
                    @test nbad <= max(1, length(d) ÷ 50)
                else
                    @test nbad == 0
                end
            end
        end
        off = sort([k => v for (k, v) in worst if v[1] != 0]; by = kv -> -kv[2][1])
        isempty(off) || @info "Tier A on real data, as (points, max index difference)" off
    end

    @testset "Tier B: the float bands" begin
        # Two bounds, because two different things are being measured — the split
        # `radar_geogrid.jl` makes, at the magnitudes real geometry produces.
        #
        # A band dividing by the range spacing `dr` inherits nothing from the azimuth solve and agrees
        # to 1.7e-7. A band dividing by the per-point along-track step carries the reference's
        # ~0.0013-line azimuth offset: that displaces the platform by
        # `|v| * 0.0013 / prf` ≈ 6.5 mm, which against a ~4.97 m along-track step is a few times 1e-4.
        # Measured worst here is 3.5e-4, above the 2e-4 the synthetic fixtures bound — the residual is
        # a property of the acquisition's geometry, and the fixture's orbit is not a real one.
        along_track = (:off2vx_dy, :off2vy_dy, :off2vy_dr)
        bound(f) = f in along_track ? 1e-3 : 1e-6

        per_band = Dict{String,Float64}()
        for (file, fields) in reference_files(result)
            fields[1] in FLOAT_BANDS || continue
            isfile(joinpath(REALDATA_DIR, file)) || continue
            for (bi, f) in enumerate(fields)
                want = read_reference_band(REALDATA_DIR, file, bi)
                got = getfield(result, f)
                @test size(got) == size(want)

                worst = 0.0
                nsentinel = 0
                for k in eachindex(got, want)
                    w, g = Float64(want[k]), got[k]
                    # A point the reference skipped must be skipped here too, and the reverse. Counted
                    # rather than asserted per point, so one testset failure reports every mismatch.
                    if w == -32767.0 || g == -32767.0
                        nsentinel += (w != g)
                        continue
                    end
                    worst = max(worst, abs(g - w) / max(abs(w), 1.0))
                end
                @test nsentinel == 0
                @test worst < bound(f)
                per_band["$file band$bi ($f)"] = worst
            end
        end
        @info "real-data float bands, worst relative difference" sort(collect(per_band); by = last,
                                                                    rev = true)
    end

    @testset "the same points are valid on both sides" begin
        # A run that skipped every point would satisfy every band comparison above, since a sentinel
        # matching a sentinel is agreement. So the count of computed points is asserted against the
        # reference's own, rather than against a threshold: the grid extends well beyond the swath, so
        # about half of it is legitimately outside and the absolute fraction is not the property worth
        # fixing in a test.
        want = read_reference_band(REALDATA_DIR, "window_scale_factor.tif", 1)
        @test nvalid(result) == count(!=(-32767.0), want)
        @info "real-data coverage" nvalid = nvalid(result) npoints = npoints(result) fraction =
            nvalid(result) / npoints(result)
    end
end
