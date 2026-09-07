# The radar geometry against a delivered ITS_LIVE product.
#
# `radar_realdata.jl` compares against a reference this repository runs itself, with inputs it chooses.
# This compares against a product ASF DAAC produced and published — different machine, different
# parameter set, no involvement from here — which is the one check that cannot be arranged to pass.
#
# The delivered NetCDF carries no geogrid band directly. What it carries is `M11` and `M12`, which
# `netcdf_output.py:1107,1137` derives from three of the nine outputs:
#
#     M11 =  offset2vy_2 / (offset2vx_1 * offset2vy_2 - offset2vx_2 * offset2vy_1) / scale_factor_1
#     M12 = -offset2vx_2 / (offset2vx_1 * offset2vy_2 - offset2vx_2 * offset2vy_1) / scale_factor_1
#
# and `dr_to_vr_factor`, which `netcdf_output.py:293` takes as the median of `offset2vr` — band 3 of the
# off2vel file. Recomputing all three from this package's own output and comparing is therefore an
# end-to-end check of the operator and the scale factors against production.
#
# Needs the ITS_LIVE product and the parameter rasters on disk, neither committed:
#
#     IPG_ITSLIVE_PRODUCT=<pair>.nc julia --project=. -e 'using TestEnv; TestEnv.activate();
#                                                        include("test/radar_itslive_product.jl")'

using ImagePairGeometry
using ImagePairGeometry: nodata_from, mapgrid, fast_transform, Orbit, RadarCoordinate,
                         CoregisteredPair, LookLeft, LookRight, incidence_angle,
                         window_geotransform
using ArchGDAL
using Dates
using JSON3
using Rasters
using StaticArrays
using Test

const RA_EXT = Base.get_extension(ImagePairGeometry, :ImagePairGeometryRastersExt)
using .RA_EXT: RasterInputs

const PRODUCT = get(ENV, "IPG_ITSLIVE_PRODUCT", "")
const META_DIR = get(ENV, "IPG_METADATA_DIR", "")
const PARAMS = get(ENV, "IPG_PARAMS_DIR", expanduser("~/data/autorift/tests/params"))

# The sentinel every band of every geogrid output uses.
const SENTINEL = -32767.0

hexparse(v) = v isa Number ? Float64(v) : parse(Float64, v.hex)
cf_epoch(units::AbstractString) = DateTime(replace(strip(split(units, "since")[2]), " " => "T"))

"""
    read_product(path) -> (M11, M12, geotransform)

The two conversion-matrix bands of a delivered product, and its geotransform.

Read through GDAL's NetCDF driver rather than a NetCDF library, so the geotransform comes back as pixel
**edges**. The file's own `x`/`y` coordinate variables are pixel *centres*; aligning against those
without the half-pixel shift lands one pixel out, which reads as a few-per-mille disagreement
concentrated where the terrain is steep rather than as an indexing error.
"""
function read_product(path::AbstractString)
    band(v) = ArchGDAL.read("NETCDF:\"$path\":$v") do ds
        (permutedims(ArchGDAL.read(ds, 1), (1, 2)), ArchGDAL.getgeotransform(ds))
    end
    m11, gt = band("M11")
    m12, _ = band("M12")
    return m11, m12, gt
end

if isempty(PRODUCT)
    @info """skipping the ITS_LIVE product cross-check; set IPG_ITSLIVE_PRODUCT to a delivered
             NISAR pair NetCDF"""
elseif isempty(META_DIR)
    @info """skipping the ITS_LIVE product cross-check; set IPG_METADATA_DIR to a directory holding
             reference_metadata.json and secondary_metadata.json for that pair"""
else
    ref = JSON3.read(read(joinpath(META_DIR, "reference_metadata.json"), String))
    sec = JSON3.read(read(joinpath(META_DIR, "secondary_metadata.json"), String))

    # The acquisition, from the harvested metadata of the same granules the product names.
    g, orb = ref.geometry, ref.orbit
    epoch = cf_epoch(orb.epoch)
    orbit = Orbit(; time = [hexparse(v) for v in orb.time],
                  position = [SVector{3,Float64}(hexparse.(p)...) for p in orb.position],
                  velocity = [SVector{3,Float64}(hexparse.(v)...) for v in orb.velocity])
    look = lowercase(ref.identification.look_direction) == "left" ? LookLeft : LookRight
    kw = (; orbit, starting_range = hexparse(g.starting_range),
          dr = hexparse(g.range_pixel_spacing), sensing_start = hexparse(g.sensing_start),
          prf = hexparse(g.prf), nsamples = g.nsamples, nlines = g.nlines, look_side = look,
          wavelength = hexparse(g.wavelength),
          orbit_epoch_offset = (epoch - DateTime(Date(epoch))).value / 1000)
    coord = RadarCoordinate(; kw..., incidence_angle = incidence_angle(; kw...))
    dt = (cf_epoch(sec.orbit.epoch) - epoch).value / 1000 +
         hexparse(sec.geometry.sensing_start) - hexparse(g.sensing_start)
    pair = CoregisteredPair(coord; dt)

    dem = Raster(joinpath(PARAMS, "NPS_0120m_h.tif"); lazy = true, missingval = nothing)
    from_file = mapgrid(dem)
    grid = MapGrid(geotransform = from_file.geotransform, size = from_file.size, crs = 3413)
    param(n) = Raster(joinpath(PARAMS, "NPS_0120m_$n.tif"); lazy = true, missingval = nothing)
    src = RasterInputs(dem = dem, dhdx = param("dhdx"), dhdy = param("dhdy"),
                       vx = param("vx"), vy = param("vy"),
                       srx = param("vxSearchRange"), sry = param("vySearchRange"),
                       csminx = param("xMinChipSize"), csminy = param("yMinChipSize"),
                       csmaxx = param("xMaxChipSize"), csmaxy = param("yMaxChipSize"),
                       ssm = param("StableSurface"))

    # The window comes from the run record rather than from `footprint_bounds`, so the comparison is
    # band-aligned with the reference's own offsets — the same choice `radar_realdata.jl` makes. The
    # default window starts at the DEM's own corner, which is a couple of hundred rows away.
    run = JSON3.read(read(joinpath(META_DIR, "run.json"), String))
    w = run.window
    win = CartesianIndices((w.pOff + 1:(w.pOff + w.pCount), w.lOff + 1:(w.lOff + w.lCount)))

    result = pairgeometry_blocked(grid, pair, src;
                                  transform = () -> fast_transform(3413, 4326), window = win,
                                  ntasks = max(1, Threads.nthreads()),
                                  nodata = nodata_from(-32767.0))

    # `netcdf_output.py:293` medians `offset2vr` after casting to Float32, and stores the result as a
    # Float32 attribute, so the comparison is at that precision rather than Float64.
    @testset "dr_to_vr_factor matches the delivered product" begin
        vr = sort!(filter(!=(Float32(SENTINEL)), Float32.(vec(result.off2vx_dr))))
        # `netcdf_output.py:293` takes `np.median`, which averages the two middle values of an even
        # count; reproduced rather than taking the lower one.
        ours = isodd(length(vr)) ? Float64(vr[(length(vr) + 1) ÷ 2]) :
               (Float64(vr[length(vr) ÷ 2]) + Float64(vr[length(vr) ÷ 2 + 1])) / 2
        theirs = ArchGDAL.read("NETCDF:\"$PRODUCT\":M11") do ds
            parse(Float64, ArchGDAL.metadataitem(ds, "M11#dr_to_vr_factor"))
        end
        @test isapprox(Float64(ours), theirs; rtol = 1e-6)
        @info "dr_to_vr_factor" ours = Float64(ours) theirs relative =
            abs(Float64(ours) - theirs) / abs(theirs)
    end

    @testset "M11 and M12 match the delivered product" begin
        m11p, m12p, pgt = read_product(PRODUCT)

        # Both grids are EPSG:3413 at the same spacing, so the offset between them is whole pixels —
        # asserted, because a fractional offset would mean the two are not the same grid at all and
        # every comparison below would be silently averaging neighbours.
        ogt = window_geotransform(grid, win)
        dx = (ogt[1] - pgt[1]) / pgt[2]
        dy = (pgt[4] - ogt[4]) / abs(pgt[6])
        @test isapprox(dx, round(dx); atol = 1e-6)
        @test isapprox(dy, round(dy); atol = 1e-6)
        ix, iy = Int(round(dx)), Int(round(dy))

        det = result.off2vx_dx .* result.off2vy_dy .- result.off2vx_dy .* result.off2vy_dx
        m11o = result.off2vy_dy ./ det ./ result.scale_x
        m12o = .-result.off2vx_dy ./ det ./ result.scale_x
        missing_here = (result.off2vx_dx .== SENTINEL) .| (result.off2vx_dy .== SENTINEL) .|
                       (result.off2vy_dx .== SENTINEL) .| (result.off2vy_dy .== SENTINEL) .|
                       (result.scale_x .== SENTINEL)

        # The overlap in the product's index space.
        c0, r0 = max(0, ix), max(0, iy)
        c1 = min(size(m11p, 1), ix + size(m11o, 1))
        r1 = min(size(m11p, 2), iy + size(m11o, 2))
        @test c1 > c0 && r1 > r0

        for (name, ours, theirs) in (("M11", m11o, m11p), ("M12", m12o, m12p))
            rels = Float64[]
            for pc in (c0 + 1):c1, pr in (r0 + 1):r1
                oc, orr = pc - ix, pr - iy
                missing_here[oc, orr] && continue
                w = Float64(theirs[pc, pr])
                (isnan(w) || w == SENTINEL) && continue
                got = ours[oc, orr]
                isfinite(got) || continue
                push!(rels, abs(got - w) / max(abs(w), 1e-30))
            end
            n = length(rels)
            sort!(rels)
            worst = rels[end]
            med = rels[(n + 1) ÷ 2]
            p999 = rels[max(1, ceil(Int, 0.999 * n))]
            # The product stores these as Float32, so agreement is bounded by that and not by anything
            # here: the median sits at 2.4e-8, which is where a Float32 round-trip of a value this size
            # lands. The tail is wider because `det` is a difference of products, so cancellation
            # amplifies the stored inputs' last bits — hence a tight bound on the bulk and a loose one
            # on the maximum, rather than letting a handful of cancellation-prone points set the bound
            # for 1.6 million others.
            @test n > 1_000_000
            @test med < 1e-7
            @test p999 < 1e-6
            @test worst < 1e-4
            @info "$name against the delivered product" points = n median = med p999 worst
        end
    end
end
