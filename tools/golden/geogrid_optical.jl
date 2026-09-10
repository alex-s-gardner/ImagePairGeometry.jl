# Layer 1, projected path: this package's geogrid against the golden runs' own output.
#
#     julia --project=tools/golden -t 8 tools/golden/geogrid_optical.jl LC08_L1TP_009011
#     julia --project=tools/golden -t 8 tools/golden/geogrid_optical.jl --all
#
# Every input the reference had, this run has: the same twelve parameter rasters windowed to the same
# geogrid window (`params.jl`), the same image overlap to full precision (`scenes.jl`), the same repeat
# interval and chip size from the log. So a band difference is a difference in the kernel, which is
# what makes the bitwise gate on the integer bands meaningful rather than optimistic.
#
# The grid comes from the cached DEM rather than being constructed, and the window is the cache's whole
# extent, so the geotransform `pairgeometry` computes is the file's own and no offset arithmetic stands
# between the two sides. `check_params` has already asserted that geotransform equals the one the run's
# `window_location.tif` carries.
#
# The float bound is the case's own: zero — bitwise — where the grid and the image share a CRS, and
# `FLOAT_REL_BOUND` where they do not, matching `test/geogrid.jl`'s split for the same reason. Every
# golden case reprojects, since the parameter grids are polar stereographic and the imagery is UTM, so
# in practice the relative bound is what applies; the same-CRS branch exists because a case could be
# added that does not.

using ImagePairGeometry
using ImagePairGeometry: nodata_from, fast_transform, mapgrid, window_geotransform,
                         reference_files, nodata_from
using Rasters
using DimensionalData
using DiskArrays
using Printf

include(joinpath(@__DIR__, "cases.jl"))
include(joinpath(@__DIR__, "scenes.jl"))
include(joinpath(@__DIR__, "params.jl"))
include(joinpath(@__DIR__, "compare.jl"))

const RASTERS_EXT = Base.get_extension(ImagePairGeometry, :ImagePairGeometryRastersExt)
RASTERS_EXT === nothing && error("the Rasters extension did not load; `RasterInputs` is unavailable")
using .RASTERS_EXT: RasterInputs

"""
Largest difference tolerated on a float band where the grid and the image are in different CRSs —
relative to the value for the scale factors, and to the operator's own magnitude for the off2vel
components, per [`compare_float_band`](@ref).

`test/geogrid.jl`'s bound, and it holds on production data at the same value. The difference it
tolerates comes from the transform, not the kernel, and its size is a measured property of two things
the kernel composes.

The kernel builds each axis unit vector from the *difference* of two inverse-transformed coordinates one
pixel apart, so an error in the transform is amplified by `|x| / spacing` — around 1e5 at ITS_LIVE
scale. FastGeoProjections and PROJ agree on the *coordinates* to 3.7e-15 relative for every pair the
golden set uses, but on that one-pixel difference only to 2.1e-11 for EPSG:3413↔32622 and 2.7e-9 for
3413↔32607.

`off2vy_dy` then divides by `xunit[1]`, which is near zero wherever the image axes are nearly
perpendicular to the grid's — so the same absolute error reads `1/|xunit[1]|` times larger. Measured
across three cases: `xunit[1]` of 0.99, 0.10 and 0.003 give 1×, 9.6× and 332× amplification, matching
the observed 1e-9, 1.3e-7 and 4.2e-6 self-normalized differences. Substituting PROJ for
FastGeoProjections drops the band to exactly zero at most points, which is what identifies the
transform as the source.

Normalizing each operator by its own largest component reports one error for one operator instead of
that spread, which is why this bound needs no widening for production data. Every integer band is
bitwise across all 18 optical runs regardless.
"""
const FLOAT_REL_BOUND = 1e-7

"""
    geogrid_result(r::GoldenRun; ntasks) -> PairGeometry

This package's geogrid output for run `r`, over the run's own window and inputs.
"""
function geogrid_result(r::GoldenRun; ntasks::Integer = max(1, Threads.nthreads()))
    paths = fetch_params(r)
    check_params(r, paths)

    dem = Raster(paths["dem"]; lazy = true, missingval = nothing)
    fromfile = mapgrid(dem)
    grid = MapGrid(geotransform = fromfile.geotransform, size = fromfile.size, crs = r.epsg)

    coord = overlap_geometry(r)
    fp = ImageFootprint(origin = coord.origin, spacing = coord.spacing, size = coord.size)
    # The reference takes every parameter from the overlap and the secondary only for the interval,
    # which is what pairing the overlap with itself expresses.
    pair = coregister(fp, fp; dt = r.dt)

    # The cached rasters are exactly the geogrid window, so the window is their whole extent.
    win = CartesianIndices((1:fromfile.size[1], 1:fromfile.size[2]))

    lazy(field) = haskey(paths, field) ?
                  Raster(paths[field]; lazy = true, missingval = nothing) : nothing
    src = RasterInputs(dem = dem,
                       dhdx = lazy("dhdx"), dhdy = lazy("dhdy"),
                       vx = lazy("vx"), vy = lazy("vy"),
                       srx = lazy("srx"), sry = lazy("sry"),
                       csminx = lazy("csminx"), csminy = lazy("csminy"),
                       csmaxx = lazy("csmaxx"), csmaxy = lazy("csmaxy"),
                       ssm = lazy("ssm"))

    image_epsg = image_crs(r)
    return pairgeometry_blocked(grid, pair, src;
                                transform = () -> fast_transform(r.epsg, image_epsg),
                                window = win, ntasks,
                                params = GeometryParams(chip_size_0 = r.chip_size_0),
                                nodata = nodata_from(r.nodata))
end

"""
    image_crs(r::GoldenRun) -> Int

The EPSG code of the imagery this run's geogrid read.

Taken from the scene metadata rather than guessed from the overlap's coordinates: a UTM zone cannot be
inferred from an easting, and the four reprojected cases were warped into their *reference scene's*
CRS, which is not the zone the secondary's coordinates suggest.
"""
function image_crs(r::GoldenRun)
    ref, _ = scene_paths(r)
    if startswith(basename(ref), "T") || occursin("gcp-public-data-sentinel-2", ref)
        return cached_gdal_epsg(ref)
    end
    # Reprojecting warps both scenes into the *reference scene's* CRS (`utils.py:160-174`), and
    # filtering preserves it, so in all three provenances the reference scene's STAC item names the
    # code geogrid worked in. `granule_of` strips the band suffix, so a `reprojected/` path resolves to
    # the same granule as the delivered one.
    return stac_epsg(granule_of(ref))
end

"""
    stac_epsg(granule) -> Int

A Landsat scene's EPSG code from its cached STAC item.
"""
function stac_epsg(granule::AbstractString)
    path = joinpath(PARAMS_CACHE, "stac", granule * ".json")
    isfile(path) || (mkpath(dirname(path));
                     Downloads.download("$LANDSATLOOK/$granule", path))
    item = JSON3.read(read(path, String))
    haskey(item.properties, :var"proj:epsg") || error(
        "the STAC item for $granule carries no `proj:epsg`")
    return Int(item.properties[:var"proj:epsg"])
end

"""
    cached_gdal_epsg(url) -> Int

A raster's EPSG code from the cached `gdalinfo -json` header [`cached_gdal_geometry`](@ref) wrote.
"""
function cached_gdal_epsg(url::AbstractString)
    cached_gdal_geometry(url)      # ensures the header is cached
    info = JSON3.read(read(header_cache_path(url), String))
    # `gdalinfo -json` reports the code under `stac`, alongside a full WKT that also carries it as its
    # outermost authority ID. Reading the field rather than the WKT: an authority ID appears on the
    # datum and on each axis too, so picking one out of the text means picking the right one.
    (haskey(info, :stac) && haskey(info.stac, :var"proj:epsg")) || error(
        "the cached header for $url carries no `stac.proj:epsg`; GDAL reports it for any raster " *
        "whose CRS it recognizes, so this file's projection is not one it resolved")
    return Int(info.stac[:var"proj:epsg"])
end

"""
Points per integer band allowed to differ by exactly one because the float behind them lands within
the transform's noise of a `.5` rounding boundary. See [`compare_int_band`](@ref) for the measurement;
one such point exists across all 18 optical runs, so this is a bound of the same order rather than a
budget anything is expected to fill.
"""
const BOUNDARY_POINTS = 4

"""
    check_optical(r::GoldenRun; ntasks) -> Vector{BandResult}

Run `r`'s geogrid output against this package's, band by band plus the coverage count.
"""
function check_optical(r::GoldenRun; ntasks::Integer = max(1, Threads.nthreads()))
    r.radar && throw(ArgumentError("$(short_name(r)) run $(r.run) is a radar case; " *
                                   "use geogrid_radar.jl"))
    result = geogrid_result(r; ntasks)
    # Bitwise floats are achievable only where no reprojection happens, and a boundary allowance is
    # needed only where one does — the same condition, since both are consequences of the transform.
    same_crs = r.epsg == image_crs(r)
    rs = compare_geometry(r, result;
                          float_bound = same_crs ? 0.0 : FLOAT_REL_BOUND,
                          allow_boundary = same_crs ? 0 : BOUNDARY_POINTS)
    pushfirst!(rs, compare_coverage(r, result))
    return rs
end

"""
    main_optical(args) -> Bool

Compare every projected-path run named by `args`, printing each band's verdict. Returns whether all
passed.
"""
function main_optical(args)
    name = isempty(args) || startswith(first(args), "--") ? nothing : first(args)
    runs = filter(r -> !r.radar, goldenruns(; name))
    isempty(runs) && error("no projected-path golden run on disk" *
                           (name === nothing ? "" : " matching \"$name\""))
    allok = true
    for r in runs
        @printf("\n=== %s run %d  [%s, grid EPSG %d, imagery EPSG %d]\n", short_name(r), r.run,
                scene_provenance(r), r.epsg, image_crs(r))
        allok &= report(check_optical(r))
    end
    println()
    println(allok ? "every band of every run agrees within its gate" :
            "at least one band is outside its gate")
    return allok
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main_optical(ARGS) ? 0 : 1)
end
