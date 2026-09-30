# The parameter rasters a golden run read, windowed to that run's geogrid and cached locally.
#
# A run reads twelve rasters from `s3://its-live-data/autorift_parameters/v001` over `/vsicurl/`, at
# 120 m over a whole hemisphere: the NPS and SPS tiles are 68480² each. Only the geogrid window is
# ever touched — around 2500² — so what is cached is the window rather than the tile, keyed by run.
#
# The window is exactly the run's own: the log's `Origin index (in DEM) of geogrid` and `Dimensions of
# geogrid` give it, and reading the same window the reference read is what makes a band comparison a
# statement about the kernel rather than about alignment. `pairgeometry` is then run over the whole
# cached grid, so its window is the cache's own extent and no offset arithmetic enters.
#
# The URLs come from the log, never from a naming rule. The reference velocity rasters are `vx0`/`vy0`
# where every other quantity drops the digit, and a run against `vx` instead reads a different
# velocity field — a plausible one, so the mistake would surface as a small bias in `window_offset`
# rather than as a failure.
#
# Fetching goes through the `gdal_translate` executable rather than ArchGDAL. The Sentinel-2 tiles on
# the Google Cloud mirror do not open under ArchGDAL's bundled GDAL on this machine while the system
# one reads them, so both this and `cached_gdal_geometry` use the system binary for remote reads and
# keep ArchGDAL for the local files it handles.

using Printf

include(joinpath(@__DIR__, "cases.jl"))

"""
    params_dir(r::GoldenRun) -> String

Where this run's windowed parameter rasters live. Keyed by product and run, since two runs of one case
can differ in the window they used.
"""
params_dir(r::GoldenRun) = joinpath(PARAMS_CACHE, r.case.product, string(r.run))

"""
    fetch_params(r::GoldenRun; force = false) -> Dict{String,String}

Window every parameter raster this run read into the local cache, returning the paths by field name.

Each raster is cut to the run's geogrid window with `gdal_translate -srcwin`, which issues a few range
requests rather than reading the tile. Already-cached files are left alone unless `force`.
"""
function fetch_params(r::GoldenRun; force::Bool = false)
    dir = params_dir(r)
    mkpath(dir)

    # `-srcwin` is zero-based, which is the convention the log's origin index is already in. The
    # window's `CartesianIndices` is one-based, so the offset is its first index less one.
    xoff = first(r.window.indices[1]) - 1
    yoff = first(r.window.indices[2]) - 1
    nx = length(r.window.indices[1])
    ny = length(r.window.indices[2])

    out = Dict{String,String}()
    for (field, url) in sort(collect(r.param_urls))
        path = joinpath(dir, field * "_" * basename(url))
        out[field] = path
        (isfile(path) && !force) && continue

        # Written to a temporary name and renamed, so a cached file is either absent or complete. A
        # partially written GeoTIFF still opens: GDAL reads its header and returns the sentinel for the
        # blocks not yet on disk, so a reader sees a valid raster full of nodata rather than an error.
        # That reaches a band comparison as a coverage difference in the kernel — measured, on a run
        # whose cache was being written concurrently: `window_offset` differed on 3.5 million points
        # while every other band was bitwise.
        #
        # The temporary name keeps the `.tif` extension. `gdal_translate` picks its output driver from
        # the extension when `-of` is absent, so a `.tif.partial` target is not a GeoTIFF target and the
        # command fails outright.
        @info "windowing a parameter raster" field basename(url) window = (xoff, yoff, nx, ny)
        tmp = path * ".partial.tif"
        try
            # Retried, because these are range requests against a public bucket over the open internet
            # and a single failure is far more often the network than the request. An unretried sweep of
            # 26 runs times 12 rasters fails somewhere almost every time.
            ok = false
            for attempt in 1:4
                try
                    run(pipeline(`gdal_translate -q -srcwin $xoff $yoff $nx $ny $url $tmp`;
                                 stdout = devnull, stderr = devnull))
                    ok = isfile(tmp)
                    ok && break
                catch e
                    attempt == 4 && rethrow()
                    @warn "a windowed read failed; retrying" field attempt
                    isfile(tmp) && rm(tmp; force = true)
                    sleep(2.0 * attempt)
                end
            end
            ok || error("gdal_translate produced no $tmp after 4 attempts")
            mv(tmp, path; force = true)
        finally
            isfile(tmp) && rm(tmp; force = true)
        end
    end
    return out
end

"""
    check_params(r::GoldenRun, paths) -> Nothing

Assert the cached rasters describe the grid the run's geogrid output describes.

The DEM's cached window is the grid `pairgeometry` runs over, so its geotransform must equal the one
the run's `window_location.tif` carries. A disagreement means the window was cut at the wrong offset,
and every band comparison after it would be comparing different ground.
"""
function check_params(r::GoldenRun, paths::Dict{String,String})
    dem_gt = ArchGDAL.read(ds -> ArchGDAL.getgeotransform(ds), paths["dem"])
    out_gt = ArchGDAL.read(ds -> ArchGDAL.getgeotransform(ds),
                           joinpath(r.dir, "window_location.tif"))
    Tuple(dem_gt) == Tuple(out_gt) || error("""
        the cached DEM window and the run's geogrid output describe different grids:
          cached DEM         $(Tuple(dem_gt))
          window_location    $(Tuple(out_gt))
        The window was cut at the wrong offset, so no band comparison would be about the kernel.""")

    # Every raster is on the DEM's grid, since geogrid indexes them all with one window.
    for (field, path) in sort(collect(paths))
        gt = ArchGDAL.read(ds -> ArchGDAL.getgeotransform(ds), path)
        Tuple(gt) == Tuple(dem_gt) || error(
            "the cached `$field` window has geotransform $(Tuple(gt)) but the DEM's is " *
            "$(Tuple(dem_gt)); geogrid reads every parameter raster with one window, so these " *
            "must agree")
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    name = isempty(ARGS) || startswith(first(ARGS), "--") ? nothing : first(ARGS)
    force = "--force" in ARGS
    runs = goldenruns(; name)
    isempty(runs) && error("no golden run on disk" * (name === nothing ? "" : " matching \"$name\""))
    for r in runs
        paths = fetch_params(r; force)
        check_params(r, paths)
        @printf("%-42s run %-4d %2d rasters cached and checked\n", short_name(r), r.run,
                length(paths))
    end
end
