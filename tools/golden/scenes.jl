# The image geometry a golden run's geogrid was called with, to full precision.
#
# `runGeogrid` receives the *overlap* of the two scenes, not either scene: `GeogridOptical.coregister`
# intersects them and hands back a window into each (`GeogridOptical.py:240-300`). The log prints that
# overlap — `X-direction coordinate: -2.12881e+06  15` — through C++ `ostream`'s six significant
# figures, which is not enough to place a 15 m pixel. So the geometry is reconstructed here and the
# log's line serves as the cross-check.
#
# Reconstruction has two ingredients, and `ImagePairGeometry.coregister` is the second:
#
#   1. each scene's own geometry, from the landsatlook STAC API for Landsat and from the tile's own
#      grid for Sentinel-2;
#   2. `coregister`, which implements the same intersection the reference does.
#
# Verified on `LC09_L1GT_215109`, whose two scenes differ in origin by 300 m in x and y: from the two
# STAC transforms, `coregister` returns origin `(-2128807.5, 1160407.5)` and size `(18361, 18341)`,
# and the log prints `-2.12881e+06  1.16041e+06` and `Dimensions: 18361 18341`. The origin agrees to
# the last printed digit and both dimensions agree exactly.
#
# **Three cases geogrid a reprojected scene rather than the delivered one.** Where the two scenes are
# in different UTM zones, `hyp3_autorift` warps both into the reference's CRS before coregistering
# (`process.py:60-66`, `utils.py:156-185`), and geogrid then sees an image whose geometry is a
# `gdal.Warp` product that no catalogue describes. `targetAlignedPixels=True` is what makes those
# recoverable anyway: it snaps the warped origin to a multiple of the 15 m resolution, and only one
# such multiple renders to the six figures the log prints. Checked over a ±1000-pixel search around
# each printed value — exactly one candidate each for all three.
#
# So `scene_geometry` returns the origin as a *reconstruction plus its evidence*, and
# `check_printed` is what a caller asserts against. A wrong origin here moves every pixel index by a
# fraction of a pixel, which is the failure mode
# `AutoRIFT.jl/tools/golden/intermediate.jl` records as invisible in a median: zero residual under
# uniform motion, growing with the velocity gradient.

using ArchGDAL
using Downloads
using ImagePairGeometry: ImageFootprint, ProjectedCoordinate, coregister
using JSON3
using Printf

const LANDSATLOOK = "https://landsatlook.usgs.gov/stac-server/collections/landsat-c2l1/items"

# Which band each platform correlates, from `process.py:77-89`: Landsat 4/5 use B2 (green) and
# Landsat 7/8/9 use B8 (panchromatic), so the pixel size is 30 m on the older pair and 15 m on the
# newer. The STAC asset keys for those two bands.
const STAC_BAND = Dict("L4" => "green", "L5" => "green",
                       "L7" => "pan", "L8" => "pan", "L9" => "pan")

"""
    SceneGeometry

One scene's own grid: where its first pixel's outer corner sits, its signed spacing, and its size in
pixels. The `(nx, ny)` convention [`ImageFootprint`](@ref) uses, not STAC's `[rows, cols]`.
"""
struct SceneGeometry
    origin::NTuple{2,Float64}
    spacing::NTuple{2,Float64}
    size::NTuple{2,Int}
    source::String
end

"""
    stac_geometry(granule, band; cache = true) -> SceneGeometry

A Landsat scene's grid from the landsatlook STAC API.

The API is open — the *files* redirect to an ERS login and the `usgs-landsat` bucket is
requester-pays, but the item metadata needs no credential. Responses are cached under
`PARAMS_CACHE/stac`, so a sweep re-reads nothing and runs offline once populated.
"""
function stac_geometry(granule::AbstractString, band::AbstractString; cache::Bool = true)
    dir = joinpath(PARAMS_CACHE, "stac")
    path = joinpath(dir, granule * ".json")
    if !(cache && isfile(path))
        mkpath(dir)
        Downloads.download("$LANDSATLOOK/$granule", path)
    end
    item = JSON3.read(read(path, String))

    assets = item.assets
    haskey(assets, Symbol(band)) || error(
        "the STAC item for $granule has no \"$band\" asset; it has " *
        join(String.(keys(assets)), ", "))
    a = assets[Symbol(band)]

    # A Landsat 7/8/9 item carries the grid per asset, because its panchromatic band is finer than
    # its multispectral ones. A Landsat 4/5 item carries it at item level instead — every band of a
    # TM scene is 30 m, so one grid describes them all. Preferring the asset keeps the pan band's own
    # 15 m grid where both are present.
    where, tf, shape = if haskey(a, :var"proj:transform")
        "$granule/$band", collect(Float64, a[:var"proj:transform"]),
        collect(Int, a[:var"proj:shape"])
    elseif haskey(item.properties, :var"proj:transform")
        "$granule (item level)", collect(Float64, item.properties[:var"proj:transform"]),
        collect(Int, item.properties[:var"proj:shape"])
    else
        error("""the STAC item for $granule carries `proj:transform` neither on its "$band" asset
                 nor at item level, so its grid is not recoverable from STAC.""")
    end

    # STAC's `proj:transform` is the GDAL geotransform's first six terms in row-major order
    # (a, b, c, d, e, f), so the origin is (c, f) and the spacing (a, e). `proj:shape` is
    # `[rows, cols]` = (ny, nx).
    return SceneGeometry((tf[3], tf[6]), (tf[1], tf[5]), (shape[2], shape[1]),
                         "landsatlook STAC $where")
end

"""
    aligned_origin(printed, step; halfpixel, search = 1000) -> Float64

The unique multiple of `step` — offset by half a step when `halfpixel` — whose `%g` rendering is
`printed`.

This is what recovers a coordinate the log truncated to six significant figures. It is only sound
because the grid is quantized: a Landsat pan scene's origin sits on `n * 15 + 7.5`, and a
`gdal.Warp` with `targetAlignedPixels` lands on `n * 15`. Throws unless exactly one candidate in
`±search` steps matches, so an ambiguous recovery is an error rather than a first guess.
"""
function aligned_origin(printed::AbstractString, step::Real; halfpixel::Bool,
                        search::Integer = 1000)
    v = parse(Float64, printed)
    base = halfpixel ? step / 2 : zero(step)
    hits = Float64[]
    k0 = round(Int, (v - base) / step)
    for k in (k0 - search):(k0 + search)
        x = k * step + base
        @sprintf("%g", x) == printed && push!(hits, x)
    end
    isempty(hits) && error(
        "no multiple of $step" * (halfpixel ? " offset by half of it" : "") *
        " within $search steps of $v renders as \"$printed\"")
    length(hits) == 1 || error(
        "\"$printed\" does not determine a coordinate on a $step grid: candidates $hits")
    return only(hits)
end

"""
    check_printed(name, value, printed) -> Nothing

Assert that `value` renders exactly as the log's `printed` field.

The cross-check on every reconstructed coordinate. `%g` is what `GeogridOptical.cpp` prints through,
so a reconstruction that disagrees in the sixth figure is a different coordinate than the run used.
"""
function check_printed(name::AbstractString, value::Real, printed::Real)
    # The log's own value, re-rendered, is the string the run printed — comparing renderings rather
    # than numbers is what makes this a test of agreement to the printed precision instead of a
    # tolerance nobody derived.
    got = @sprintf("%g", value)
    want = @sprintf("%g", printed)
    got == want || error("""$name reconstructs to $value, which prints as "$got", but the run's log
                             printed "$want". These are different coordinates.""")
    return nothing
end

# ---------------------------------------------------------------------------
# The scenes one run's geogrid was called with
# ---------------------------------------------------------------------------

"""
    scene_paths(r::GoldenRun) -> (reference, secondary)

The two image files this run's `runGeogrid` opened, as the log names them.

The **last** reference/secondary pair the log logs, not the first: `process.py` logs the delivered
scene, then logs again after reprojecting or filtering, and it is the final pair that geogrid saw.
Taking the first pair instead reads the geometry of a file the run did not use — which for the four
reprojected cases is in a different CRS entirely.
"""
function scene_paths(r::GoldenRun)
    text = read(joinpath(r.dir, "capture.log"), String)
    ms = collect(eachmatch(r"INFO - (Reference|Secondary) scene path: (\S+)$"m, text))
    isempty(ms) && error("the log of run $(r.run) of $(r.case.product) names no scene paths")
    ref = last([m for m in ms if m.captures[1] == "Reference"]).captures[2]
    sec = last([m for m in ms if m.captures[1] == "Secondary"]).captures[2]
    return (String(ref), String(sec))
end

"""
    scene_provenance(r) -> Symbol

Where the images geogrid opened came from: `:delivered`, `:filtered` or `:reprojected`.

This decides how the overlap geometry is recovered, and the three differ in kind:

  * `:delivered` — the scene as the archive holds it. Its grid comes from the catalogue.
  * `:filtered` — a Wallis or FFT prefilter, which `process.py:290-300` writes with the input's own
    `image_transform`. Same grid as delivered, so the catalogue still describes it.
  * `:reprojected` — a `gdal.Warp` into the reference's CRS, which no catalogue describes. Its origin
    is recovered from the log through [`aligned_origin`](@ref), which is sound only because the warp
    passes `targetAlignedPixels=True`.
"""
function scene_provenance(r::GoldenRun)
    ref, _ = scene_paths(r)
    # The directory is relative in the log — `reprojected/LC08_..._B8.TIF` — because the driver writes
    # it beside its working directory, so the match is on the leading component rather than on a
    # slash-delimited segment.
    parts = splitpath(ref)
    "reprojected" in parts && return :reprojected
    "filtered" in parts && return :filtered
    return :delivered
end

"""
    granule_of(path) -> String

The granule name in a scene path, with the band suffix and extension removed.

Works on a delivered `/vsis3/` path, a `reprojected/` one and a `filtered/` one alike, since all
three keep the granule name in the filename.
"""
function granule_of(path::AbstractString)
    stem = first(splitext(basename(path)))
    # Landsat: strip the band suffix `_B8` / `_B2`. Sentinel-2: the tile-and-band form
    # `T07VEG_20200626T204021_B08` names no granule, so the caller handles S2 separately.
    m = match(r"^(.*)_B\d+$", stem)
    return m === nothing ? stem : String(m.captures[1])
end

"""
    overlap_geometry(r::GoldenRun) -> ProjectedCoordinate

The image coordinate system this run's geogrid computed against.

Built by intersecting the two scenes with [`coregister`](@ref) — the same intersection
`GeogridOptical.py:240-300` performs — and then asserted against the origin, spacing and size the run
printed. The assertion is the point: the reconstruction is only useful if it lands on the run's own
numbers, and a half-pixel error would otherwise pass silently.
"""
function overlap_geometry(r::GoldenRun)
    r.radar && throw(ArgumentError("overlap_geometry is for the projected path; " *
                                   "$(short_name(r)) run $(r.run) is radar"))
    a, b = scene_footprints(r)
    pair = coregister(a, b; dt = r.dt)
    c = pair.coordinate

    check_printed("$(short_name(r)) overlap x origin", c.origin[1], r.printed_origin[1])
    check_printed("$(short_name(r)) overlap y origin", c.origin[2], r.printed_origin[2])
    c.spacing == r.printed_spacing || error(
        "$(short_name(r)): reconstructed spacing $(c.spacing), log printed $(r.printed_spacing)")
    c.size == r.printed_size || error(
        "$(short_name(r)): reconstructed overlap is $(c.size) pixels, log printed " *
        "$(r.printed_size). The two scenes' grids do not intersect where the run's did.")
    return c
end

"""
    gdal_geometry(path) -> SceneGeometry

A raster's grid, read from the raster. For a file no catalogue describes, or to check one that is.

Works on a `/vsicurl/` URL as well as a local path, which is how a Sentinel-2 tile on the Google
Cloud mirror is reached — anonymous, so it needs no credential.
"""
function gdal_geometry(path::AbstractString)
    ArchGDAL.read(path) do ds
        gt = ArchGDAL.getgeotransform(ds)
        (gt[3] == 0 && gt[5] == 0) || error(
            "$path is a rotated raster (geotransform $gt); the overlap arithmetic here assumes " *
            "north-up, as `GeogridOptical.coregister` does")
        return SceneGeometry((gt[1], gt[4]), (gt[2], gt[6]),
                             (ArchGDAL.width(ds), ArchGDAL.height(ds)), path)
    end
end

"""
    scene_footprints(r::GoldenRun) -> (a, b)

The two scenes' [`ImageFootprint`](@ref)s, in the geometry geogrid saw them in.

How each is obtained depends on [`scene_provenance`](@ref):

  * `:delivered` and `:filtered` share a grid, so both come from the catalogue — landsatlook STAC for
    Landsat, and the tile itself for Sentinel-2, whose Google Cloud mirror is anonymous.
  * `:reprojected` has no catalogue entry. Both scenes are warped into the reference's CRS on the
    reference's own grid, so the overlap's spacing is the reference's and its origin lies on the
    warp's aligned grid — [`aligned_origin`](@ref) recovers it from the log, and the size is the
    log's own `Dimensions`, which is exact.

The reprojected branch reconstructs the *overlap* directly rather than the two scenes, since the two
warped grids are not recoverable separately. [`overlap_geometry`](@ref)'s assertions then compare it
against the log, which is the only evidence available for those four cases.
"""
function scene_footprints(r::GoldenRun)
    prov = scene_provenance(r)
    ref, sec = scene_paths(r)

    if prov === :reprojected
        # The two warped grids are not separately recoverable, so the overlap is reconstructed as one
        # footprint and `coregister` intersects it with itself — which returns it unchanged. The log's
        # `Dimensions` is exact and its origin lies on the warp's aligned grid.
        sp = r.printed_spacing
        x = aligned_origin(@sprintf("%g", r.printed_origin[1]), abs(sp[1]); halfpixel = false)
        y = aligned_origin(@sprintf("%g", r.printed_origin[2]), abs(sp[2]); halfpixel = false)
        fp = ImageFootprint(origin = (x, y), spacing = sp, size = r.printed_size)
        return (fp, fp)
    end

    if startswith(basename(ref), "T") || occursin("gcp-public-data-sentinel-2", ref)
        # Sentinel-2: read the tiles themselves, through a cached header. Both scenes of both S2
        # cases are the same MGRS tile, so the overlap is the whole tile — the intersection is still
        # computed rather than assumed, since nothing here guarantees it.
        return (footprint(cached_gdal_geometry(ref)), footprint(cached_gdal_geometry(sec)))
    end

    band = STAC_BAND[platform_of(r)]
    return (footprint(stac_geometry(granule_of(ref), band)),
            footprint(stac_geometry(granule_of(sec), band)))
end

"""
    platform_of(r) -> String

The case's platform as `STAC_BAND` keys it: `"L4"`, `"L5"`, `"L7"`, `"L8"`, `"L9"`.

Read from the manifest rather than from the granule name, since the manifest is the authoritative
job record.
"""
platform_of(r::GoldenRun) = r.case.platform

footprint(g::SceneGeometry) =
    ImageFootprint(origin = g.origin, spacing = g.spacing, size = g.size)

"""
    cached_gdal_geometry(url) -> SceneGeometry

[`gdal_geometry`](@ref) through a cached local copy of the header, for a raster ArchGDAL's own GDAL
build cannot open remotely.

The Sentinel-2 tiles are JPEG 2000 on an HTTPS mirror, and `ArchGDAL.read` returns a null dataset for
them while the system `gdalinfo` reads them fine — a difference in the two GDAL builds' network or
JP2 configuration, not in the data. So the geometry is taken once with `gdalinfo -json` and cached as
JSON under `PARAMS_CACHE/headers`, which also makes a sweep offline after the first run.

Only the header is fetched, not the image: `gdalinfo` reads the JP2 codestream header and a few
range requests, not the 10980² samples.
"""
function cached_gdal_geometry(url::AbstractString)
    dir = joinpath(PARAMS_CACHE, "headers")
    key = bytes2hex(codeunits(url))[1:min(end, 32)] * "_" * basename(url) * ".json"
    path = joinpath(dir, key)
    if !isfile(path)
        mkpath(dir)
        # `-nomd -norat` keeps the JSON to the geometry; the metadata of an S2 tile is large and
        # nothing here reads it.
        out = read(`gdalinfo -json -nomd -norat $url`, String)
        write(path, out)
    end
    info = JSON3.read(read(path, String))
    gt = collect(Float64, info.geoTransform)
    (gt[3] == 0 && gt[5] == 0) || error(
        "$url is a rotated raster (geotransform $gt); the overlap arithmetic here assumes " *
        "north-up, as `GeogridOptical.coregister` does")
    sz = collect(Int, info.size)
    return SceneGeometry((gt[1], gt[4]), (gt[2], gt[6]), (sz[1], sz[2]), url)
end
