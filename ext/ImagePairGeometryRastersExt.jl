module ImagePairGeometryRastersExt

# Reading inputs from rasters, and writing the nine outputs.
#
# Two things the core deliberately does not do, because doing them would put GDAL behind every use of
# the kernel:
#
# Reading. A `RasterInputs` source reads each block's window from disk-backed rasters, so a grid too
# large to materialize never is. The read is a plain `view` plus `copyto!`, which for a
# `DiskArrays`-backed parent is already a chunk-aware windowed read — no per-backend code, and the
# same path serves GeoTIFF, Zarr and NetCDF.
#
# Writing. The reference's consumers read nine GeoTIFFs with fixed names, so `write_geotiffs` emits
# exactly those, with the band order, data types and nodata values `REFERENCE_FILES` records.
#
# Grid and pair construction from rasters lives here too: a `MapGrid` from a DEM and an
# `ImageFootprint` from a scene are one-liners over the geotransform, but they are the place a
# half-pixel or an axis-order mistake would enter, so they are written once rather than at each call
# site.

using ImagePairGeometry
using ImagePairGeometry: PairGeometry, AbstractInputSource, GeometryInputs, REFERENCE_FILES,
                         RADAR_REFERENCE_FILES, reference_files, INT_BANDS, FLOAT_BANDS,
                         nodata_from, NoDataPolicy
using Rasters
using Rasters: AbstractRaster
using DimensionalData
using ArchGDAL
using DiskArrays

const GFT = Rasters.GeoFormatTypes

# Where a lookup coordinate sits within its pixel, as an offset from it to the pixel's outer edge.
#
# GDAL geotransforms and `MapGrid` put the origin at that edge, while an `ImageFootprint` origin is
# the pixel *center*, so a lookup's convention has to be established before either can be built from
# it.
#
# For a file-backed raster the geotransform is read from the file, which settles the question with no
# inference — see `_file_geotransform`. Inference is needed only for a raster built in memory, and
# there the sampling is what says which convention its coordinates follow.
#
# `Start` and `End` are the *low* and *high* edge of a cell in coordinate order, not the near and far
# edge in index order. On a reverse-ordered axis — a north-up raster's Y, where the step is negative —
# a `Start` lookup therefore holds the edge furthest from the geotransform origin, one whole step
# away. Taking `Start` to mean "no adjustment" gets X right and Y wrong by a pixel, which is exactly
# the kind of error a north-up raster hides until the outputs are compared against a reference.
const Lk = DimensionalData.Dimensions.Lookups

_locus_offset(::Lk.Start, step) = step < 0 ? -step : 0.0
_locus_offset(::Lk.End, step) = step < 0 ? 0.0 : -step
_locus_offset(::Lk.Center, step) = -step / 2

function _edge_offset(l, step)
    # `locus` is defined for `Intervals` sampling. A `Points` lookup names positions rather than
    # spanning cells, which for an in-memory raster is the center convention.
    loc = try
        Lk.locus(l)
    catch
        Lk.Center()
    end
    return _locus_offset(loc, step)
end

"""
    _file_geotransform(r) -> NTuple{6,Float64} or nothing

`r`'s geotransform as GDAL reports it, or `nothing` for a raster not backed by a file.

Preferred over deriving one from the lookups because it is the file's own answer. Rasters sets a
lookup's sampling from the `AREA_OR_POINT` tag — `Points` for `Point`, `Intervals{Start}` for `Area`
— while reporting the *same* coordinates either way, namely the geotransform's. So a raster tagged
`AREA_OR_POINT=Point`, which both the ITS_LIVE parameter rasters and the Landsat scenes are, gets
`Points` sampling with coordinates that are already pixel edges. Inferring the convention from the
sampling would take those edges for centers and displace the grid half a pixel.
"""
function _file_geotransform(r)
    path = Rasters.filename(r)
    path === nothing && return nothing
    return ArchGDAL.read(path) do ds
        gt = ArchGDAL.getgeotransform(ds)
        # Refuse a rotated grid here rather than silently dropping the rotation terms; `MapGrid`
        # rejects them too, but this keeps the message about the file.
        (iszero(gt[3]) && iszero(gt[5])) || throw(ArgumentError(
            "$path has a rotated geotransform ($(gt[3]), $(gt[5])); this package's grid " *
            "arithmetic assumes a north-up grid"))
        (gt[1], gt[2], gt[3], gt[4], gt[5], gt[6])
    end
end

"""
    ImagePairGeometry.mapgrid(dem::AbstractRaster) -> MapGrid

A `MapGrid` describing `dem`'s grid.

The grid's origin is the *outer edge* of the first pixel, the convention `MapGrid` and GDAL share.
For a raster backed by a file that is GDAL's geotransform, read from the file. For one built in
memory the convention is inferred from the lookup's sampling, and a `Center` or point lookup is
adjusted by half a pixel, since taking its coordinates for edges would offset every output by that
much.

A raster whose lookups are not regularly spaced is refused: the index arithmetic this package
reproduces assumes a constant step.
"""
function ImagePairGeometry.mapgrid(dem::AbstractRaster)
    x, y = dims(dem, X), dims(dem, Y)
    dx, dy = _step_of(x, :X), _step_of(y, :Y)
    gt = _file_geotransform(dem)
    origin = gt === nothing ?
        (Float64(first(x)) + _edge_offset(lookup(x), dx),
         Float64(first(y)) + _edge_offset(lookup(y), dy)) : (gt[1], gt[4])
    return MapGrid(geotransform = (origin[1], dx, 0.0, origin[2], 0.0, dy),
                   size = (length(x), length(y)),
                   crs = crs(dem))
end

"""
    ImagePairGeometry.image_footprint(image::AbstractRaster) -> ImageFootprint

An `ImageFootprint` describing where `image` sits, for `coregister`.

Origin is the first pixel's *center*, which is what `ImageFootprint` documents and what the
reference's `startingX`/`startingY` are — so the pixel edge the geotransform names is shifted half a
pixel inward.

Only the geometry is read, never the pixels, so this is cheap on a disk-backed scene and needs no
data at all.
"""
function ImagePairGeometry.image_footprint(image::AbstractRaster)
    x, y = dims(image, X), dims(image, Y)
    dx, dy = _step_of(x, :X), _step_of(y, :Y)
    gt = _file_geotransform(image)
    edge = gt === nothing ?
        (Float64(first(x)) + _edge_offset(lookup(x), dx),
         Float64(first(y)) + _edge_offset(lookup(y), dy)) : (gt[1], gt[4])
    return ImageFootprint(origin = (edge[1] + dx / 2, edge[2] + dy / 2), spacing = (dx, dy),
                          size = (length(x), length(y)))
end

function _step_of(d, name::Symbol)
    l = lookup(d)
    s = try
        DimensionalData.step(l)
    catch
        throw(ArgumentError(
            "the $name axis is not regularly spaced, so it has no single pixel size. The grid " *
            "arithmetic this package reproduces assumes a constant step."))
    end
    iszero(s) && throw(ArgumentError("the $name axis has zero step"))
    return Float64(s)
end

"""
    RasterInputs(; dem, dhdx = nothing, ..., ssm = nothing)

An input source reading each block's window from rasters.

Accepts the same eleven optional rasters as `GeometryInputs`, and each may be lazy: a
disk-backed `Raster` is read one window at a time, so a grid larger than memory is never
materialized. All must share the grid the geometry is computed on.

Open each with `missingval = nothing`, so its stored values arrive unmasked:

```julia
param(name) = Raster(name; lazy = true, missingval = nothing)
src = RasterInputs(dem = param("h.tif"), vx = param("vx.tif"), ...)
```

A raster whose element type admits `missing` is refused. Which value counts as missing is decided by
`NoDataPolicy` from the DEM's sentinel — the reference applies that one sentinel to every raster —
so a raster's own declared nodata is not it. The ITS_LIVE parameter rasters declare nodata `0` for
chip size and stable surface and `32767` for search range, all of which are ordinary data here.

Windows are indexed in grid coordinates, so the rasters must cover the whole grid, not just the
window — which is the usual case, since the DEM is what defines the grid.
"""
struct RasterInputs{D,S,V,R,Cn,Cx,M} <: AbstractInputSource
    dem::D
    dhdx::S
    dhdy::S
    vx::V
    vy::V
    srx::R
    sry::R
    csminx::Cn
    csminy::Cn
    csmaxx::Cx
    csmaxy::Cx
    ssm::M
end

# Each raster must carry its stored values, not `missing` where its own nodata was.
#
# Rasters masks a file's declared nodata to `missing` by default, which is the wrong model for this
# computation twice over. The reference reads *one* sentinel, from the DEM, and tests every raster
# against that (`geogridOptical.cpp:337-339`) — so a value matching some other raster's declared
# nodata is ordinary data. The ITS_LIVE rasters make that concrete: `xMinChipSize` and
# `StableSurface` declare nodata 0 and `vxSearchRange` declares 32767, none of which the reference
# treats as missing. And `missing` cannot be read into the `Float64` the kernel works in, so it
# would fail on the first block regardless.
function _check_unmasked(name::Symbol, r)
    eltype(r) >: Missing || return nothing
    throw(ArgumentError(
        "RasterInputs `$name` has element type $(eltype(r)), so Rasters is masking its nodata to " *
        "`missing`. Open it with `missingval = nothing` to get the stored values: which value " *
        "counts as missing is decided by `NoDataPolicy` from the DEM's sentinel, matching the " *
        "reference, and a raster's own declared nodata is not it."))
end

function RasterInputs(; dem, dhdx = nothing, dhdy = nothing, vx = nothing, vy = nothing,
                      srx = nothing, sry = nothing, csminx = nothing, csminy = nothing,
                      csmaxx = nothing, csmaxy = nothing, ssm = nothing)
    sz = size(dem)
    for (name, r) in ((:dem, dem), (:dhdx, dhdx), (:dhdy, dhdy), (:vx, vx), (:vy, vy), (:srx, srx),
                      (:sry, sry), (:csminx, csminx), (:csminy, csminy), (:csmaxx, csmaxx),
                      (:csmaxy, csmaxy), (:ssm, ssm))
        r === nothing && continue
        size(r) == sz || throw(DimensionMismatch(
            "RasterInputs `$name` is $(size(r)) but `dem` is $sz; every input must be on the " *
            "grid the geometry is computed on"))
        _check_unmasked(name, r)
    end
    return RasterInputs(dem, dhdx, dhdy, vx, vy, srx, sry, csminx, csminy, csmaxx, csmaxy, ssm)
end

# `view` then `copyto!` is a chunk-aware windowed read for any DiskArrays-backed parent, so one
# implementation serves every backend. `Float64` because the kernel works in it throughout and the
# reference reads every input as `GDT_Float64` regardless of the file's own type.
#
# Raw values, with each raster's own nodata left in place as the number it is on disk. Rasters would
# otherwise mask it to `missing`, which is both untypable as `Float64` and the wrong semantics: the
# reference reads one sentinel from the DEM and tests every raster against *that*
# (`geogridOptical.cpp:337-339`), so a value equal to some other raster's nodata is ordinary data.
# The ITS_LIVE rasters make the difference concrete — `xMinChipSize` and `StableSurface` declare
# nodata 0, and `vxSearchRange` declares 32767, none of which the reference treats as missing.
# `NoDataPolicy` is where a missing value is decided, and it is given the DEM's sentinel by the
# caller.
function _read(r, block::CartesianIndices{2})
    dest = Array{Float64}(undef, size(block))
    # A Raster is (X, Y) here; `block` indexes the grid the same way.
    copyto!(dest, view(parent(r), block.indices[1], block.indices[2]))
    return dest
end
_read(::Nothing, ::CartesianIndices{2}) = nothing

function ImagePairGeometry.readblock(src::RasterInputs, block::CartesianIndices{2})
    return GeometryInputs(dem = _read(src.dem, block),
                          dhdx = _read(src.dhdx, block), dhdy = _read(src.dhdy, block),
                          vx = _read(src.vx, block), vy = _read(src.vy, block),
                          srx = _read(src.srx, block), sry = _read(src.sry, block),
                          csminx = _read(src.csminx, block), csminy = _read(src.csminy, block),
                          csmaxx = _read(src.csmaxx, block), csmaxy = _read(src.csmaxy, block),
                          ssm = _read(src.ssm, block))
end

"""
    ImagePairGeometry.blocksize_from_chunks(src::RasterInputs; floor = 256) -> NTuple{2,Int}

A block size aligned to the inputs' own chunking.

Reading a window that straddles chunk boundaries reads whole chunks and discards the rest, so a
block that is a whole number of chunks reads each byte once. `floor` guards the pathological case: a
striped GeoTIFF — what `Rasters.write` produces by default — reports chunks one row tall, and a block
one row tall would issue a read per row.

Returns `ImagePairGeometry.DEFAULT_BLOCKSIZE` when no input is disk-backed, since then there is no chunking to
align to.
"""
function ImagePairGeometry.blocksize_from_chunks(src::RasterInputs; floor::Int = 256)
    p = parent(src.dem)
    DiskArrays.isdisk(p) || return ImagePairGeometry.DEFAULT_BLOCKSIZE
    chunk = DiskArrays.approx_chunksize(DiskArrays.eachchunk(p))
    up(want, unit) = unit <= 0 ? want : cld(want, unit) * unit
    return (min(max(up(floor, chunk[1]), floor), size(src.dem, 1)),
            min(max(up(floor, chunk[2]), floor), size(src.dem, 2)))
end

"""
    ImagePairGeometry.write_geotiffs(dir, g::PairGeometry) -> Vector{String}

Write `g` as the reference's nine GeoTIFFs in `dir`, returning the paths written.

Names, band order, data types and nodata values match the reference, so a consumer built against its
output reads these unchanged. `Int32` bands are written as `Int32` and `Float64` as `Float64`, as the
reference writes `GDT_Int32` and `GDT_Float64`.

The two off2vel files get two bands for a projected result and three for a radar one, matching
`geogridOptical.cpp:461` and `geogridRadar.cpp:634,652`. The count comes from the result itself, via
`reference_files` — a reader indexes bands positionally, so an extra band would shift every band after
it without any error.

A file whose bands are entirely nodata is skipped, matching the reference: it writes no file at all
for an output its inputs did not support, and a consumer distinguishes "unsupported" from "computed
and empty" by the file's absence.
"""
function ImagePairGeometry.write_geotiffs(dir::AbstractString, g::PairGeometry)
    isdir(dir) || mkpath(dir)
    gt = g.geotransform
    nx, ny = size(g)
    written = String[]

    # Two-band off2vel files on the projected path, three on the radar path. `reference_files`
    # reads that off the result rather than taking a flag, so a caller cannot write the wrong count.
    for (filename, fields) in reference_files(g)
        bands = map(f -> getfield(g, f), fields)
        T = eltype(first(bands))
        sentinel = T(g.nodata.output)
        # The reference writes no file for an output its inputs did not support.
        all(b -> all(==(sentinel), b), bands) && continue

        path = joinpath(dir, filename)
        # Written through `Rasters.write` rather than by driving ArchGDAL directly. Setting a
        # geotransform on a GTiff opened for writing is silently ineffective under conditions that
        # are hard to pin down — the file ends up with GDAL's default while its projection, band data
        # and nodata all persist, so it looks georeferenced and is not. Handing Rasters a
        # georeferenced `Raster` puts that step where Rasters tests it.
        #
        # Lookups are `Intervals(Start())` to match what GDAL writes, and are built from the
        # geotransform's own low edge per axis — `Start` is the low edge in *coordinate* order, so a
        # negative step counts from the far end. This is `_edge_offset` in reverse.
        x0 = gt[1] + (gt[2] < 0 ? gt[2] : 0.0)
        y0 = gt[4] + (gt[6] < 0 ? gt[6] : 0.0)
        xdim = X(range(x0; step = gt[2], length = nx); sampling = Lk.Intervals(Lk.Start()))
        ydim = Y(range(y0; step = gt[6], length = ny); sampling = Lk.Intervals(Lk.Start()))

        # A multi-band file is one 3-D raster with a `Band` dimension, not a stack: `Rasters.write`
        # splits a stack into one file per layer, and the reference's consumers expect a single file
        # with its bands in order.
        cube = Array{T}(undef, nx, ny, length(bands))
        for (i, band) in enumerate(bands)
            cube[:, :, i] .= band
        end
        Rasters.write(path, Rasters.Raster(cube, (xdim, ydim, Rasters.Band(1:length(bands)));
                                           missingval = T(g.nodata.output),
                                           crs = _crs_of(g.crs)); force = true)
        isfile(path) || error("write_geotiffs did not produce $path")
        push!(written, path)
    end
    return written
end

# The CRS as something Rasters accepts: an EPSG integer becomes a `GFT.EPSG`, anything already a
# `GeoFormat` passes through, and `nothing` stays nothing.
_crs_of(::Nothing) = nothing
_crs_of(epsg::Integer) = GFT.EPSG(Int(epsg))
_crs_of(crs::GFT.GeoFormat) = crs

export RasterInputs

end
