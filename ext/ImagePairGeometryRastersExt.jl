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
                         INT_BANDS, FLOAT_BANDS, nodata_from, NoDataPolicy
using Rasters
using Rasters: AbstractRaster, Band
using DimensionalData
using ArchGDAL
using DiskArrays

const RA = Rasters
const GFT = Rasters.GeoFormatTypes

# Where a lookup coordinate sits within its pixel, as an offset from it to the pixel's outer edge.
#
# GDAL geotransforms and `MapGrid` put the origin at that edge, while an `ImageFootprint` origin is
# the pixel *center*. A raster read from a GeoTIFF is `Intervals{Start}`, so its lookups are already
# edges and the offset is zero — but one built in memory is typically `Points` or `Center`, and
# treating its coordinates as edges would displace the whole grid by half a pixel. So the locus is
# read rather than assumed.
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
    # spanning cells, which for a raster is the center convention.
    loc = try
        Lk.locus(l)
    catch
        Lk.Center()
    end
    return _locus_offset(loc, step)
end

"""
    ImagePairGeometry.mapgrid(dem::AbstractRaster) -> MapGrid

A [`MapGrid`](@ref) describing `dem`'s grid.

The grid's origin is the *outer edge* of the first pixel, the convention `MapGrid` and GDAL share. A
raster read from a GeoTIFF is `Intervals{Start}`, so its lookups are already edges; one built in
memory with a `Center` or point lookup is adjusted by half a pixel, since taking its coordinates for
edges would offset every output by that much.

A raster whose lookups are not regularly spaced is refused: the index arithmetic this package
reproduces assumes a constant step.
"""
function ImagePairGeometry.mapgrid(dem::AbstractRaster)
    x, y = dims(dem, X), dims(dem, Y)
    dx, dy = _step_of(x, :X), _step_of(y, :Y)
    return MapGrid(geotransform = (Float64(first(x)) + _edge_offset(lookup(x), dx), dx, 0.0,
                                   Float64(first(y)) + _edge_offset(lookup(y), dy), 0.0, dy),
                   size = (length(x), length(y)),
                   crs = crs(dem))
end

"""
    ImagePairGeometry.footprint(image::AbstractRaster) -> ImageFootprint

An [`ImageFootprint`](@ref) describing where `image` sits, for [`coregister`](@ref).

Origin is the first pixel's *center*, which is what `ImageFootprint` documents and what the
reference's `startingX`/`startingY` are — so a lookup holding edges is shifted half a pixel inward.

Only the geometry is read, never the pixels, so this is cheap on a disk-backed scene and needs no
data at all.
"""
function ImagePairGeometry.footprint(image::AbstractRaster)
    x, y = dims(image, X), dims(image, Y)
    dx, dy = _step_of(x, :X), _step_of(y, :Y)
    # From the edge to the center: undo the edge offset, then add half a pixel.
    cx = Float64(first(x)) + _edge_offset(lookup(x), dx) + dx / 2
    cy = Float64(first(y)) + _edge_offset(lookup(y), dy) + dy / 2
    return ImageFootprint(origin = (cx, cy), spacing = (dx, dy),
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

Accepts the same eleven optional rasters as [`GeometryInputs`](@ref), and each may be lazy: a
disk-backed `Raster` is read one window at a time, so a grid larger than memory is never
materialized. All must share the grid the geometry is computed on.

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

function RasterInputs(; dem, dhdx = nothing, dhdy = nothing, vx = nothing, vy = nothing,
                      srx = nothing, sry = nothing, csminx = nothing, csminy = nothing,
                      csmaxx = nothing, csmaxy = nothing, ssm = nothing)
    sz = size(dem)
    for (name, r) in ((:dhdx, dhdx), (:dhdy, dhdy), (:vx, vx), (:vy, vy), (:srx, srx),
                      (:sry, sry), (:csminx, csminx), (:csminy, csminy), (:csmaxx, csmaxx),
                      (:csmaxy, csmaxy), (:ssm, ssm))
        r === nothing && continue
        size(r) == sz || throw(DimensionMismatch(
            "RasterInputs `$name` is $(size(r)) but `dem` is $sz; every input must be on the " *
            "grid the geometry is computed on"))
    end
    return RasterInputs(dem, dhdx, dhdy, vx, vy, srx, sry, csminx, csminy, csmaxx, csmaxy, ssm)
end

# `view` then `copyto!` is a chunk-aware windowed read for any DiskArrays-backed parent, so one
# implementation serves every backend. `Float64` because the kernel works in it throughout and the
# reference reads every input as `GDT_Float64` regardless of the file's own type.
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

Returns [`DEFAULT_BLOCKSIZE`](@ref) when no input is disk-backed, since then there is no chunking to
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

A file whose bands are entirely nodata is skipped, matching the reference: it writes no file at all
for an output its inputs did not support, and a consumer distinguishes "unsupported" from "computed
and empty" by the file's absence.
"""
function ImagePairGeometry.write_geotiffs(dir::AbstractString, g::PairGeometry)
    isdir(dir) || mkpath(dir)
    gt = g.geotransform
    nx, ny = size(g)
    written = String[]

    for (filename, fields) in REFERENCE_FILES
        bands = map(f -> getfield(g, f), fields)
        T = eltype(first(bands))
        sentinel = T(g.nodata.output)
        # The reference writes no file for an output its inputs did not support.
        all(b -> all(==(sentinel), b), bands) && continue

        path = joinpath(dir, filename)
        # Built in an in-memory dataset and copied out, rather than written into a GTiff directly.
        # A GTiff created for writing accepts a geotransform only under conditions that are easy to
        # violate silently — the file then carries GDAL's default `(0, 1, 0, 0, 0, 1)` while the
        # projection reads back correctly, so the output looks georeferenced and is not. Copying from
        # MEM sidesteps that: the geotransform is set on a dataset that always accepts it, and
        # `CreateCopy` carries it across.
        ArchGDAL.create(""; driver = ArchGDAL.getdriver("MEM"),
                        width = nx, height = ny, nbands = length(bands), dtype = T) do mem
            ArchGDAL.setgeotransform!(mem, collect(gt))
            _setcrs!(mem, g.crs)
            for (i, band) in enumerate(bands)
                b = ArchGDAL.getband(mem, i)
                ArchGDAL.setnodatavalue!(b, Float64(g.nodata.output))
                # ArchGDAL indexes (x, y); the bands are already in that order.
                ArchGDAL.write!(b, band)
            end
            ArchGDAL.destroy(ArchGDAL.copy(mem; filename = path,
                                           driver = ArchGDAL.getdriver("GTiff")))
        end
        push!(written, path)
    end
    return written
end

# The CRS reaches here as an EPSG integer, a `GeoFormatTypes` object of any flavour, or nothing.
# `convert(WellKnownText, x)` handles the GeoFormatTypes cases — wrapping in `WellKnownText(x)`
# instead builds a nested object GDAL cannot read.
_setcrs!(ds, ::Nothing) = nothing
_setcrs!(ds, epsg::Integer) =
    ArchGDAL.setproj!(ds, ArchGDAL.toWKT(ArchGDAL.importEPSG(Int(epsg))))
_setcrs!(ds, crs::GFT.WellKnownText) = ArchGDAL.setproj!(ds, GFT.val(crs))
_setcrs!(ds, crs::GFT.EPSG) =
    ArchGDAL.setproj!(ds, ArchGDAL.toWKT(ArchGDAL.importEPSG(Int(GFT.val(crs)[1]))))
_setcrs!(ds, crs::GFT.GeoFormat) =
    ArchGDAL.setproj!(ds, GFT.val(convert(GFT.WellKnownText, crs)))

export RasterInputs

end
