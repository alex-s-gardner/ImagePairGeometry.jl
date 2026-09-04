# The two image coordinate systems, and the grid the geometry is computed on.
#
# A grid point is mapped into image coordinates one of two ways, and which one is the top-level
# dispatch of the package:
#
#   `ProjectedCoordinate`  the image is on a map projection, so the mapping is a coordinate
#                          transform plus an affine index calculation. Optical imagery
#                          (Landsat, Sentinel-2) is the usual case, but the type is named for
#                          the coordinate system rather than the sensor because a
#                          geocoded/terrain-corrected radar product also belongs here.
#
#   `RadarCoordinate`      the image is in slant-range/azimuth, so the mapping needs orbit
#                          state vectors and a range-Doppler solve.
#
# `ProjectedCoordinate` describes the *coregistered intersection* of the pair, not either image
# on its own — see `coregister` in `pair.jl` for why, and for what goes wrong if it is built
# from the reference image alone.

"""
    _as_geoformat(crs) -> Union{Nothing,GFT.GeoFormat}

A CRS in the form the GeoJulia packages pass around: a `GeoFormatTypes.GeoFormat`, or `nothing`.

An `Integer` is read as an EPSG code, since that is how a geogrid case is usually specified and how
the reference's own parameters arrive. Anything already a `GeoFormat` passes through, so a caller
holding a WKT or a PROJ string is not forced to convert. Normalising here rather than at each
consumer means `GeoInterface.crs` has one type to return and the Rasters extension has one to write.
"""
_as_geoformat(::Nothing) = nothing
_as_geoformat(epsg::Integer) = GFT.EPSG(Int(epsg))
_as_geoformat(crs::GFT.GeoFormat) = crs
_as_geoformat(x) = throw(ArgumentError(
    "crs must be a GeoFormatTypes.GeoFormat, an Integer EPSG code, or nothing; got $(typeof(x))"))

"""
    AbstractImageCoordinate

The coordinate system of an image, determining how a map-projected grid point maps to a pixel
index in it.

Subtypes: [`ProjectedCoordinate`](@ref), [`RadarCoordinate`](@ref).

The interface a subtype provides: [`nsamples`](@ref) and [`nlines`](@ref) for the image size,
[`xsize`](@ref) and [`ysize`](@ref) for the ground pixel size along each axis, and a
[`footprint_bounds`](@ref) method. That is what [`CoregisteredPair`](@ref) and the chip-size
conversion consume, and it is deliberately small: the two coordinate systems have almost nothing in
common structurally — one is an origin, a spacing and a size, the other a range axis, a time axis and
an orbit — so what they share is the handful of quantities the kernel actually asks for rather than a
common representation.
"""
abstract type AbstractImageCoordinate end

"""
    ProjectedCoordinate(; origin, spacing, size)

An image on a map projection, indexed by an affine relation between projected coordinates and
pixel indices.

# Fields
- `origin`: projected coordinate of the *first pixel's center*, as `(x, y)`. Geogrid's
  `startingX`/`startingY`.
- `spacing`: signed pixel size, as `(x, y)`. Negative `y` for the usual north-up raster. Signed
  rather than absolute because the index calculation divides by it, so the sign carries the axis
  direction.
- `size`: `(samples, lines)`, i.e. `(ncolumns, nrows)`.

For an image *pair* these describe the overlap of the two images, which is what
[`coregister`](@ref) computes and what pixel indices are relative to. Constructing one from a
single image's geotransform is correct only when that image is the overlap.

The CRS is deliberately absent: it enters through the coordinate transform passed to the kernel,
so the same coordinate system can be paired with a transform from any grid CRS.
"""
struct ProjectedCoordinate{T<:Real} <: AbstractImageCoordinate
    origin::NTuple{2,T}
    spacing::NTuple{2,T}
    size::NTuple{2,Int}

    function ProjectedCoordinate{T}(origin, spacing, size) where {T<:Real}
        all(>(0), size) || throw(ArgumentError(
            "ProjectedCoordinate size must be positive, got $size"))
        all(!iszero, spacing) || throw(ArgumentError(
            "ProjectedCoordinate spacing must be nonzero, got $spacing"))
        all(isfinite, origin) || throw(ArgumentError(
            "ProjectedCoordinate origin must be finite, got $origin"))
        all(isfinite, spacing) || throw(ArgumentError(
            "ProjectedCoordinate spacing must be finite, got $spacing"))
        return new{T}(origin, spacing, size)
    end
end

function ProjectedCoordinate(origin::NTuple{2}, spacing::NTuple{2}, size::NTuple{2,Integer})
    T = promote_type(map(typeof, origin)..., map(typeof, spacing)...)
    return ProjectedCoordinate{T}(T.(origin), T.(spacing), Int.(size))
end

ProjectedCoordinate(; origin, spacing, size) = ProjectedCoordinate(origin, spacing, size)

"""
    xsize(c::AbstractImageCoordinate) -> Float64
    ysize(c::AbstractImageCoordinate) -> Float64

Ground pixel size along each image axis, in meters — geogrid's `X_res`/`Y_res`, reported in its
scalar output and used downstream to convert a chip size in meters to pixels.

Part of the [`AbstractImageCoordinate`](@ref) interface. A projected image reads them off its
spacing; a radar image derives them from the incidence angle and the platform speed, since it has no
geotransform to read. Both are absolute, so the axis direction is not carried here.
"""
function xsize end, function ysize end

"""
    xsize(c::ProjectedCoordinate)
    ysize(c::ProjectedCoordinate)

The absolute value of the signed pixel spacing along each axis.
"""
xsize(c::ProjectedCoordinate) = abs(c.spacing[1])
ysize(c::ProjectedCoordinate) = abs(c.spacing[2])

"""
    spacing(c::ProjectedCoordinate) -> NTuple{2,Float64}

The nominal pixel spacing as a `Float64` pair, **signed** — negative `y` for a north-up raster.

What the output expressions divide by, so the sign matters: [`xsize`](@ref) and [`ysize`](@ref) are
the absolute values and are what the chip-size conversion wants instead.

Not part of the [`AbstractImageCoordinate`](@ref) interface. A radar image has no single spacing to
return — its along-track step is per grid point — so the kernel takes the pair as an argument and each
path supplies it its own way.
"""
spacing(c::ProjectedCoordinate) = (Float64(c.spacing[1]), Float64(c.spacing[2]))

"""
    nsamples(c::AbstractImageCoordinate) -> Int
    nlines(c::AbstractImageCoordinate) -> Int

Image width and height in pixels — columns and rows for a projected image, range samples and azimuth
lines for a radar one.

Part of the [`AbstractImageCoordinate`](@ref) interface, and what the bounds test on a computed pixel
index compares against.
"""
function nsamples end, function nlines end

"""
    nsamples(c::ProjectedCoordinate)
    nlines(c::ProjectedCoordinate)

The two components of the coordinate's `size`.
"""
nsamples(c::ProjectedCoordinate) = c.size[1]
nlines(c::ProjectedCoordinate) = c.size[2]
