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
    AbstractImageCoordinate

The coordinate system of an image, determining how a map-projected grid point maps to a pixel
index in it.

Subtypes: [`ProjectedCoordinate`](@ref), [`RadarCoordinate`](@ref).
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
    RadarCoordinate

An image in radar slant-range/azimuth coordinates.

Not yet implemented: constructing one throws. The type exists so that the dispatch on
[`AbstractImageCoordinate`](@ref) is fixed while only the projected path is available, and so
that a caller reaching for radar gets a clear error rather than a `MethodError` on a type that
does not exist.
"""
struct RadarCoordinate <: AbstractImageCoordinate
    function RadarCoordinate(args...; kw...)
        throw(ArgumentError(
            "RadarCoordinate is not implemented yet: only ProjectedCoordinate is available. " *
            "The radar path needs orbit interpolation and a range-Doppler solve; see the " *
            "roadmap in README.md."))
    end
end

"""
    xsize(c::ProjectedCoordinate)
    ysize(c::ProjectedCoordinate)

Absolute pixel size along each axis — geogrid's `X_res`/`Y_res`, reported in its scalar output
and used downstream to convert a chip size in meters to pixels.
"""
xsize(c::ProjectedCoordinate) = abs(c.spacing[1])
ysize(c::ProjectedCoordinate) = abs(c.spacing[2])

"""
    nsamples(c::ProjectedCoordinate)
    nlines(c::ProjectedCoordinate)

Image width and height in pixels.
"""
nsamples(c::ProjectedCoordinate) = c.size[1]
nlines(c::ProjectedCoordinate) = c.size[2]
