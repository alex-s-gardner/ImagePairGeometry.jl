# The pair: what the two images jointly determine.
#
# It is tempting to treat the reference image as supplying the geometry and the secondary as
# supplying only the time separation. That is wrong, and getting it wrong misplaces every pixel
# index in the output.
#
# `coregister` computes the *intersection* of the two footprints, and the image coordinate system
# the geometry is computed against is that intersection: its north-west corner becomes
# `startingX`/`startingY` and its extent becomes `numberOfSamples`/`numberOfLines`
# (`GeogridOptical.py:277-345`, consumed at `testGeogrid.py:283-306`). Both come from
# `max`/`min` over the two images, so the secondary moves the origin and resizes the grid. It
# also yields each image's own pixel offset into the overlap, which a caller needs in order to
# read the right pixels when correlating.

"""
    ImageFootprint(; origin, spacing, size, crs = nothing)

Where one image sits in projected coordinates: enough to intersect two images without reading
pixels.

`origin` is the projected coordinate of the first pixel's center and `spacing` is the signed
pixel size, matching a GDAL geotransform's `(gt[1], gt[4])` and `(gt[2], gt[6])`. See
[`ProjectedCoordinate`](@ref), whose fields have the same meaning.

`crs` is optional and used for one purpose: [`coregister`](@ref) refuses a pair whose two footprints
declare *different* CRSs. Intersecting footprints is pure arithmetic in one coordinate system, so two
images in different systems have no meaningful overlap — the numbers come out plausible and describe
nowhere. The reference makes that check by opening both images with GDAL and comparing EPSG codes; here
it is available when the caller supplies the codes and skipped when it cannot be made, which is what
`REFERENCE.md` records as the divergence.

Given as an `Integer` it is read as an EPSG code, as [`MapGrid`](@ref) reads one. `nothing` — the default,
and what every footprint built before this argument existed has — means the comparison is the caller's,
and only the pixel geometry is checked.
"""
struct ImageFootprint{T<:Real,C}
    origin::NTuple{2,T}
    spacing::NTuple{2,T}
    size::NTuple{2,Int}
    crs::C

    # Normalizing here rather than only in the outer constructor: the default inner constructor would
    # otherwise accept an `Integer` and store it raw, and then `32607` and `GFT.EPSG(32607)` — the same
    # CRS — would compare unequal and `coregister` would refuse a pair it should accept.
    function ImageFootprint{T,C}(origin, spacing, size, crs) where {T<:Real,C}
        c = _as_geoformat(crs)
        return new{T,typeof(c)}(origin, spacing, size, c)
    end
end

function ImageFootprint(origin::NTuple{2}, spacing::NTuple{2}, size::NTuple{2,Integer},
                        crs = nothing)
    T = promote_type(map(typeof, origin)..., map(typeof, spacing)...)
    c = _as_geoformat(crs)
    return ImageFootprint{T,typeof(c)}(T.(origin), T.(spacing), Int.(size), c)
end

ImageFootprint(; origin, spacing, size, crs = nothing) =
    ImageFootprint(origin, spacing, size, crs)

GeoInterface.crs(f::ImageFootprint) = f.crs

# Two CRSs agree, or one of them is unknown.
#
# Unknown is not taken as agreement — it is taken as "not checked here", which is the honest reading and
# is what every caller before `crs` existed is doing. Comparing the `GeoFormat`s directly rather than
# resolving them: two spellings of one CRS (an EPSG code and its WKT) would compare unequal, so this
# refuses a pair a projection library would accept. That is the conservative direction — it asks the
# caller to be consistent rather than silently accepting a mismatch — and resolving them properly would
# mean a projection library in the core, which this package does not have.
_check_crs(::Nothing, ::Nothing) = nothing
_check_crs(::Any, ::Nothing) = nothing
_check_crs(::Nothing, ::Any) = nothing
function _check_crs(a, b)
    a == b || throw(ArgumentError(
        "coregister requires both images in one coordinate reference system, but the reference " *
        "declares $a and the secondary $b. Intersecting two footprints is arithmetic in a single " *
        "system, so the overlap of images in different ones is not meaningful. Reproject one before " *
        "pairing. Two spellings of the same CRS also compare unequal here, since resolving them would " *
        "need a projection library; pass them in the same form."))
    return nothing
end

"""
    CoregisteredPair

A pair of acquisitions, the image coordinate system their geometry is computed against, and their
time separation.

# Fields
- `coordinate`: the image coordinate system pixel indices are relative to — any
  [`AbstractImageCoordinate`](@ref). For a projected pair whose overlap was computed this is that
  overlap, not either input image; for a radar pair it is the reference acquisition outright, since
  `testGeogrid.py:427-470` takes every radar parameter from image 1.
- `secondary_offset_coordinate`: the secondary acquisition's own coordinate system, or `nothing`.
  Only [`pixel_offset`](@ref) reads it; the geometry kernel does not, so a pair without one produces
  every output it produces today. See below.
- `reference_offset`, `secondary_offset`: the `(column, row)` index of the overlap's first pixel
  within each input image, zero-based as the reference reports them. A caller reads its image
  windows at these offsets. Both zero when `coordinate` is already the window — see the
  one-argument constructor.
- `dt`: time separation in seconds, secondary minus reference. Enters the geometry as the
  factor converting velocity to displacement.

Built by [`coregister`](@ref) when the overlap has to be computed, or directly from a coordinate
when it does not.

# Why the secondary's coordinate system is carried

`coordinate` describes where a grid point falls in the *reference* image, and the kernel hands that
one pixel index to a correlator for both images. That is right only where the secondary sits on the
reference's grid. Two radar acquisitions have different orbits, so it does not: the same ground point
falls at a different range sample and azimuth line in each, and recovering the difference needs the
secondary's own orbit, range origin, range spacing and PRF. [`pixel_offset`](@ref) computes it.

`nothing` means no secondary geometry was supplied, which is every pair built from a single
coordinate. It is not an error: the geometry outputs do not depend on it, so only `pixel_offset`
objects, and it says what to supply.

# What a supplied secondary must satisfy

Checked here rather than at first use, since each is a comparison of a few numbers and
`pixel_offset`'s field is lazy — a pair that cannot be answered would otherwise fail inside a block
loop rather than at the call that built it.

- The same *type* as `coordinate`. The offset is a pixel shift in the reference's axes, and that only
  describes two images indexed the same way; a radar reference against a projected secondary shares
  no such frame.
- For two [`ProjectedCoordinate`](@ref)s, matching `spacing`, as [`coregister`](@ref) requires of two
  footprints and for the same reason.
"""
struct CoregisteredPair{C<:AbstractImageCoordinate,S}
    coordinate::C
    secondary_offset_coordinate::S
    reference_offset::NTuple{2,Int}
    secondary_offset::NTuple{2,Int}
    dt::Float64

    function CoregisteredPair{C,S}(coordinate, secondary, reference_offset, secondary_offset,
                                   dt) where {C<:AbstractImageCoordinate,S}
        _check_secondary(coordinate, secondary)
        return new{C,S}(coordinate, secondary, reference_offset, secondary_offset, dt)
    end
end

function CoregisteredPair(coordinate::AbstractImageCoordinate, secondary,
                          reference_offset::NTuple{2,Integer},
                          secondary_offset::NTuple{2,Integer}, dt::Real)
    return CoregisteredPair{typeof(coordinate),typeof(secondary)}(
        coordinate, secondary, Int.(reference_offset), Int.(secondary_offset), Float64(dt))
end

# No secondary is the state every pair built before `pixel_offset` existed is in, so nothing to check.
_check_secondary(::AbstractImageCoordinate, ::Nothing) = nothing

# Mismatched kinds. A named error rather than a `MethodError`, because the argument is a caller's own
# coordinate and the reason it cannot be paired is worth stating. The same-kind methods that accept a
# pair, and whatever else each path requires of one, are in `misregistration.jl` beside the
# `pixel_offset` methods they are preconditions for — `RadarCoordinate` does not exist yet at this point
# in the load order, and the checks belong with the operation they protect in any case.
_check_secondary(c::AbstractImageCoordinate, s::AbstractImageCoordinate) = throw(ArgumentError(
    "a CoregisteredPair's secondary coordinate must be the same kind as its reference, but the " *
    "reference is a $(nameof(typeof(c))) and the secondary a $(nameof(typeof(s))). A pixel offset " *
    "between the two is a shift in the reference's own axes, and these two are not indexed in a " *
    "common frame, so no such shift exists."))

"""
    CoregisteredPair(coordinate; dt, secondary = nothing)

A pair whose image coordinate system is already known, with both offsets zero.

For a [`RadarCoordinate`](@ref), and for a [`ProjectedCoordinate`](@ref) the caller built directly
from a view of an overlap it has already sliced. In both cases the coordinate *is* the window, so
there is no offset into a larger image to record — which is what the zeros mean, rather than "the
overlap happens to start at the origin".

`secondary` is the secondary acquisition's own coordinate system, needed only by
[`pixel_offset`](@ref). It defaults to `nothing`, which is what every pair not computing an offset
wants; see [`CoregisteredPair`](@ref) for what a supplied one must satisfy.

[`coregister`](@ref) is the other way to obtain a pair, for the case that needs it: two footprints
whose intersection has to be computed, which is also where nonzero offsets come from.
"""
CoregisteredPair(coordinate::AbstractImageCoordinate; dt::Real, secondary = nothing) =
    CoregisteredPair(coordinate, secondary, (0, 0), (0, 0), Float64(dt))

"""
    coregister(reference::ImageFootprint, secondary::ImageFootprint; dt) -> CoregisteredPair

Intersect two image footprints, giving the overlap's coordinate system and each image's offset
into it.

Reproduces `GeogridOptical.coregister` (`GeogridOptical.py:277-345`) including its four failure
conditions, which are errors rather than clamps: the reference refuses a pair it cannot align
instead of silently processing a partial overlap.

The overlap is `W = max(x₁, x₂)`, `N = min(y₁, y₂)`, `E = min(east₁, east₂)`,
`S = max(south₁, south₂)`, where each image's far edge is its *last pixel center*
(`origin + (size - 1) * spacing`) rather than its outer boundary. Index arithmetic uses
[`nround`](@ref) — half to even — because the reference computes it in NumPy, not in the C++
kernel where `std::round` applies.

`dt` is seconds, secondary minus reference. The reference derives it from calendar dates only
(`testGeogrid.py:351-354`), so it is always a whole number of days there; nothing here requires
that.

Throws `ArgumentError` if the two spacings differ, if the overlap is empty, or if any of the
reference's four conditions fail.

# Example

```jldoctest
julia> using ImagePairGeometry

julia> a = ImageFootprint(origin = (100.0, 900.0), spacing = (10.0, -10.0), size = (50, 50));

julia> b = ImageFootprint(origin = (150.0, 950.0), spacing = (10.0, -10.0), size = (50, 50));

julia> p = coregister(a, b; dt = 86400.0);

julia> p.coordinate.origin, p.coordinate.size
((150.0, 900.0), (45, 45))

julia> p.reference_offset, p.secondary_offset
((5, 0), (0, 5))
```
"""
function coregister(reference::ImageFootprint, secondary::ImageFootprint; dt::Real)
    t1, t2 = reference, secondary

    # The reference compares EPSG codes and refuses a mismatch. That check is available here when both
    # footprints carry a CRS and is skipped when either does not, since a footprint's CRS is optional —
    # the arithmetic below needs none, which is what keeps this function testable without GDAL. Recorded
    # in REFERENCE.md as a deliberate divergence.
    _check_crs(t1.crs, t2.crs)
    t1.spacing == t2.spacing || throw(ArgumentError(
        "coregister requires matching pixel spacing: reference $(t1.spacing) vs secondary " *
        "$(t2.spacing). Reproject one image before pairing."))

    # Overlap bounds. Far edges are last-pixel centers, per the reference.
    W = max(t1.origin[1], t2.origin[1])
    N = min(t1.origin[2], t2.origin[2])
    E = min(t1.origin[1] + (t1.size[1] - 1) * t1.spacing[1],
            t2.origin[1] + (t2.size[1] - 1) * t2.spacing[1])
    S = max(t1.origin[2] + (t1.size[2] - 1) * t1.spacing[2],
            t2.origin[2] + (t2.size[2] - 1) * t2.spacing[2])

    x1a = Int(nround((W - t1.origin[1]) / t1.spacing[1]))
    x1b = Int(nround((E - t1.origin[1]) / t1.spacing[1]))
    y1a = Int(nround((N - t1.origin[2]) / t1.spacing[2]))
    y1b = Int(nround((S - t1.origin[2]) / t1.spacing[2]))

    x2a = Int(nround((W - t2.origin[1]) / t2.spacing[1]))
    x2b = Int(nround((E - t2.origin[1]) / t2.spacing[1]))
    y2a = Int(nround((N - t2.origin[2]) / t2.spacing[2]))
    y2b = Int(nround((S - t2.origin[2]) / t2.spacing[2]))

    # The reference's three checks, in its order and with its intent. Its messages are terse to
    # the point of being unhelpful, so these say which value offended.
    if x1a > t1.size[1] - 1 || x1b > t1.size[1] - 1 || y1a > t1.size[2] - 1 ||
       y1b > t1.size[2] - 1 || x2a > t2.size[1] - 1 || x2b > t2.size[1] - 1 ||
       y2a > t2.size[2] - 1 || y2b > t2.size[2] - 1
        throw(ArgumentError(
            "coregistered index exceeds image bounds: reference ($x1a:$x1b, $y1a:$y1b) in " *
            "size $(t1.size), secondary ($x2a:$x2b, $y2a:$y2b) in size $(t2.size). The two " *
            "images most likely do not overlap."))
    end
    if x1a < 0 || x1b < 0 || y1a < 0 || y1b < 0 || x2a < 0 || x2b < 0 || y2a < 0 || y2b < 0
        throw(ArgumentError(
            "coregistered index is negative: reference ($x1a:$x1b, $y1a:$y1b), secondary " *
            "($x2a:$x2b, $y2a:$y2b). The two images most likely do not overlap."))
    end
    if (x1b - x1a) != (x2b - x2a) || (y1b - y1a) != (y2b - y2a)
        throw(ArgumentError(
            "coregistered overlap differs between images: reference " *
            "$(x1b - x1a + 1)x$(y1b - y1a + 1), secondary $(x2b - x2a + 1)x$(y2b - y2a + 1). " *
            "Pixel grids are offset by a fraction of a pixel."))
    end

    # The reference produces a zero-width overlap silently and hands it to GDAL, which then
    # creates a degenerate dataset. Fail here instead — see REFERENCE.md.
    nx, ny = x1b - x1a + 1, y1b - y1a + 1
    (nx > 0 && ny > 0) || throw(ArgumentError(
        "coregistered overlap is empty: $(nx)x$(ny) pixels. The two images do not overlap."))

    # No secondary coordinate: `coord` is the overlap, which both images share by construction, so
    # there is no per-image geometry left to differ. See `CoregisteredPair`.
    coord = ProjectedCoordinate((W, N), t1.spacing, (nx, ny))
    return CoregisteredPair(coord, nothing, (x1a, y1a), (x2a, y2a), Float64(dt))
end
