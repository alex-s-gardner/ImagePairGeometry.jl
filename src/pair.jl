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
    ImageFootprint(; origin, spacing, size)

Where one image sits in projected coordinates: enough to intersect two images without reading
pixels.

`origin` is the projected coordinate of the first pixel's center and `spacing` is the signed
pixel size, matching a GDAL geotransform's `(gt[1], gt[4])` and `(gt[2], gt[6])`. See
[`ProjectedCoordinate`](@ref), whose fields have the same meaning.
"""
struct ImageFootprint{T<:Real}
    origin::NTuple{2,T}
    spacing::NTuple{2,T}
    size::NTuple{2,Int}
end

function ImageFootprint(origin::NTuple{2}, spacing::NTuple{2}, size::NTuple{2,Integer})
    T = promote_type(map(typeof, origin)..., map(typeof, spacing)...)
    return ImageFootprint{T}(T.(origin), T.(spacing), Int.(size))
end

ImageFootprint(; origin, spacing, size) = ImageFootprint(origin, spacing, size)

"""
    CoregisteredPair

The overlap of two co-registered images, and the offsets into each.

# Fields
- `coordinate`: the overlap as a [`ProjectedCoordinate`](@ref). Pixel indices the geometry
  produces are relative to this, not to either input image.
- `reference_offset`, `secondary_offset`: the `(column, row)` index of the overlap's first pixel
  within each input image, zero-based as the reference reports them. A caller reads its image
  windows at these offsets.
- `dt`: time separation in seconds, secondary minus reference. Enters the geometry as the
  factor converting velocity to displacement.

Built by [`coregister`](@ref).
"""
struct CoregisteredPair{T<:Real}
    coordinate::ProjectedCoordinate{T}
    reference_offset::NTuple{2,Int}
    secondary_offset::NTuple{2,Int}
    dt::Float64
end

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

    # The reference compares EPSG codes and refuses a mismatch. Footprints carry no CRS, so the
    # equivalent check available here is that the pixel geometry agrees; a caller holding CRSs
    # must compare them itself. Recorded in REFERENCE.md as a deliberate divergence.
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

    coord = ProjectedCoordinate((W, N), t1.spacing, (nx, ny))
    return CoregisteredPair(coord, (x1a, y1a), (x2a, y2a), Float64(dt))
end
