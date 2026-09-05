module ImagePairGeometryFastGeoProjectionsExt

# Native-Julia coordinate transforms, for the CRS pairs FastGeoProjections implements.
#
# The kernel makes three transform calls per grid point and the projection library is around 88% of a
# cross-CRS run (`benchmark/cost_share.jl`), so the transform is what a cross-CRS run costs. Measured
# on the kernel's own call pattern — one forward and two inverse per point, EPSG:3413 to EPSG:32624 —
# a native transformation is 178 ns/point against PROJ's 445.
#
# The two do not agree bitwise: 7.4e-7 m separates them across the Greenland domain at three heights,
# which is 7e-13 relative against coordinates of ~1e6 m. That sits inside the ≤1e-7 relative bound
# `REFERENCE.md` sets for the Float64 output bands, and it moves no rounded pixel index — 0 of 320,000
# checked at 10 m and at 2.3 m spacing — so every integer band stays bitwise. PROJ remains what the
# reference fixtures are generated and asserted against; this is an alternative for a production run,
# not a replacement for the comparison.
#
# Nothing here needs a per-task anything. A `FastGeoProjections.Transformation` over a native pair is
# immutable and holds no PROJ state, so one object serves every thread — which is why this offers a
# `TransformPair` directly where the PROJ path has to offer a factory.

using ImagePairGeometry
using ImagePairGeometry: TransformPair, IdentityTransform, transform_pair
using FastGeoProjections: FastGeoProjections, EPSG

"""
    FastTransform(source, target; always_xy = false)

A coordinate transform between two EPSG codes, evaluated by FastGeoProjections.

Callable as `(x, y, z) -> (x′, y′, z′)`, the interface
[`ImagePairGeometry.AbstractCoordTransform`](@ref) defines. The third coordinate is carried through
unchanged: both CRSs of a projected pair share a datum, so a height is not a quantity either
projection transforms.

`always_xy = false` matches the PROJ path, which reproduces the reference's
`osr.CoordinateTransformation` on SRSs from `ImportFromEPSG` — authority axis order.
"""
struct FastTransform{F} <: ImagePairGeometry.AbstractCoordTransform
    f::F
    source::Int
    target::Int
    always_xy::Bool
end

function FastTransform(source::Integer, target::Integer; always_xy::Bool = false)
    f = FastGeoProjections.Transformation(EPSG(Int(source)), EPSG(Int(target)); always_xy)
    return FastTransform(f, Int(source), Int(target), always_xy)
end

@inline (t::FastTransform)(x, y, z) = (t.f(x, y)..., z)

ImagePairGeometry.inverse(t::FastTransform) =
    FastTransform(t.target, t.source; always_xy = t.always_xy)

Base.show(io::IO, t::FastTransform) =
    print(io, "FastTransform(EPSG:", t.source, " => EPSG:", t.target,
          "; always_xy = ", t.always_xy, ")")

"""
    ImagePairGeometry.fast_transform(grid_crs, image_crs; always_xy = false) -> TransformPair

A [`TransformPair`](@ref) between two EPSG codes, evaluated by FastGeoProjections.

`grid_crs` and `image_crs` are EPSG codes — integers, or `EPSG` objects. Pass the result as
`pairgeometry`'s or `pairgeometry_blocked`'s `transform`; unlike the PROJ path it needs no factory,
because the transformation holds no per-thread state.

Equal codes give an [`IdentityTransform`](@ref) rather than a projection composed with its own
inverse. PROJ resolves that case to a `+proj=noop` pipeline and returns its input unchanged, and the
same-CRS fixtures asserted against it are bitwise on every band; composing unproject with project
would instead land 5.2e-7 m away and break them.

```julia
tf = fast_transform(3413, 32624)
r = pairgeometry_blocked(grid, pair, source; transform = tf, ntasks = 8)
```
"""
function ImagePairGeometry.fast_transform(grid_crs, image_crs; always_xy::Bool = false)
    g, i = _epsg_code(grid_crs), _epsg_code(image_crs)
    g == i && return transform_pair(IdentityTransform())
    return TransformPair(FastTransform(g, i; always_xy),
                         FastTransform(i, g; always_xy))
end

_epsg_code(x::Integer) = Int(x)
_epsg_code(x::EPSG) = Int(first(x.val))
_epsg_code(x) = throw(ArgumentError(
    "fast_transform takes EPSG codes as integers or EPSG objects, got $(typeof(x)). " *
    "FastGeoProjections looks a transformation up by code rather than parsing a CRS " *
    "description, so a WKT or PROJ string has no code to resolve; use proj_transform for those."))

export FastTransform

end
