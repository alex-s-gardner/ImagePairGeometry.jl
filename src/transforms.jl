# Coordinate transforms, as a dispatch rather than a branch.
#
# The kernel needs three transforms per grid point: one forward, grid CRS to image CRS, and two
# inverse, for points a pixel away along each image axis. The transform is a parameter rather than
# a hardcoded PROJ call for two reasons.
#
# Testing: `AffineTransform` has a closed form, so the kernel's arithmetic can be checked against
# hand-derived exact values with no PROJ involved and no reference to compare against.
#
# Speed: when the grid and image share a CRS the transform is the identity, and then the axis
# difference vectors are exactly `(spacing, 0, 0)` and `(0, spacing, 0)`, the unit vectors are
# exactly axis-aligned, and the scale factors are exactly 1. Dispatching on `IdentityTransform`
# lets the kernel skip three PROJ calls per point — measured at ~800 ns each for a real
# reprojection — while running the same code path.

"""
    AbstractCoordTransform

A transform between coordinate systems, callable as `t(x, y, z) -> (x′, y′, z′)`.

Subtypes: [`IdentityTransform`](@ref), [`AffineTransform`](@ref). A `Proj.Transformation` also
satisfies the interface and is what a georeferenced caller passes; it is not a subtype because it
belongs to another package.
"""
abstract type AbstractCoordTransform end

"""
    IdentityTransform()

The transform between a coordinate system and itself.

Returns its argument unchanged, which is what PROJ's `+proj=noop` pipeline does for two identical
CRSs — bit for bit, including the vertical coordinate. Dispatching on this type rather than
calling PROJ makes the same-CRS case exact by construction and removes three transform calls per
grid point.
"""
struct IdentityTransform <: AbstractCoordTransform end

@inline (::IdentityTransform)(x, y, z) = (x, y, z)

"""
    inverse(t::AbstractCoordTransform)

The transform in the opposite direction.

The kernel needs both directions — forward to find a pixel index, inverse to find where a
one-pixel step in the image lands on the grid — and takes them as one object so a caller cannot
supply a mismatched pair.
"""
inverse(::IdentityTransform) = IdentityTransform()

"""
    AffineTransform(; a, b, c, d, e, f, scale_z = 1.0)

The planar affine map `(x, y) -> (a*x + b*y + c, d*x + e*y + f)`, with `z` scaled by `scale_z`.

Exists for testing: a rotation, scaling and translation has an exactly known inverse and exactly
known axis unit vectors, so it pins the kernel's arithmetic without reference to PROJ or to the
reference implementation. Not a substitute for a map projection — it cannot represent one.
"""
struct AffineTransform{T<:Real} <: AbstractCoordTransform
    a::T; b::T; c::T
    d::T; e::T; f::T
    scale_z::T

    function AffineTransform{T}(a, b, c, d, e, f, scale_z) where {T<:Real}
        det = a * e - b * d
        iszero(det) && throw(ArgumentError(
            "AffineTransform is singular: a*e - b*d = 0 for a=$a, b=$b, d=$d, e=$e"))
        return new{T}(a, b, c, d, e, f, scale_z)
    end
end

function AffineTransform(; a, b, c, d, e, f, scale_z = 1.0)
    T = promote_type(map(typeof, (a, b, c, d, e, f, scale_z))...)
    return AffineTransform{T}(T(a), T(b), T(c), T(d), T(e), T(f), T(scale_z))
end

@inline (t::AffineTransform)(x, y, z) =
    (t.a * x + t.b * y + t.c, t.d * x + t.e * y + t.f, t.scale_z * z)

function inverse(t::AffineTransform{T}) where {T}
    det = t.a * t.e - t.b * t.d
    ia, ib = t.e / det, -t.b / det
    id, ie = -t.d / det, t.a / det
    return AffineTransform(a = ia, b = ib, c = -(ia * t.c + ib * t.f),
                           d = id, e = ie, f = -(id * t.c + ie * t.f),
                           scale_z = one(T) / t.scale_z)
end

"""
    TransformPair(forward, inverse)

The forward and inverse transforms between a grid CRS and an image CRS, held together.

`forward` maps grid coordinates to image coordinates; `inverse` maps back. Construct from a single
transform with [`transform_pair`](@ref), which derives the inverse, or pass both explicitly when
they come from a library that builds each separately.
"""
struct TransformPair{F,I}
    forward::F
    inverse::I
end

"""
    transform_pair(t) -> TransformPair

A [`TransformPair`](@ref) from `t` and its [`inverse`](@ref).
"""
transform_pair(t::AbstractCoordTransform) = TransformPair(t, inverse(t))
transform_pair(p::TransformPair) = p

"""
    isidentity(t) -> Bool

Whether `t` is the identity.

A predicate for inspecting a transform, not the mechanism behind the fast path: that is chosen by
dispatch on [`IdentityTransform`](@ref), so nothing in the kernel branches on this at runtime. A
`Proj.Transformation` between two identical CRSs is a no-op pipeline but reports `false`, since only
the type is examined.
"""
isidentity(::IdentityTransform) = true
isidentity(::Any) = false
isidentity(p::TransformPair) = isidentity(p.forward) && isidentity(p.inverse)

"""
    AbstractTransformFactory

Builds a fresh [`TransformPair`](@ref) each time it is called, as `factory()`.

A blocked run calls it once per task, so each task owns a transform for its lifetime. That is what
`Proj` needs: a `Proj.Transformation` wraps a `PJ*` on a context that PROJ documents as usable from
one thread at a time, and building two concurrently on the shared global context corrupts its
SQLite handle. A caller wrapping such a library supplies the factory, and
[`InterpolatedTransform`](@ref) is one that returns a lattice approximation.
"""
abstract type AbstractTransformFactory end

# The single entry point taking anything a caller may supply and returning a `TransformPair`. Lives
# here rather than beside its callers because it is the transform layer's own dispatch: both the
# driver and the blocked driver resolve through it, and so does anything built on them.
#
# A transform or a pair is taken as given; a factory is invoked once per task.
#
# No `Any` fallback: a `Proj.Transformation` is not a `Base.Callable`, so a fallback would silently
# invoke it as a zero-argument factory and fail somewhere unhelpful. Plain functions are accepted
# too, dispatched on `Function` rather than `Base.Callable` — the latter includes `Type`, and both
# `AbstractCoordTransform` and `TransformPair` are themselves callable, so a broader signature would
# be ambiguous against them.
_resolve_transform(t::AbstractCoordTransform) = transform_pair(t)
_resolve_transform(p::TransformPair) = p
_resolve_transform(factory::AbstractTransformFactory) = transform_pair(factory())
_resolve_transform(factory::Function) = transform_pair(factory())
_resolve_transform(x) = throw(ArgumentError(
    "transform must be an AbstractCoordTransform, a TransformPair, an AbstractTransformFactory, " *
    "or a zero-argument function returning one; got $(typeof(x)). Wrap a Proj.Transformation pair " *
    "in TransformPair(fwd, inv), or supply a zero-argument factory so each task builds its own."))
