# Native-Julia coordinate transforms, for the CRS pairs FastGeoProjections implements.
#
# This is the package's only transform implementation: nothing in `src/` links a projection library.
# The kernel makes three transform calls per grid point, and on the projected path the transform is
# most of what a cross-CRS run costs — measured on the kernel's own call pattern, one forward and two
# inverse per point from EPSG:3413 to EPSG:32624, a native transformation is 178 ns/point against
# PROJ's 445. On the radar path the two solves dominate instead and the two are within a few percent.
#
# The two do not agree bitwise: 2.4e-8 m separates them across the Greenland domain at three heights,
# which is 2e-14 relative against coordinates of ~1e6 m. That sits inside the ≤1e-7 relative bound
# `REFERENCE.md` sets for the Float64 output bands, and it moves no rounded pixel index — 0 of 320,000
# checked at 10 m and at 2.3 m spacing — so every integer band stays bitwise. PROJ remains what the
# reference fixtures were generated with, and `test/` loads Proj.jl to assert against them; the
# package itself does not depend on it.
#
# Nothing here needs a per-task anything. A `FastGeoProjections.Transformation` over a native pair is
# immutable and holds no library state, so one object serves every thread — which is why this offers a
# `TransformPair` directly where a PROJ-backed transform would have to offer a factory.
#
# The axis-order default is `always_xy = true` rather than the reference's authority order. For the
# projected pairs the fixtures use the two are identical, so nothing asserted against the reference
# depends on the choice; where it matters is a geographic CRS, and there authority order would hand the
# radar path a reversed pair. See `fast_transform`.

"""
    FastTransform(source, target; always_xy = true)

A coordinate transform between two EPSG codes, evaluated by FastGeoProjections.

Callable as `(x, y, z) -> (x′, y′, z′)`, the interface [`AbstractCoordTransform`](@ref) defines. The
third coordinate is carried through unchanged: both CRSs of a projected pair share a datum, so a
height is not a quantity either projection transforms.

`always_xy = true` — easting/longitude first — is the default. See [`fast_transform`](@ref) on why,
and on the one case where the choice is observable.
"""
struct FastTransform{F} <: AbstractCoordTransform
    f::F
    source::Int
    target::Int
    always_xy::Bool
end

function FastTransform(source::Integer, target::Integer; always_xy::Bool = true)
    f = FGP.Transformation(FGP.EPSG(Int(source)), FGP.EPSG(Int(target)); always_xy)
    return FastTransform(f, Int(source), Int(target), always_xy)
end

@inline (t::FastTransform)(x, y, z) = (t.f(x, y)..., z)

inverse(t::FastTransform) = FastTransform(t.target, t.source; always_xy = t.always_xy)

Base.show(io::IO, t::FastTransform) =
    print(io, "FastTransform(EPSG:", t.source, " => EPSG:", t.target,
          "; always_xy = ", t.always_xy, ")")

"""
    fast_transform(grid_crs, image_crs; always_xy = true) -> TransformPair

A [`TransformPair`](@ref) between two EPSG codes, evaluated by FastGeoProjections.

`grid_crs` and `image_crs` are EPSG codes — integers, or `FastGeoProjections.EPSG` objects. Pass the
result as `pairgeometry`'s or `pairgeometry_blocked`'s `transform`; it needs no factory even for a
threaded run, because the transformation holds no per-thread state.

Equal codes give an [`IdentityTransform`](@ref) rather than a projection composed with its own
inverse. PROJ resolves that case to a `+proj=noop` pipeline and returns its input unchanged, and the
same-CRS fixtures asserted against it are bitwise on every band; composing unproject with project
would instead land tens of nanometres away and break them.

# Axis order

`always_xy = true` is the default: a coordinate pair is `(easting, northing)` or `(longitude,
latitude)`, whatever the CRS's authority order says.

For a projected-to-projected pair the flag is inert — every EPSG code this package is used with orders
a projected pair easting-first anyway, so the fixtures are unaffected by the choice. Where it decides
anything is a geographic CRS: EPSG:4326's authority order is `(latitude, longitude)`, so
`always_xy = false` returns that pair reversed relative to what the radar path reads.

That is why the default is `true` rather than matching the reference's
`osr.CoordinateTransformation` on SRSs from `ImportFromEPSG`, which is authority order. The radar path
transforms grid coordinates to geodetic degrees and reads the result as `(lon, lat, h)`; an
authority-order transform swaps the two and yields geometry that is wrong everywhere without raising —
the solve converges against a target on the wrong side of the planet. Defaulting to the order the
kernel actually wants removes a silent failure whose only symptom is bad output.

Pass `always_xy = false` to reproduce authority order deliberately.

```julia
tf = fast_transform(3413, 32624)
r = pairgeometry_blocked(grid, pair, source; transform = tf, ntasks = 8)

# A radar pair: grid CRS to geodetic degrees, needing no flag now that xy-order is the default.
tf = fast_transform(32632, 4326)
```
"""
function fast_transform(grid_crs, image_crs; always_xy::Bool = true)
    g, i = _epsg_code(grid_crs), _epsg_code(image_crs)
    g == i && return transform_pair(IdentityTransform())
    return TransformPair(FastTransform(g, i; always_xy),
                         FastTransform(i, g; always_xy))
end

_epsg_code(x::Integer) = Int(x)
_epsg_code(x::FGP.EPSG) = Int(first(x.val))
_epsg_code(x) = throw(ArgumentError(
    "fast_transform takes EPSG codes as integers or EPSG objects, got $(typeof(x)). " *
    "FastGeoProjections looks a transformation up by code rather than parsing a CRS " *
    "description, so a WKT or PROJ string has no code to resolve. Build a TransformPair from a " *
    "library that parses one -- Proj.jl among them -- and pass that instead."))
