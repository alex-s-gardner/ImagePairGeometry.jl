# Where the terrain is, for the solve that has to find it.
#
# `rdr2geo` inverts radar coordinates to a ground point by iterating on the target's height: a height
# gives a look angle, which places the target on the range sphere, which gives a location — and the
# terrain at that location gives the next height. The reference only ever supplies a *constant*, through
# `DEMInterpolator(zz)` at a scalar `zz` (`GeogridRadar.py:176`, `:289`), so the iteration converges in a
# handful of steps and the location it converges to is on a sphere rather than on terrain.
#
# A resampling field is indexed by output pixel, so it needs the inverse over real terrain: a reference
# range and azimuth, and the ground point actually imaged there. That means sampling a DEM per candidate
# location, which changes the iteration's character rather than its structure — see `AbstractHeightSource`.
#
# Two things this has to preserve, and both are why it is dispatch rather than a branch on a flag:
#
# The constant case must be *bitwise* what it was. The footprint bounds and the incidence angle both go
# through `rdr2geo` at a scalar height, and both are asserted against the reference; a `height_at` that
# returned `Float64(h)` through a different code path would be fine, but one that computed anything else
# would move every radar output.
#
# And a bare `Real` must keep working as `height`. Every existing caller passes one, so the sources below
# are an addition to that spelling rather than a replacement for it.

"""
    AbstractHeightSource

Terrain height as a function of position, for [`rdr2geo`](@ref) to iterate against.

A source is called as `height_at(src, lon, lat)` with **longitude and latitude in radians** — the units
`rdr2geo` has in hand mid-iteration, not the degrees a raster is indexed in. A source wrapping a
geographic raster converts; that conversion is the source's job precisely because the solve cannot know
what frame the data is in.

Implement [`height_at`](@ref) for a new source. [`ConstantHeight`](@ref) is the reference's behavior and
is what a bare `Real` becomes; the `Rasters` extension adds a raster-backed source, since sampling one
is IO.

# A varying height changes what convergence means

At a constant height the iteration is a contraction on a smooth surface and settles in a few steps —
which is why the reference can run a fixed count and ignore the result. Over real terrain the surface it
is chasing has slope, and where that slope approaches the look direction the fixed point can be
ill-conditioned or absent: that is layover, and there the question "which ground point does this pixel
image" has no single answer. So [`rdr2geo_converged`](@ref)'s flag stops being informational and becomes
the thing a caller has to read. `rdr2geo` itself still returns isce3's answer either way, as isce3 does.
"""
abstract type AbstractHeightSource end

"""
    height_at(src, lon, lat) -> Float64

Terrain height above the ellipsoid at `(lon, lat)`, in **radians**, in meters.

The one method a [`AbstractHeightSource`](@ref) must provide. A bare `Real` and a [`ConstantHeight`](@ref)
both answer with their own value and ignore the position, which is the reference's behavior; any other
callable is invoked as `src(lon, lat)`.
"""
function height_at end

# A bare `Real` is the spelling every existing caller uses, and this is exactly the `Float64(height)` the
# solve did before height sources existed — so the constant path is bitwise unchanged.
@inline height_at(h::Real, ::Real, ::Real) = Float64(h)

# Any other callable is taken at its word. Deliberately last in specificity, so `Real` and the declared
# sources win: without this a caller would have to wrap a closure in a type to use one.
@inline height_at(f, lon::Real, lat::Real) = Float64(f(lon, lat))

"""
    ConstantHeight(h)

A terrain height that is the same everywhere: the reference's behavior, spelled as a source.

`rdr2geo(...; height = 500.0)` and `rdr2geo(...; height = ConstantHeight(500.0))` are the same
computation to the bit. The type exists so a caller can be explicit about it, and so that a function
taking a source has something to default to.
"""
struct ConstantHeight{T<:Real} <: AbstractHeightSource
    height::T
end

@inline height_at(h::ConstantHeight, ::Real, ::Real) = Float64(h.height)

"""
    reference_height(src) -> Float64

The height the solve starts its iteration from.

For a constant source this is that constant, and the first snap then lands on the answer. For a
varying one it is a representative height — a DEM's mean, say — since the iteration has to begin
somewhere before it knows the location. A starting value far from the terrain costs iterations rather
than correctness, up to the point where the near-nadir guard in [`rdr2geo_converged`](@ref) rejects the
geometry outright.
"""
function reference_height end

@inline reference_height(h::Real) = Float64(h)
@inline reference_height(h::ConstantHeight) = Float64(h.height)

# A callable with no declared reference height starts from sea level, which is the honest default: it is
# where the ellipsoid is, and a source that wants better should be a type that says so.
@inline reference_height(::Any) = 0.0
