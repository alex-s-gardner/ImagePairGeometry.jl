# Where a ground point falls in the *secondary* image, relative to where it falls in the reference.
#
# The rest of this package answers "where does this grid point fall in the reference image", and hands
# that one index to a correlator for both images of the pair. This file answers the question that makes
# doing so valid: by how much do the two disagree?
#
# One interface, one method per coordinate system, on the same principle as `pointgeometry` — which has
# a `ProjectedCoordinate` method that is three transform calls and a `RadarCoordinate` method that is
# three solves, both producing the `PointGeometry` that `kernel/outputs.jl` consumes without knowing
# which it got. Everything built on `pixel_offset` is written against the interface, so a coordinate
# system added later supplies one method and inherits the rest.
#
# The offset is expressed in the *reference's* pixel axes, which is why the two coordinates must be the
# same kind: a pixel shift describes the relationship between two images only when both are indexed the
# same way. `CoregisteredPair` enforces that when the pair is built.

"""
    OffsetField(pair, grid, window; dem, transform = IdentityTransform())

The [`pixel_offset`](@ref) of every point of `window`, evaluated on indexing.

An `AbstractMatrix{NTuple{2,Float64}}` sharing `window`'s shape, so it lines up with a
[`PairGeometry`](@ref) over the same window and can be read a block at a time. Nothing is stored: each
element is two solves when it is asked for, and a windowed read costs that window.

`dem` supplies the elevation — an array over `window`, or a scalar for a constant height. `grid` and
`transform` are as [`pairgeometry`](@ref) takes them.

Indexed by position within the window rather than by grid index, as [`GeometryInputs`](@ref)'s arrays
are, so a block of a larger field is a `view`.

# Cost

Two zero-Doppler solves per element on the radar path, about 10 µs. A Sentinel-1 subswath is 33 million
samples, so evaluating one everywhere is minutes and gigabytes — which is what
[`LatticeOffsetField`](@ref) exists to avoid. This exact form is what that one is measured against, and
what to use where the window is small.
"""
struct OffsetField{P<:CoregisteredPair,G<:MapGrid,D,T} <: AbstractMatrix{NTuple{2,Float64}}
    pair::P
    grid::G
    window::CartesianIndices{2}
    dem::D
    transform::T

    function OffsetField{P,G,D,T}(pair, grid, window, dem, transform) where {P,G,D,T}
        if dem isa AbstractArray
            size(dem) == size(window) || throw(DimensionMismatch(
                "OffsetField dem is $(size(dem)) but the window is $(size(window)); the elevation " *
                "must cover exactly the window being evaluated"))
        end
        # Fail here rather than at the first indexed element: a field over a pair that cannot be
        # answered is a mistake in the call that built it, not in the read.
        pair.secondary_offset_coordinate === nothing && throw(ArgumentError(
            "OffsetField needs a pair carrying a secondary coordinate; this one has none, so there " *
            "is no offset to evaluate. See CoregisteredPair."))
        return new{P,G,D,T}(pair, grid, window, dem, transform)
    end
end

OffsetField(pair::CoregisteredPair, grid::MapGrid, window::CartesianIndices{2}; dem,
            transform = IdentityTransform()) =
    OffsetField{typeof(pair),typeof(grid),typeof(dem),typeof(transform)}(
        pair, grid, window, dem, transform)

Base.size(f::OffsetField) = size(f.window)
Base.axes(f::OffsetField) = axes(f.window)
Base.IndexStyle(::Type{<:OffsetField}) = IndexCartesian()

@inline _elevation(dem::AbstractArray, i::Int, j::Int) = Float64(dem[i, j])
@inline _elevation(z::Real, ::Int, ::Int) = Float64(z)

Base.@propagate_inbounds function Base.getindex(f::OffsetField, i::Int, j::Int)
    @boundscheck checkbounds(f, i, j)
    # `window` gives grid indices; `i, j` index within it, so the grid point is offset by its origin.
    gi, gj = f.window.indices[1][i], f.window.indices[2][j]
    gx, gy = gridpoint_center(f.grid, gi, gj)
    return pixel_offset(f.pair, gx, gy, _elevation(f.dem, i, j); transform = f.transform)
end

"""
    LatticeOffsetField(pair, grid, window; dem, transform = IdentityTransform(),
                       lattice = 32, interpolation = Bilinear(), zrange = DEFAULT_ZRANGE)

[`OffsetField`](@ref) tabulated on a coarse lattice and interpolated between its nodes.

Same interface — an `AbstractMatrix{NTuple{2,Float64}}` over `window` — and the same result to the
accuracy the lattice spacing buys. `lattice` is the node spacing as a multiple of the grid spacing, so
`32` puts a node every 32 grid points and costs `1/1024` of the solves.

This is what makes the field affordable. The offset varies smoothly with position — it is set by the two
orbits' geometry, which changes over kilometres, not pixels — so a coarse lattice reproduces it far
below the precision a correlator works to. `docs/src/coregistration.md` tabulates the measured error
against spacing.

Built on [`CoordLattice`](@ref), which already tabulates a callable of `(x, y, z)` with the stencil
halo and the two elevation levels.

# `zrange` is the knob that matters, not just `lattice`

One caveat that does not apply to `CoordLattice`'s use for coordinate transforms: there, interpolating
linearly in elevation is *exact*, because elevation enters a horizontal coordinate only through a datum
shift. Here it is an approximation, and over a wide `zrange` it is the error that dominates rather than
the node spacing.

Measured over a 128×128 grid at 200 m spacing on the 24-day Sentinel-1 pair, at a constant 500 m
elevation, as the maximum absolute error in samples against the exact field:

| `lattice` | default `zrange`, −200 to 4000 m | `zrange` = 400 to 600 m |
|---|---|---|
| 4 | 6.4e-5 | 3.6e-6 |
| 8 | 6.4e-5 | 1.5e-5 |
| 16 | 6.4e-5 | 6.2e-5 |
| 32 | 1.9e-4 | 2.5e-4 |
| 64 | 9.3e-4 | 9.9e-4 |

With the default range the error floors at 6.4e-5 samples for any spacing of 16 or finer: that floor is
the elevation interpolation across 4200 m, not the lattice. Narrowing `zrange` to the relief actually
present removes it, and the spacing then scales as the square, four times per doubling, as bilinear
interpolation should.

Both are far below what a correlator resolves — 9.3e-4 samples at the coarsest setting here is a
nanometre-scale ground displacement — so the practical guidance is that `lattice = 32` is ample and the
default `zrange` costs nothing that matters. The table is here because the *shape* of the error is worth
knowing: a caller who tightens the lattice and sees no improvement is looking at the elevation term, and
[`height_sensitivity`](@ref) is what says whether that term is large enough to care about at all.
"""
struct LatticeOffsetField{L,D} <: AbstractMatrix{NTuple{2,Float64}}
    lattice::L
    grid::MapGrid
    window::CartesianIndices{2}
    dem::D
end

function LatticeOffsetField(pair::CoregisteredPair, grid::MapGrid, window::CartesianIndices{2};
                            dem, transform = IdentityTransform(), lattice::Integer = 32,
                            interpolation::LatticeInterpolation = Bilinear(),
                            zrange = DEFAULT_ZRANGE)
    lattice >= 1 || throw(ArgumentError(
        "LatticeOffsetField lattice must be at least 1 grid spacing, got $lattice"))
    pair.secondary_offset_coordinate === nothing && throw(ArgumentError(
        "LatticeOffsetField needs a pair carrying a secondary coordinate; this one has none. See " *
        "CoregisteredPair."))

    # The offset as a `CoordLattice`-shaped callable: it tabulates two outputs of `(x, y, z)` and
    # carries the third through, which is exactly the shape of an offset plus its elevation.
    f = (x, y, z) -> (pixel_offset(pair, x, y, z; transform)..., z)

    dx, dy = gridspacing(grid)
    spacing = (abs(dx) * lattice, abs(dy) * lattice)

    # A cell of slack beyond the window's own extent. `_grid_bounds` returns the bounds of the queried
    # grid-point centres exactly, and `build_lattice` extends by whole nodes from there, so a corner
    # point can land precisely on the last node the stencil reaches — where `CoordLattice` refuses the
    # query rather than clamping, which is the behavior that keeps a bounds error from becoming a
    # plausible coordinate. One extra cell in each direction costs one node per side and removes the
    # question; `_inverse_bounds` gives itself the same slack for the same reason.
    b = _grid_bounds(grid, window)
    bounds = Extent(X = (b.X[1] - spacing[1], b.X[2] + spacing[1]),
                    Y = (b.Y[1] - spacing[2], b.Y[2] + spacing[2]))
    L = build_lattice(f, bounds, spacing, interpolation; zrange)
    return LatticeOffsetField{typeof(L),typeof(dem)}(L, grid, window, dem)
end

Base.size(f::LatticeOffsetField) = size(f.window)
Base.axes(f::LatticeOffsetField) = axes(f.window)
Base.IndexStyle(::Type{<:LatticeOffsetField}) = IndexCartesian()

Base.@propagate_inbounds function Base.getindex(f::LatticeOffsetField, i::Int, j::Int)
    @boundscheck checkbounds(f, i, j)
    gi, gj = f.window.indices[1][i], f.window.indices[2][j]
    gx, gy = gridpoint_center(f.grid, gi, gj)
    ds, dl, _ = f.lattice(gx, gy, _elevation(f.dem, i, j))
    return (ds, dl)
end

# What each path requires of a secondary coordinate, beyond its being the same kind as the reference.
# `pair.jl` refuses a mismatched kind; these accept a matching one, and each states its own precondition
# so that a pair which `pixel_offset` cannot answer is refused when it is built rather than when the
# offset is first read. They live here, beside the methods they guard, rather than with the struct.
_check_secondary(::RadarCoordinate, ::RadarCoordinate) = nothing

function _check_secondary(c::ProjectedCoordinate, s::ProjectedCoordinate)
    # The same requirement `coregister` places on two footprints, for the same reason: an offset in
    # pixels presumes one pixel size.
    c.spacing == s.spacing || throw(ArgumentError(
        "a projected pair's two coordinates must share a pixel spacing, but the reference has " *
        "$(c.spacing) and the secondary $(s.spacing). Reproject one image before pairing."))
    return nothing
end

"""
    pixel_offset(pair::CoregisteredPair, x, y, z; transform = IdentityTransform())
        -> NTuple{2,Float64}

Where the ground point `(x, y, z)` falls in the secondary image, minus where it falls in the reference,
in the reference's pixel axes.

`x`, `y` are in grid coordinates and `z` is the elevation, as the geometry kernel takes them. The
result is `(dsample, dline)` for a radar pair and `(dcolumn, drow)` for a projected one — fractional,
not rounded, since a resampler needs the fractional part and the rounded pixel indices
[`pairgeometry`](@ref) reports have already discarded it.

`transform` maps grid coordinates to the image's, exactly as [`pairgeometry`](@ref) takes it, and is
used on the radar path only — there it supplies the grid-to-geodetic direction the solves need. The
projected path ignores it, since its answer does not depend on position at all.

Requires `pair.secondary_offset_coordinate`. A pair built from a single coordinate does not have one
and this throws saying so; see [`CoregisteredPair`](@ref).

# Sign

Positive means the point sits at a *higher* sample or line in the secondary than in the reference. Add
this to a reference pixel index to reach the corresponding secondary pixel; subtract it from a
displacement a correlator measured between the two to remove the geometry's contribution.
"""
function pixel_offset(pair::CoregisteredPair, x::Real, y::Real, z::Real;
                      transform = IdentityTransform())
    return _pixel_offset(pair.coordinate, pair.secondary_offset_coordinate,
                         Float64(x), Float64(y), Float64(z), transform)
end

# No secondary geometry. Named rather than a `MethodError` because this is the state every pair built
# from one coordinate is in, and the fix is to supply the argument rather than to change the call.
@noinline _pixel_offset(::AbstractImageCoordinate, ::Nothing, ::Float64, ::Float64, ::Float64, _) =
    throw(ArgumentError(
        "this CoregisteredPair carries no secondary coordinate, so the offset between the two " *
        "images cannot be computed. Build the pair with `secondary = <the secondary's " *
        "coordinate>`, or from two acquisitions where a reader supplies both."))

# Two projected images, both terrain-corrected onto the same map grid: the grid point is at the same
# place in each, so the offset is zero. That is the answer rather than a placeholder.
#
# An orthorectified product has already had the terrain displacement `dh * tan(theta)` removed using the
# processor's DEM and view model. What is left is the residual of that correction — the error in the DEM
# it used and in its own view model, neither of which the product reports and neither of which is a
# function of anything a `ProjectedCoordinate` holds. So there is nothing here to derive: a correction
# proportional to an unknown elevation error would be a free parameter, not geometry. Where the
# producer's accuracy is not enough, the remedy is a shift measured over stable ground, which is what a
# correlator produces.
#
# `docs/src/coregistration.md` records the measurement behind this: across both cross-path Landsat pairs
# in the golden set and a same-path control, `view:off_nadir` is 0 for every scene — Landsat 7 and 8 do
# not steer cross-track — so even an adjacent-path pair views its overlap at nearly one geometry and
# there is no view separation for a parallax term to act on.
@inline _pixel_offset(::ProjectedCoordinate, ::ProjectedCoordinate,
                      ::Float64, ::Float64, ::Float64, _) = (0.0, 0.0)

# The radar method: a zero-Doppler solve against each acquisition's own orbit. `radar/rdr2rdr.jl` holds
# the solving; this converts the grid point it needs, which is the same two steps the radar
# `pointgeometry` takes (`radar/geometry.jl`) and for the same reason — the transform produces degrees and
# the solves want ECEF meters.
@inline function _pixel_offset(ref::RadarCoordinate, sec::RadarCoordinate,
                               x::Float64, y::Float64, z::Float64, transform)
    tf = _resolve_transform(transform)
    lon_d, lat_d, h = tf.forward(x, y, z)
    xyz = lonlat_to_xyz(Ellipsoid(), SVector{3,Float64}(lon_d * DEG2RAD, lat_d * DEG2RAD, h))
    return _radar_offset(ref, sec, xyz)
end
