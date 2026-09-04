# Approximating a coordinate transform on a coarse lattice.
#
# The projection library is around 88% of a cross-CRS run (`benchmark/cost_share.jl`), and the kernel
# makes three calls per grid point. Batching them through `proj_trans_generic` measures 1.00x — the
# cost is the projection math, not the call — so the only lever is making fewer calls. A transform
# between two map projections is smooth over a few hundred meters, so it can be evaluated on a
# coarse lattice of nodes and interpolated between them, the approach `gdalwarp` takes with
# `GDALApproxTransformer`.
#
# A `CoordLattice` is callable as `(x, y, z) -> (x′, y′, z′)`, so it substitutes into a
# `TransformPair` and the kernel needs no change. That is what makes the two modes possible:
# `location_x`/`location_y` come from the forward transform alone, so interpolating only the
# *inverse* leaves them bitwise exact while still dropping two of the three calls. The bands that go
# through the velocity operator carry the inverse's error and can tip by one; which band lands where
# is tabulated in `docs/interpolated-transform.md`, along with the measured cost of each mode.
#
# Not GDAL's approximating transformer, though it is the same idea: that one requires the points
# handed to it to be collinear and ordered, so it can transform the endpoints exactly and bisect
# between them. This kernel queries one point at a time, and its two inverse queries sit off the
# line the forward queries trace, so there is no line to hand it. A lattice also serves all three
# query types from one structure built once per task.

"""
    LatticeInterpolation

How a [`CoordLattice`](@ref) interpolates between its nodes.

Subtypes: [`NearestNode`](@ref), [`Bilinear`](@ref), [`Bicubic`](@ref). Chosen by dispatch rather
than a stored flag, so the per-point arithmetic specializes with no branch.
"""
abstract type LatticeInterpolation end

"""
    NearestNode()

Take the value at the nearest lattice node, with no interpolation.

Reproduces no transform beyond the nodes themselves, so it is not a candidate for production. It
exists to bound an error budget from below: comparing it against [`Bilinear`](@ref) at the same
spacing separates the error the lattice's coarseness carries from the error the interpolation adds.

Named for the node rather than as `Nearest` to leave that name free: `AutoRIFT.Nearest` is a distinct
resampling method for displacement fields, and a caller using both packages has both in scope.
"""
struct NearestNode <: LatticeInterpolation end

"""
    Bilinear()

Linear interpolation in each direction from the four nodes surrounding the point.

Exact for a transform that is affine over a lattice cell, which is what makes the error fall as the
square of the lattice spacing. What `gdalwarp` uses, and what the measurements in
`docs/interpolated-transform.md` were taken with.
"""
struct Bilinear <: LatticeInterpolation end

"""
    Bicubic()

Cubic convolution from the sixteen nodes surrounding the point, with the Catmull-Rom parameter.

Reproduces an affine transform exactly, as [`Bilinear`](@ref) does, and a quadratic one besides, so
it holds a given accuracy at a coarser lattice — fewer nodes to build, which is where the projection
calls are. The per-point arithmetic is roughly twice `Bilinear`'s, and it needs a two-node margin
around the queried region rather than one.
"""
struct Bicubic <: LatticeInterpolation end

# Nodes needed outside the queried region for the kernel's stencil to be defined at the boundary.
# Interpolating from nodes that do not exist would mean clamping at the edge, which biases the
# boundary; the lattice is built large enough that the stencil is always complete instead.
latticehalo(::NearestNode) = 1
latticehalo(::Bilinear) = 1
latticehalo(::Bicubic) = 2

"""
    CoordLattice

A coordinate transform tabulated on a regular lattice and interpolated between its nodes.

Callable as `(x, y, z) -> (x′, y′, z′)`, so it stands in for a transform anywhere one is taken —
including inside a [`TransformPair`](@ref), which is how [`InterpolatedTransform`](@ref) uses it.

# Fields
- `x`, `y`: the transformed coordinates at each node, `(nx, ny, nlevels)`. The trailing axis is the
  elevations in `zrange`: two where the transform's horizontal result depends on elevation, one where
  it does not.
- `origin`: the `(x, y)` input coordinate of node `(1, 1)`.
- `spacing`: node spacing in input coordinates, positive in both directions.
- `zrange`: the elevations bounding what the lattice was tabulated over.
- `interpolation`: the [`LatticeInterpolation`](@ref) used between nodes.
- `flat`: whether a query can read one level and ignore elevation.

Interpolating linearly between two elevations is exact rather than approximate: elevation enters a
horizontal coordinate only through a datum shift, which is linear in it. For two CRSs on one datum —
every ITS_LIVE projection — there is no such shift, so [`build_lattice`](@ref) tabulates one level
instead of two and a query reads it directly.

A query outside the lattice throws. Clamping instead would turn a bounds error into a plausible
coordinate, which is exactly the failure this package's bitwise tests exist to catch. A `NaN`
coordinate is not a bounds error and passes through as `NaN`, matching the exact path: the reference
reaches a `NaN` pixel index from a `NaN` elevation and reports it in bounds (see `REFERENCE.md`).
"""
struct CoordLattice{M<:LatticeInterpolation}
    x::Array{Float64,3}
    y::Array{Float64,3}
    origin::NTuple{2,Float64}
    spacing::NTuple{2,Float64}
    zrange::NTuple{2,Float64}
    interpolation::M
    flat::Bool

    function CoordLattice{M}(x, y, origin, spacing, zrange, interpolation) where {M}
        axes(x) == axes(y) || throw(DimensionMismatch(
            "CoordLattice x and y node tables must match: $(axes(x)) vs $(axes(y))"))
        size(x, 3) in (1, 2) || throw(ArgumentError(
            "CoordLattice node tables must have one or two elevation levels, got $(size(x, 3))"))
        all(>(0), spacing) || throw(ArgumentError(
            "CoordLattice spacing must be positive, got $spacing"))
        h = latticehalo(interpolation)
        min(size(x, 1), size(x, 2)) >= 2h + 2 || throw(ArgumentError(
            "CoordLattice is $(size(x, 1))x$(size(x, 2)) nodes, too small for " *
            "$(typeof(interpolation)), which needs at least $(2h + 2) in each direction"))
        # One level means the transform was found not to depend on elevation, so a query reads it
        # directly. With two, they still have to be interchangeable to skip one, tested on bit
        # patterns rather than `==` because `-0.0 == 0.0` holds while the bits differ.
        flat = size(x, 3) == 1 ||
               (reinterpret(UInt64, x[:, :, 1]) == reinterpret(UInt64, x[:, :, 2]) &&
                reinterpret(UInt64, y[:, :, 1]) == reinterpret(UInt64, y[:, :, 2]))
        return new{M}(x, y, origin, spacing, zrange, interpolation, flat)
    end
end

CoordLattice(x, y, origin, spacing, zrange, interpolation::M) where {M<:LatticeInterpolation} =
    CoordLattice{M}(x, y, origin, spacing, zrange, interpolation)

"""
    latticesize(L::CoordLattice) -> NTuple{2,Int}

Number of nodes in each direction.
"""
latticesize(L::CoordLattice) = (size(L.x, 1), size(L.x, 2))

"""
    build_lattice(t, bounds::Extent, spacing, interpolation;
                  zrange = DEFAULT_ZRANGE) -> CoordLattice

Tabulate the transform `t` on a lattice of the given spacing covering `bounds`.

`t` is called as `t(x, y, z) -> (x′, y′, z′)`. `bounds` is the region the lattice must be able to
answer queries in, as an `Extents.Extent{(:X, :Y)}` in `t`'s input coordinates; the lattice extends
a kernel-dependent margin beyond it so the interpolation stencil is complete at the boundary.
`spacing` is a positive `(dx, dy)` in input coordinates.

This is the only place the underlying transform is called. A non-finite node means the transform
failed over the region and is thrown rather than interpolated, since it would otherwise spread `NaN`
across every cell touching that node.

One elevation level is tabulated where the transform's horizontal result does not depend on
elevation, two where it does, so the cost is `nx * ny` calls rather than `2 * nx * ny` for a
same-datum pair. Which case holds is established by probing the two `zrange` extremes at the
lattice's own corners and center: elevation enters a horizontal coordinate only through a datum
shift, and a pipeline either applies one everywhere over a region this size or nowhere.
"""
function build_lattice(t, bounds::Extent, spacing::NTuple{2,Real},
                       interpolation::LatticeInterpolation; zrange = DEFAULT_ZRANGE)
    all(>(0), spacing) || throw(ArgumentError(
        "build_lattice spacing must be positive, got $spacing"))
    sx, sy = Float64(spacing[1]), Float64(spacing[2])
    xlo, xhi = Float64(bounds.X[1]), Float64(bounds.X[2])
    ylo, yhi = Float64(bounds.Y[1]), Float64(bounds.Y[2])
    xhi >= xlo && yhi >= ylo || throw(ArgumentError(
        "build_lattice bounds are empty: X = $(bounds.X), Y = $(bounds.Y)"))

    h = latticehalo(interpolation)
    # Cover the bounds, then extend by the stencil margin on both sides.
    nx = Int(ceil((xhi - xlo) / sx)) + 1 + 2h
    ny = Int(ceil((yhi - ylo) / sy)) + 1 + 2h
    origin = (xlo - h * sx, ylo - h * sy)

    z0, z1 = Float64(zrange[1]), Float64(zrange[2])
    xmax = origin[1] + (nx - 1) * sx
    ymax = origin[2] + (ny - 1) * sy
    xmid = origin[1] + (nx ÷ 2) * sx
    ymid = origin[2] + (ny ÷ 2) * sy
    nlevels = _depends_on_z(t, (origin[1], xmid, xmax), (origin[2], ymid, ymax), z0, z1) ? 2 : 1

    X = Array{Float64,3}(undef, nx, ny, nlevels)
    Y = Array{Float64,3}(undef, nx, ny, nlevels)
    for l in 1:nlevels
        z = l == 1 ? z0 : z1
        for j in axes(X, 2), i in axes(X, 1)
            gx = origin[1] + (i - 1) * sx
            gy = origin[2] + (j - 1) * sy
            px, py, _ = t(gx, gy, z)
            (isfinite(px) && isfinite(py)) || throw(ArgumentError(
                "build_lattice got a non-finite result from the transform at ($gx, $gy, $z): " *
                "($px, $py). The lattice must cover a region the transform is defined over, " *
                "including its $h-node margin beyond the queried bounds."))
            X[i, j, l] = px
            Y[i, j, l] = py
        end
    end
    return CoordLattice(X, Y, origin, (sx, sy), (z0, z1), interpolation)
end

# Whether `t`'s horizontal result moves with elevation anywhere among the probed points.
#
# Bitwise, not a tolerance: a difference of any size means the pipeline carries a vertical component,
# and then both levels have to be tabulated. Equal at every probe means it carries none, since a datum
# shift is a property of the pipeline rather than of position — it cannot switch on within a region a
# single lattice covers.
function _depends_on_z(t, xs, ys, z0::Float64, z1::Float64)
    z0 == z1 && return false
    for x in xs, y in ys
        a = t(x, y, z0)
        b = t(x, y, z1)
        ((a[1] === b[1]) & (a[2] === b[2])) || return true
    end
    return false
end

# The cell containing `v` along direction `d`, as the index of the cell's lower node and the fraction
# across it, having checked the interpolation stencil fits. `h` is the stencil's reach either side.
@inline function _cell(L::CoordLattice, v::Float64, d::Int, h::Int)
    # Divides rather than multiplying by a stored reciprocal: `x * (1/s)` is not `x / s` in the last
    # bit, and it measured no faster here — the divide is off the critical path.
    u = (v - L.origin[d]) / L.spacing[d]
    i = floor(Int, u) + 1
    n = size(L.x, d)
    (i - h >= 1) & (i + h <= n) || throw(ArgumentError(
        "CoordLattice query is outside the lattice: coordinate $d is $v, and the lattice covers " *
        "$(L.origin[d] + h * L.spacing[d]) to " *
        "$(L.origin[d] + (n - 1 - h) * L.spacing[d]). The lattice must be built over the whole " *
        "region that will be queried."))
    return (i, u - (i - 1))
end

@inline function (L::CoordLattice)(x, y, z)
    fx, fy, fz = Float64(x), Float64(y), Float64(z)
    # A `NaN` coordinate is a value the exact path propagates, not a bounds error.
    (isnan(fx) | isnan(fy)) && return (NaN, NaN, fz)

    px0, py0 = _interp(L, L.interpolation, fx, fy, 1)
    # Where the two levels are identical the second interpolation would return the same value and
    # then be blended with itself, so it is skipped: half the arithmetic for the common case, and the
    # result is the interpolated value rather than a weighted sum of two copies of it.
    L.flat && return (px0, py0, fz)

    px1, py1 = _interp(L, L.interpolation, fx, fy, 2)
    z0, z1 = L.zrange
    t = (fz - z0) / (z1 - z0)
    return (px0 + t * (px1 - px0), py0 + t * (py1 - py0), fz)
end

@inline function _interp(L::CoordLattice, m::NearestNode, x::Float64, y::Float64, l::Int)
    i, tx = _cell(L, x, 1, latticehalo(m))
    j, ty = _cell(L, y, 2, latticehalo(m))
    i += tx >= 0.5
    j += ty >= 0.5
    return (L.x[i, j, l], L.y[i, j, l])
end

@inline function _interp(L::CoordLattice, m::Bilinear, x::Float64, y::Float64, l::Int)
    i, tx = _cell(L, x, 1, latticehalo(m))
    j, ty = _cell(L, y, 2, latticehalo(m))
    return (_bilin(L.x, i, j, l, tx, ty), _bilin(L.y, i, j, l, tx, ty))
end

@inline function _bilin(A::Array{Float64,3}, i::Int, j::Int, l::Int, tx::Float64, ty::Float64)
    a = A[i, j, l] + tx * (A[i+1, j, l] - A[i, j, l])
    b = A[i, j+1, l] + tx * (A[i+1, j+1, l] - A[i, j+1, l])
    return a + ty * (b - a)
end

@inline function _interp(L::CoordLattice, m::Bicubic, x::Float64, y::Float64, l::Int)
    i, tx = _cell(L, x, 1, latticehalo(m))
    j, ty = _cell(L, y, 2, latticehalo(m))
    wx = _cubicweights(tx)
    wy = _cubicweights(ty)
    return (_bicub(L.x, i, j, l, wx, wy), _bicub(L.y, i, j, l, wx, wy))
end

# Catmull-Rom cubic convolution weights for the four nodes at offsets -1, 0, 1, 2. They sum to one
# and reproduce an affine function exactly, which is what makes the lattice exact for a transform
# that is affine over a cell.
@inline function _cubicweights(t::Float64)
    t2 = t * t
    t3 = t2 * t
    return (-0.5t3 + t2 - 0.5t,
            1.5t3 - 2.5t2 + 1.0,
            -1.5t3 + 2.0t2 + 0.5t,
            0.5t3 - 0.5t2)
end

@inline function _bicub(A::Array{Float64,3}, i::Int, j::Int, l::Int,
                        wx::NTuple{4,Float64}, wy::NTuple{4,Float64})
    s = 0.0
    for b in 1:4
        r = 0.0
        for a in 1:4
            r += wx[a] * A[i+a-2, j+b-2, l]
        end
        s += wy[b] * r
    end
    return s
end

"""
    InterpolatedTransform(transform, grid, pair; lattice = 4, mode = :hybrid,
                          interpolation = Bilinear(), window = nothing,
                          zrange = DEFAULT_ZRANGE)

A transform factory that approximates `transform` on a coarse lattice.

Pass it as `pairgeometry`'s or `pairgeometry_blocked`'s `transform`. It is an
[`AbstractTransformFactory`](@ref), so a blocked run calls it once per task and each task owns its
own lattice and its own PROJ context for that task's lifetime — the same ownership the exact PROJ
path needs, for the same reason.

`transform` is anything the exact path accepts: an [`AbstractCoordTransform`](@ref), a
[`TransformPair`](@ref), or a factory such as `ProjTransformFactory`. `grid` and `pair` give the
region the lattice must cover; `window` defaults to the whole grid intersected with the pair's
footprint, and must be the same window the result is computed over.

# Modes

- `:hybrid` — the forward transform stays exact and only the inverse is interpolated. Every integer
  band is bitwise identical to the exact path, because `location_x`/`location_y` come from the
  forward transform alone. Drops two of the three calls per point.
- `:full` — both directions are interpolated, so no call is made per point at all. Faster, but
  `location_x`/`location_y` can differ from the exact path by one pixel: the positional error is far
  below a pixel, and what shifts is which side of a rounding boundary a point falls on. Sound where
  a one-pixel difference in a search center is acceptable, which for a correlator searching tens of
  pixels it may be.

`lattice` is the node spacing as a multiple of the grid spacing; `1` puts a node at every grid point.
Coarser is faster to build and less accurate — `docs/interpolated-transform.md` tabulates the
measured error against spacing. `interpolation` selects the kernel: [`Bilinear`](@ref),
[`Bicubic`](@ref) or [`NearestNode`](@ref).

Both lattices are spaced by the same ground distance. The inverse lattice is tabulated over image
coordinates and extended by one image pixel beyond the footprint, because the kernel queries it at
the point stepped one pixel along each image axis.

```julia
tf = InterpolatedTransform(ProjTransformFactory(3413, 32624), grid, pair;
                           lattice = 4, mode = :hybrid, window = win)
r = pairgeometry_blocked(grid, pair, source; transform = tf, window = win, ntasks = 8)
```
"""
struct InterpolatedTransform{T,G,P,M<:LatticeInterpolation} <: AbstractTransformFactory
    transform::T
    grid::G
    pair::P
    lattice::Int
    mode::Symbol
    interpolation::M
    window::CartesianIndices{2}
    zrange::NTuple{2,Float64}
end

"""
    LATTICE_MODES

The modes [`InterpolatedTransform`](@ref) accepts: `(:hybrid, :full)`.
"""
const LATTICE_MODES = (:hybrid, :full)

function InterpolatedTransform(transform, grid::MapGrid, pair::CoregisteredPair;
                               lattice::Integer = 4, mode::Symbol = :hybrid,
                               interpolation::LatticeInterpolation = Bilinear(),
                               window = nothing, zrange = DEFAULT_ZRANGE)
    lattice >= 1 || throw(ArgumentError(
        "InterpolatedTransform lattice must be at least 1 grid spacing, got $lattice"))
    mode in LATTICE_MODES || throw(ArgumentError(
        "InterpolatedTransform mode must be one of $LATTICE_MODES, got :$mode"))
    zrange[2] >= zrange[1] || throw(ArgumentError(
        "InterpolatedTransform zrange must be ordered, got $zrange"))

    # The window is resolved here rather than per task so every task builds the same lattice over the
    # same region: a lattice sized from a block would make the result depend on the block size, which
    # `pairgeometry_blocked` guarantees it does not.
    win = window === nothing ?
        grid_window(grid, footprint_bounds(_resolve_transform(transform), pair.coordinate)) :
        window
    return InterpolatedTransform(transform, grid, pair, Int(lattice), mode, interpolation, win,
                                 (Float64(zrange[1]), Float64(zrange[2])))
end

# The bounds of `window`'s grid-point centers, which is what the forward transform is queried at.
function _grid_bounds(grid::MapGrid, window::CartesianIndices{2})
    xr, yr = window.indices
    xa, ya = gridpoint_center(grid, first(xr), first(yr))
    xb, yb = gridpoint_center(grid, last(xr), last(yr))
    return Extent(X = minmax(xa, xb), Y = minmax(ya, yb))
end

# The bounds the inverse transform is queried at.
#
# Not the image footprint. `pointgeometry` transforms a grid point and steps one image pixel from the
# result *before* `inbounds` rejects it, so every point of the window is queried — including the ones
# outside the image, which a window derived from a rotated footprint's bounding box always has. So
# the region is the forward image of the whole window, plus a pixel for the step.
#
# The window's perimeter is sampled rather than just its corners: a projection maps a straight edge
# to a curved one, so the corners alone can fall inside the true extent. Sampling is cheap here —
# once per lattice, against once per grid point in the kernel.
function _inverse_bounds(fwd, grid::MapGrid, window::CartesianIndices{2},
                         c::ProjectedCoordinate, zrange::NTuple{2,Float64},
                         margin::NTuple{2,Float64})
    xr, yr = window.indices
    xs = unique((first(xr):max(1, length(xr) ÷ 64):last(xr)..., last(xr)))
    ys = unique((first(yr):max(1, length(yr) ÷ 64):last(yr)..., last(yr)))

    xlo = ylo = Inf
    xhi = yhi = -Inf
    function note(i, j, z)
        gx, gy = gridpoint_center(grid, i, j)
        px, py, _ = fwd(gx, gy, z)
        (isfinite(px) && isfinite(py)) || return nothing
        xlo = min(xlo, px); xhi = max(xhi, px)
        ylo = min(ylo, py); yhi = max(yhi, py)
        return nothing
    end
    for z in zrange
        for i in xs
            note(i, first(yr), z)
            note(i, last(yr), z)
        end
        for j in ys
            note(first(xr), j, z)
            note(last(xr), j, z)
        end
    end
    isfinite(xlo) || throw(ArgumentError(
        "InterpolatedTransform could not bound the region the inverse transform is queried over: " *
        "the forward transform returned no finite value on the window's perimeter"))

    # The one-pixel step off each axis in either direction, plus a whole lattice cell of margin. The
    # perimeter is sampled rather than exhaustive, and under `:full` the forward that generated these
    # bounds is itself interpolated, so the true extremum can sit slightly outside what was sampled.
    # A cell of slack costs one node of build in each direction and removes the whole question.
    px = abs(Float64(c.spacing[1])) + margin[1]
    py = abs(Float64(c.spacing[2])) + margin[2]
    return Extent(X = (xlo - px, xhi + px), Y = (ylo - py, yhi + py))
end

function (f::InterpolatedTransform)()
    tf = _resolve_transform(f.transform)
    # One ground distance for both lattices, so `lattice` means the same thing in each direction.
    dx, dy = gridspacing(f.grid)
    spacing = (abs(dx) * f.lattice, abs(dy) * f.lattice)

    # The inverse lattice must cover wherever the *forward it is paired with* lands, so under `:full`
    # the bounds are taken from the forward lattice rather than the exact transform. The two differ by
    # the interpolation error, which is far below a pixel but still enough to put a perimeter point
    # outside a region bounded by the other one.
    if f.mode === :hybrid
        fwd = tf.forward
    else
        fwd = build_lattice(tf.forward, _grid_bounds(f.grid, f.window), spacing, f.interpolation;
                            zrange = f.zrange)
    end
    inv_bounds = _inverse_bounds(fwd, f.grid, f.window, f.pair.coordinate, f.zrange, spacing)
    return TransformPair(fwd, build_lattice(tf.inverse, inv_bounds, spacing, f.interpolation;
                                            zrange = f.zrange))
end
