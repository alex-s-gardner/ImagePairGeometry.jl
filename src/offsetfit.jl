# The offset field as a polynomial, and the order selection that fixes its degree.
#
# `LatticeOffsetField` is the general answer and needs no fit: it is a table, and a table reproduces
# anything. What a polynomial buys is that a handful of coefficients can be written into a product's
# metadata and re-applied later by something that has neither this package nor the orbits.
#
# The order is not fixed here, and that is the point. First order is what a near-repeat pair looks like it
# needs, and on the one measured pair it is *not* enough: the range ramp is visibly curved, and a straight
# line through the swath edges misses mid-swath by 0.15 samples. So the caller states an accuracy target
# and gets the lowest order meeting it, with the residual reported either way. A model named in advance
# would have shipped that 0.15 as a silent bias.
#
# The solve is written out rather than delegated. `LinearAlgebra` is not a dependency of this package and
# should not become one — it pulls BLAS into a core that is otherwise dependency-light and
# `juliac --trim`-compilable — and the systems here are tiny: an order-3 fit in three variables is 20
# terms, so the normal equations are 20x20. Gaussian elimination with partial pivoting on that is
# microseconds and a few kilobytes.

"""
    OFFSET_FIT_MAX_ORDER

Highest polynomial order [`fit_offset`](@ref) will try when selecting one: 4.

Not a limit on what the arithmetic can do — a bound on what is meaningful. A field that a quartic in
position cannot describe to a fraction of a pixel is not a smooth geometric offset with a bit of
curvature; it is either a pair whose two geometries differ enough that no shift relates them (see
[`height_sensitivity`](@ref)) or a lattice too coarse for the fit to see the field. Both are better
reported than fitted around, so the search stops here rather than climbing until something fits.
"""
const OFFSET_FIT_MAX_ORDER = 4

"""
    OFFSET_FIT_MIN_NODES_PER_TERM

Nodes required per polynomial term before a fit is attempted: 4.

An order-`n` fit in three variables has `binomial(n + 3, 3)` terms — 4, 10, 20, 35 for orders 1 to 4 — and
a least-squares solve is only constrained when the samples outnumber them. Equality gives interpolation
rather than a fit: the residual is zero by construction and says nothing about the field between the
nodes, which is exactly the quantity the residual is being read for. Four times over-determines it enough
that the reported residual reflects the field rather than the sampling.
"""
const OFFSET_FIT_MIN_NODES_PER_TERM = 4

"""
    OffsetFit

A polynomial describing an offset field, with the residual it achieves.

# Fields
- `order`: the total degree in `(x, y, z)`.
- `zorder`: the highest elevation exponent, capped by how many distinct heights were fitted. See
  [`offset_fit_terms`](@ref).
- `dsample`, `dline`: coefficients, one per term, in the order [`offset_fit_terms`](@ref) generates.
- `origin`, `scale`: the centring and scaling applied to `(x, y, z)` before evaluation. Fitting in raw
  projected coordinates would square and cube numbers of order 1e6, so the design matrix would lose most
  of its precision to the conditioning; these map the fitted region to roughly `[-1, 1]`.
- `residual`: the largest absolute discrepancy over the fitted nodes, as `(dsample, dline)`, in pixels.
- `nnodes`: how many nodes were fitted.

Callable as `f(x, y, z) -> (dsample, dline)`, so it substitutes for an offset field anywhere one is
evaluated pointwise.

Built by [`fit_offset`](@ref).
"""
struct OffsetFit
    order::Int
    zorder::Int
    dsample::Vector{Float64}
    dline::Vector{Float64}
    origin::NTuple{3,Float64}
    scale::NTuple{3,Float64}
    residual::NTuple{2,Float64}
    nnodes::Int
end

"""
    offset_fit_terms(order; zorder = order) -> Vector{NTuple{3,Int}}

Exponent triples of a polynomial of total degree `order` in three variables, in a fixed order.

`(i, j, k)` with `i + j + k <= order`, ascending by total degree and lexicographic within it, so the
constant is first and a fit of order `n` shares its leading terms with one of order `n + 1`. That makes
the coefficient vectors of two orders comparable and is why the enumeration is pinned rather than left to
whatever a loop produces.

`zorder` caps the elevation exponent independently, and [`fit_offset`](@ref) sets it from how many
distinct heights were sampled. Sampling `m` heights determines a polynomial of degree at most `m - 1` in
elevation: with two heights, `z^2` is a linear combination of `1` and `z` over the sampled points, so its
coefficient is not determined and including the term makes the normal equations singular. Capping is the
fix rather than regularising, because the undetermined coefficient carries no information — the two
sampled heights genuinely say nothing about curvature between them.
"""
function offset_fit_terms(order::Integer; zorder::Integer = order)
    order >= 0 || throw(ArgumentError("polynomial order must be non-negative, got $order"))
    zorder >= 0 || throw(ArgumentError("elevation order must be non-negative, got $zorder"))
    terms = NTuple{3,Int}[]
    for d in 0:order, i in d:-1:0, j in (d - i):-1:0
        k = d - i - j
        k <= zorder && push!(terms, (i, j, k))
    end
    return terms
end

@inline function _eval_terms!(row::Vector{Float64}, terms::Vector{NTuple{3,Int}},
                              x::Float64, y::Float64, z::Float64)
    for (t, (i, j, k)) in pairs(terms)
        row[t] = x^i * y^j * z^k
    end
    return row
end

"""
    fit_offset(field, grid, window; heights, target = 0.1, order = nothing,
               max_order = OFFSET_FIT_MAX_ORDER) -> OffsetFit

Fit a polynomial to an offset field, choosing the lowest order whose residual meets `target`.

`field` is anything callable as `field(x, y, z) -> (dsample, dline)` — an [`OffsetFit`](@ref), a
`CoordLattice`-backed field, or a closure over [`pixel_offset`](@ref). `grid` and `window` say where to
sample it, and `heights` gives the elevations to sample at: at least two, since a fit that never varies
`z` cannot determine its `z` coefficients and would report a residual that says nothing about elevation.

`target` is the residual to meet, in pixels, and applies to both components. Pass `order` to fix the
degree instead, in which case the residual is reported but not required to meet anything.

Throws if no order up to `max_order` meets `target`. That is the honest outcome: the field is not
described by a low-order polynomial, and [`LatticeOffsetField`](@ref) is the answer — it needs no fit and
no order.

# Example

```julia
fit = fit_offset(field, grid, window; heights = (0.0, 2000.0), target = 0.05)
fit.order        # 2 on the 24-day Sentinel-1 pair
fit.residual     # what it achieved
fit(x, y, z)     # evaluate anywhere
```
"""
function fit_offset(field, grid::MapGrid, window::CartesianIndices{2}; heights,
                    target::Real = 0.1, order::Union{Nothing,Integer} = nothing,
                    max_order::Integer = OFFSET_FIT_MAX_ORDER)
    zs = collect(Float64, heights)
    length(zs) >= 2 || throw(ArgumentError(
        "fit_offset needs at least two heights to determine the elevation terms, got $heights"))
    allunique(zs) || throw(ArgumentError(
        "fit_offset heights must be distinct, got $heights"))
    target > 0 || throw(ArgumentError("fit_offset target must be positive, got $target"))

    xs, ys, ds, ls = _sample_field(field, grid, window, zs)
    n = length(ds)

    # Centre and scale so the design matrix is conditioned. Raw projected coordinates cubed are 1e18,
    # which costs most of a Float64's precision before the solve begins.
    ox, sx = _center_scale(xs)
    oy, sy = _center_scale(ys)
    oz, sz = _center_scale(repeat(zs, inner = n ÷ length(zs)))

    # Sampling `m` heights determines at most degree `m - 1` in elevation; see `offset_fit_terms`.
    zorder = length(zs) - 1
    orders = order === nothing ? (1:Int(max_order)) : (Int(order):Int(order))
    best = nothing
    for o in orders
        terms = offset_fit_terms(o; zorder = min(o, zorder))
        nt = length(terms)
        if n < OFFSET_FIT_MIN_NODES_PER_TERM * nt
            order === nothing && break
            throw(ArgumentError(
                "fit_offset was asked for order $o, which has $nt terms, but only $n nodes were " *
                "sampled; at least $(OFFSET_FIT_MIN_NODES_PER_TERM * nt) are needed. A fit with as " *
                "many terms as samples interpolates rather than fits, and its residual would say " *
                "nothing about the field between the nodes. Widen the window or use a finer lattice."))
        end

        cs, rs = _solve_normal(terms, xs, ys, ds, (ox, oy, oz), (sx, sy, sz), zs, n)
        cl, rl = _solve_normal(terms, xs, ys, ls, (ox, oy, oz), (sx, sy, sz), zs, n)
        best = OffsetFit(o, min(o, zorder), cs, cl, (ox, oy, oz), (sx, sy, sz), (rs, rl), n)
        (rs <= target && rl <= target) && return best
    end

    if order !== nothing
        return best
    end
    r = best === nothing ? (NaN, NaN) : best.residual
    throw(ArgumentError(
        "no polynomial up to order $max_order describes this offset field to $target pixels; the " *
        "best reached $(r[1]) in sample and $(r[2]) in line. Use a LatticeOffsetField instead, which " *
        "needs no fit and no order, or check height_sensitivity — a pair whose two geometries differ " *
        "enough is not described by a pixel shift at any order."))
end

# The field sampled over the window at each height, flattened. One pass, so the field is evaluated once
# per node per height and nothing is held but the samples.
function _sample_field(field, grid::MapGrid, window::CartesianIndices{2}, zs::Vector{Float64})
    xr, yr = window.indices
    npos = length(xr) * length(yr)
    n = npos * length(zs)
    xs = Vector{Float64}(undef, n)
    ys = Vector{Float64}(undef, n)
    ds = Vector{Float64}(undef, n)
    ls = Vector{Float64}(undef, n)
    k = 0
    for z in zs, j in yr, i in xr
        gx, gy = gridpoint_center(grid, i, j)
        dsamp, dline = field(gx, gy, z)
        k += 1
        xs[k] = gx
        ys[k] = gy
        ds[k] = dsamp
        ls[k] = dline
    end
    return (xs, ys, ds, ls)
end

# Centre on the mean and scale to unit half-range, so the fitted variable spans about [-1, 1]. A constant
# input has no range to scale by and keeps a scale of one, which leaves its terms constant — correct, and
# it is how a single-height fit would degenerate if one were allowed.
function _center_scale(v::AbstractVector{Float64})
    lo, hi = extrema(v)
    o = 0.5 * (lo + hi)
    s = 0.5 * (hi - lo)
    return (o, s == 0 ? 1.0 : s)
end

# Least squares by normal equations, `(A'A) c = A'b`, solved by Gaussian elimination with partial
# pivoting. Normal equations square the condition number, which is why the inputs are centred and scaled
# first; with that done the systems here are well behaved at every order tried.
function _solve_normal(terms::Vector{NTuple{3,Int}}, xs::Vector{Float64}, ys::Vector{Float64},
                       b::Vector{Float64}, origin::NTuple{3,Float64}, scale::NTuple{3,Float64},
                       zs::Vector{Float64}, npos::Int)
    nt = length(terms)
    n = length(b)
    AtA = zeros(Float64, nt, nt)
    Atb = zeros(Float64, nt)
    row = Vector{Float64}(undef, nt)
    per = n ÷ length(zs)

    for k in 1:n
        z = zs[(k - 1) ÷ per + 1]
        _eval_terms!(row, terms, (xs[k] - origin[1]) / scale[1], (ys[k] - origin[2]) / scale[2],
                     (z - origin[3]) / scale[3])
        for a in 1:nt
            ra = row[a]
            Atb[a] += ra * b[k]
            for c in a:nt
                AtA[a, c] += ra * row[c]
            end
        end
    end
    # Only the upper triangle was accumulated; the matrix is symmetric by construction.
    for a in 1:nt, c in 1:(a - 1)
        AtA[a, c] = AtA[c, a]
    end

    coef = _gauss_solve!(AtA, Atb)

    # The residual over the fitted nodes, which is what the caller reads to decide whether the order is
    # enough. Maximum rather than RMS: a fit that is excellent on average and half a pixel out at
    # mid-swath is exactly the failure the order selection exists to catch, and an RMS would hide it.
    worst = 0.0
    for k in 1:n
        z = zs[(k - 1) ÷ per + 1]
        _eval_terms!(row, terms, (xs[k] - origin[1]) / scale[1], (ys[k] - origin[2]) / scale[2],
                     (z - origin[3]) / scale[3])
        pred = 0.0
        for a in 1:nt
            pred += coef[a] * row[a]
        end
        worst = max(worst, abs(pred - b[k]))
    end
    return (coef, worst)
end

# Gaussian elimination with partial pivoting, in place. Small and self-contained so the package needs no
# `LinearAlgebra` dependency; see the note at the top of this file.
function _gauss_solve!(A::Matrix{Float64}, b::Vector{Float64})
    n = length(b)
    for k in 1:n
        # Partial pivoting: without it a zero on the diagonal — which a term absent from the sampled
        # region produces — divides by zero and returns NaN coefficients silently.
        p, best = k, abs(A[k, k])
        for r in (k + 1):n
            v = abs(A[r, k])
            if v > best
                p = r
                best = v
            end
        end
        best == 0 && throw(ArgumentError(
            "the offset fit's normal equations are singular at term $k. The sampled region most " *
            "likely does not vary in one of the fitted directions, so its coefficients are not " *
            "determined."))
        if p != k
            for c in k:n
                A[k, c], A[p, c] = A[p, c], A[k, c]
            end
            b[k], b[p] = b[p], b[k]
        end
        piv = A[k, k]
        for r in (k + 1):n
            f = A[r, k] / piv
            f == 0 && continue
            for c in k:n
                A[r, c] -= f * A[k, c]
            end
            b[r] -= f * b[k]
        end
    end
    # Back substitution.
    x = Vector{Float64}(undef, n)
    for k in n:-1:1
        s = b[k]
        for c in (k + 1):n
            s -= A[k, c] * x[c]
        end
        x[k] = s / A[k, k]
    end
    return x
end

@inline function (f::OffsetFit)(x::Real, y::Real, z::Real)
    terms = offset_fit_terms(f.order; zorder = f.zorder)
    xn = (Float64(x) - f.origin[1]) / f.scale[1]
    yn = (Float64(y) - f.origin[2]) / f.scale[2]
    zn = (Float64(z) - f.origin[3]) / f.scale[3]
    ds = dl = 0.0
    for (t, (i, j, k)) in pairs(terms)
        v = xn^i * yn^j * zn^k
        ds += f.dsample[t] * v
        dl += f.dline[t] * v
    end
    return (ds, dl)
end

function Base.show(io::IO, ::MIME"text/plain", f::OffsetFit)
    println(io, "OffsetFit: order $(f.order), $(length(f.dsample)) terms, $(f.nnodes) nodes")
    print(io, "  residual: ", f.residual[1], " sample, ", f.residual[2], " line")
end
