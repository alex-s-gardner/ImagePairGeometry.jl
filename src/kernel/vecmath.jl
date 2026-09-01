# Three-vector primitives, evaluated in the reference's operation order.
#
# These duplicate LinearAlgebra's `dot`, `cross`, `norm` and `normalize`, and the duplication is
# deliberate: floating-point addition is not associative, so a sum evaluated in a different order
# gives a different last bit. `dot_C` (`geogridOptical.cpp:1008-1012`) sums strictly left to
# right, whereas a generic `dot` may pairwise-reduce or vectorize. Reproducing the reference to
# the last bit means fixing the order here rather than hoping a library agrees.
#
# `unitvec3` divides each component by the norm (`:1026-1032`); `LinearAlgebra.normalize` scales
# by the reciprocal instead, and `x * (1/n)` is not `x / n` in floating point.
#
# All four take `SVector{3,Float64}`: stack-allocated, so the kernel does not allocate, and
# unrolled, so there is no loop overhead to trade against the fixed evaluation order.

"""
    dot3(v, w) -> Float64

Dot product summed left to right, matching `dot_C` (`geogridOptical.cpp:1008-1012`).

Not `LinearAlgebra.dot`, whose reduction order is unspecified and which may contract to an `fma`.
"""
@inline dot3(v::SVector{3,Float64}, w::SVector{3,Float64}) =
    v[1] * w[1] + v[2] * w[2] + v[3] * w[3]

"""
    cross3(u, v) -> SVector{3,Float64}

Cross product, matching `cross_C` (`geogridOptical.cpp:1014-1018`).
"""
@inline cross3(u::SVector{3,Float64}, v::SVector{3,Float64}) = SVector{3,Float64}(
    u[2] * v[3] - u[3] * v[2],
    u[3] * v[1] - u[1] * v[3],
    u[1] * v[2] - u[2] * v[1],
)

"""
    norm3(v) -> Float64

Euclidean norm as `sqrt` of the left-to-right sum of squares, matching `norm_C`
(`geogridOptical.cpp:1020-1024`).

Unlike `LinearAlgebra.norm` this does no scaling to avoid overflow. The reference does none
either, and the inputs here are map coordinates and unit-scaled differences, orders of magnitude
away from overflowing.
"""
@inline norm3(v::SVector{3,Float64}) = sqrt(v[1] * v[1] + v[2] * v[2] + v[3] * v[3])

"""
    unitvec3(v) -> SVector{3,Float64}

`v` scaled to unit length by dividing each component by [`norm3`](@ref), matching `unitvec_C`
(`geogridOptical.cpp:1026-1032`).

Not `LinearAlgebra.normalize`, which multiplies by the reciprocal norm — a different result in
the last bit.

A zero vector yields `NaN` components rather than an error, as the reference does. The zero case
is reachable: the surface normal is set to all-zeros when no slope raster is supplied
(`geogridOptical.cpp:761-764`).
"""
@inline function unitvec3(v::SVector{3,Float64})
    n = norm3(v)
    return SVector{3,Float64}(v[1] / n, v[2] / n, v[3] / n)
end
