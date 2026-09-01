# Integer conversion, matching the reference's three distinct behaviors.
#
# The reference converts a Float64 to an integer in three different ways, and picking the wrong
# one is an off-by-one in the output raster. All three appear in the code paths this package
# reproduces, sometimes within a few lines of each other:
#
#   `std::round` then implicit `(GInt32)`   — C++, ties away from zero
#   `np.round` then `int()`                — Python, ties to even
#   implicit `(GInt32)` alone              — C++, truncation toward zero
#
# Julia's `round` is ties-to-even, so it matches `np.round` and *not* `std::round`. The two
# disagree on exactly the inputs that occur most often here: a value landing on `.5`. Ties are
# not rare — `(x_center - x_origin) / x_spacing` lands on an exact `.5` at every grid point when
# the grid spacing is an odd multiple of the image spacing (90 m grid over 30 m imagery, say).

"""
    cround(x::Float64) -> Float64

Round half away from zero, as C's `std::round` does: `cround(0.5) == 1.0`,
`cround(-0.5) == -1.0`.

Julia's `round` rounds half to even, so it returns `0.0` for both. Use this wherever the
reference calls `std::round`, and [`nround`](@ref) wherever it calls `np.round`.
"""
@inline cround(x::Float64) = round(x, RoundNearestTiesAway)

"""
    cround32(x::Float64) -> Int32

`std::round` followed by C's implicit conversion to `GInt32`, as in

    raster11[jj] = std::round(...);   // GInt32 = double

Because `cround` has already produced an integral value the subsequent truncation is the
identity, so this is a rounding conversion — unlike [`ctrunc32`](@ref).

Throws `InexactError` if the result is not representable as `Int32`. The reference instead
relies on undefined behavior, which clang implements on arm64 as saturation; failing loudly is
the deliberate divergence, recorded in `REFERENCE.md`.
"""
@inline cround32(x::Float64) = Int32(cround(x))

"""
    nround(x::Float64) -> Float64

Round half to even, as NumPy's `np.round` does. An alias for Julia's `round`, spelled out so
that call sites reproducing Python arithmetic say which reference they follow.

The reference's [`coregister`](@ref) uses `np.round`; its C++ kernel uses `std::round`. See
[`cround`](@ref).
"""
@inline nround(x::Float64) = round(x)

"""
    ctrunc32(x::Float64) -> Int32

C's implicit `double` to `GInt32` conversion with no rounding applied first: truncation toward
zero, so `ctrunc32(22.99) == 22`.

The reference uses this — not rounding — for chip size and the stable-surface mask
(`geogridOptical.cpp:874-875, 884-885, 893`), while applying `std::round` first for the offset
and search-range outputs (`:807-808, 849-863`). Substituting a rounding conversion here shifts
chip sizes by one pixel.

Throws `InexactError` outside the `Int32` range, as [`cround32`](@ref) does and for the same
reason.
"""
@inline ctrunc32(x::Float64) = Int32(trunc(x))
