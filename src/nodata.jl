# Nodata: one sentinel from the DEM, applied to rasters that do not share it.
#
# The reference reads a single nodata value from the DEM's first band
# (`geogridOptical.cpp:337-339`) and compares it against values from the velocity, search-range,
# chip-size and stable-surface rasters. Those rasters carry different sentinels in the ITS_LIVE
# parameter set — the DEM's is -32767, the search range's is +32767, chip size and the mask use 0 —
# so the single comparison catches some and misses others.
#
# That is reproduced rather than corrected, and this type exists to make it explicit which value is
# being compared where instead of leaving it implicit in a bare `Float64` threaded through the
# kernel. The consequences are recorded in `REFERENCE.md`; `searchrange_scale` and the chip-size
# path carry comments at the sites where the test fails to catch anything.

"""
    NoDataPolicy(; input, output = -32767.0)

How missing values are recognized on input and written on output.

# Fields
- `input`: the value treated as missing in *every* input raster. The reference takes this from the
  DEM's first band and applies it to all of them, so it is one value rather than one per raster.
  A raster whose own sentinel differs is not fully masked; see `REFERENCE.md`.
- `output`: written wherever an output cannot be computed. `-32767.0` in production
  (`testGeogrid.py:357`), stored as `Float64` and converted per output band.

The reference reads its input sentinel with `GetNoDataValue(NULL)`, discarding the success flag, so
a raster with no nodata set yields `0.0` and every genuine zero is then treated as missing.
Constructing this type with `input = 0.0` reproduces that; `nodata_from` makes the choice visible
at the call site.
"""
struct NoDataPolicy
    input::Float64
    output::Float64
end

NoDataPolicy(; input, output = -32767.0) = NoDataPolicy(Float64(input), Float64(output))

"""
    nodata_from(missingval; output = -32767.0) -> NoDataPolicy

A [`NoDataPolicy`](@ref) from a raster's declared missing value, falling back to `0.0` when there
is none.

The fallback reproduces the reference: it reads the DEM's nodata without checking whether one is
set, so an absent sentinel arrives as `0.0` and every zero-valued input pixel is read as missing.
Passing `nothing` here makes that consequence explicit rather than incidental.
"""
nodata_from(missingval::Real; output = -32767.0) =
    NoDataPolicy(; input = Float64(missingval), output)
nodata_from(::Nothing; output = -32767.0) = NoDataPolicy(; input = 0.0, output)

"""
    ismissingval(p::NoDataPolicy, x) -> Bool

Whether `x` is the missing-value sentinel.

Exact equality, as the reference tests it. `NaN` is therefore never equal to the sentinel and
propagates through the arithmetic instead of being masked.
"""
@inline ismissingval(p::NoDataPolicy, x::Real) = Float64(x) == p.input

"""
    NODATA_INT32

The output sentinel as it appears in the integer bands: `Int32(-32767)`.
"""
const NODATA_INT32 = Int32(-32767)
