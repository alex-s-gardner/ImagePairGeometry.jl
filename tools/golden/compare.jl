# Comparing one band against the reference's own GeoTIFF, and the gates each band is held to.
#
# The two tiers are `REFERENCE.md`'s, and they fall where they do because the outputs differ in what
# is achievable rather than in what is wanted:
#
#   Tier A — the eleven `Int32` bands. Bitwise, in every case. Each passes through a rounding or
#            truncating conversion that absorbs any last-bit difference in its inputs, so exact
#            agreement is achievable and anything less is a real difference in the kernel.
#   Tier B — the eight `Float64` bands. Bitwise where the grid and the image share a CRS, and bounded
#            in relative error where they do not, because two things make a reprojected float
#            unreproducible bit for bit: the reference is compiled with floating-point contraction, so
#            it may evaluate `a*b + c` as one `fma` where Julia rounds twice; and PROJ makes no
#            bit-reproducibility promise across platforms and does not deliver one.
#
# **Coverage is compared as a value, not skipped.** A point one side computed and the other left at the
# sentinel is a disagreement, and usually a more informative one than a value difference: it points at
# the footprint test or the nodata policy rather than at arithmetic. So the sentinel takes part in the
# comparison like any other number, and `nvalid` is asserted against the reference's own count of
# non-sentinel points.
#
# The relative measure divides by `max(|a|, |b|, 1)` rather than by `|b|`. The float bands span many
# orders of magnitude and pass through zero, so dividing by the value alone would report an unbounded
# relative error on a band whose absolute error is a rounding artifact.

using Printf
using Statistics

"""
    BandResult

One band compared: how it was judged, whether it passed, and the numbers behind the verdict.

`detail` carries enough to tell a one-ULP difference everywhere from a transposed band, since both
report "not equal". A red band names the first disagreeing position, which is what a heatmap would
show.
"""
struct BandResult
    file::String
    band::Int
    field::Symbol
    gate::String
    passed::Bool
    n::Int
    ndiff::Int
    worst::Float64
    detail::String
end

Base.show(io::IO, b::BandResult) =
    @printf(io, "%-28s b%d %-16s %-22s %s  %s", b.file, b.band, b.field, b.gate,
            b.passed ? "pass" : "FAIL", b.detail)

"""
    compare_int_band(file, band, field, ours, theirs) -> BandResult

Tier A: an integer band, bitwise or failed.

No tolerance is offered, because these are the outputs of a rounding conversion — a difference of one
is a different answer, not a rounder one.
"""
function compare_int_band(file, band, field, ours, theirs)
    size(ours) == size(theirs) || return BandResult(
        file, band, field, "bitwise", false, 0, 0, Inf,
        "shape $(size(ours)) against the reference's $(size(theirs))")

    ndiff = 0
    first_bad = nothing
    worst = 0.0
    for i in eachindex(IndexCartesian(), ours)
        a, b = Int64(ours[i]), Int64(theirs[i])
        a == b && continue
        ndiff += 1
        worst = max(worst, Float64(abs(a - b)))
        first_bad === nothing && (first_bad = (Tuple(i), a, b))
    end
    n = length(theirs)
    detail = if ndiff == 0
        "all $n equal"
    else
        pos, a, b = first_bad
        @sprintf("%d of %d differ (%.4f%%), worst by %.0f, first at %s: ours %d, reference %d",
                 ndiff, n, 100 * ndiff / n, worst, pos, a, b)
    end
    return BandResult(file, band, field, "bitwise", ndiff == 0, n, ndiff, worst, detail)
end

"""
    compare_float_band(file, band, field, ours, theirs; bound) -> BandResult

Tier B: a float band, bitwise when `bound` is zero and within `bound` relative otherwise.

`bound` is the caller's claim about the case, not a property of the band: pass `0.0` where the grid and
the image share a CRS, so no reprojection has happened and bitwise is achievable.
"""
function compare_float_band(file, band, field, ours, theirs; bound::Float64)
    size(ours) == size(theirs) || return BandResult(
        file, band, field, "shape", false, 0, 0, Inf,
        "shape $(size(ours)) against the reference's $(size(theirs))")

    gate = bound == 0 ? "bitwise" : @sprintf("relative < %.0e", bound)
    ndiff = 0
    worst = 0.0
    worst_at = nothing
    for i in eachindex(IndexCartesian(), ours)
        a, b = Float64(ours[i]), Float64(theirs[i])
        a === b && continue
        (isnan(a) && isnan(b)) && continue
        ndiff += 1
        # Non-finite on one side only is unbounded, whatever the magnitudes: a sentinel against a
        # computed value is a coverage difference and must not be absorbed by a relative measure.
        rel = (isfinite(a) && isfinite(b)) ? abs(a - b) / max(abs(a), abs(b), 1.0) : Inf
        if rel > worst
            worst = rel
            worst_at = (Tuple(i), a, b)
        end
    end
    n = length(theirs)
    passed = ndiff == 0 || (bound > 0 && worst < bound)
    detail = if ndiff == 0
        "all $n bitwise"
    else
        pos, a, b = worst_at
        @sprintf("%d of %d differ (%.2f%%), worst relative %.3g at %s: ours %.17g, reference %.17g",
                 ndiff, n, 100 * ndiff / n, worst, pos, a, b)
    end
    return BandResult(file, band, field, gate, passed, n, ndiff, worst, detail)
end

"""
    compare_geometry(r::GoldenRun, result; float_bound) -> Vector{BandResult}

Every band of `result` against the nine GeoTIFFs run `r` produced.

The file and band layout comes from `reference_files(result.coordinate)` rather than from a fixed list,
so the radar path's three-band off2vel files are read as three bands and the projected path's as two.
A reader indexes bands positionally, so the wrong count would shift every band after it silently.

A file the reference did not write is reported rather than skipped: it writes no file for an output its
inputs did not support, so its absence is a claim about the inputs that this package's own output must
match.
"""
function compare_geometry(r::GoldenRun, result; float_bound::Float64)
    out = BandResult[]
    for (file, fields) in reference_files(result.coordinate)
        path = joinpath(r.dir, file)
        if !isfile(path)
            # The reference writes no file when every band of it would be nodata. Ours must agree.
            for (b, f) in enumerate(fields)
                ours = getfield(result, f)
                sentinel = eltype(ours)(result.nodata.output)
                allsent = all(==(sentinel), ours)
                push!(out, BandResult(file, b, f, "absent both sides", allsent, length(ours),
                                      allsent ? 0 : count(!=(sentinel), ours), 0.0,
                                      allsent ? "the reference wrote no file and every value here " *
                                                "is the sentinel" :
                                      "the reference wrote no file, but this package computed " *
                                      "$(count(!=(sentinel), ours)) values"))
            end
            continue
        end
        ArchGDAL.read(path) do ds
            nb = ArchGDAL.nraster(ds)
            nb == length(fields) || error(
                "$file has $nb bands but the $(typeof(result.coordinate).name.name) layout " *
                "expects $(length(fields)); a positional reader would misread every band")
            for (b, f) in enumerate(fields)
                # GDAL hands back (x, y), which is the orientation this package's arrays use.
                theirs = ArchGDAL.read(ds, b)
                ours = getfield(result, f)
                push!(out, eltype(ours) <: Integer ?
                           compare_int_band(file, b, f, ours, theirs) :
                           compare_float_band(file, b, f, ours, theirs; bound = float_bound))
            end
        end
    end
    return out
end

"""
    compare_coverage(r::GoldenRun, result) -> BandResult

How many points each side computed, from `window_location`'s first band.

Separate from the band comparison because it is the one number that says whether a band agreement is
meaningful: two grids of sentinels agree bitwise.
"""
function compare_coverage(r::GoldenRun, result)
    path = joinpath(r.dir, "window_location.tif")
    theirs = ArchGDAL.read(ds -> ArchGDAL.read(ds, 1), path)
    sentinel = Int32(result.nodata.output)
    ours_n = count(!=(sentinel), result.location_x)
    theirs_n = count(!=(Int32(r.nodata)), theirs)
    ok = ours_n == theirs_n
    return BandResult("coverage", 1, :location_x, "counts equal", ok, length(theirs),
                      abs(ours_n - theirs_n), Float64(abs(ours_n - theirs_n)),
                      ok ? "both computed $ours_n of $(length(theirs)) points" :
                      "this package computed $ours_n points, the reference $theirs_n")
end

"""
    report(rs) -> Bool

Print each band's verdict and return whether all passed. Failures last, since a sweep's useful output
is the red rows.
"""
function report(rs::Vector{BandResult})
    for b in rs
        println("  ", b)
    end
    nfail = count(b -> !b.passed, rs)
    @printf("  %d of %d bands passed\n", length(rs) - nfail, length(rs))
    return nfail == 0
end
