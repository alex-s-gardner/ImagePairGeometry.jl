# Layer 2: the geogrid output, as the Python correlator received it.
#
#     julia --project=tools/golden tools/golden/handoff.jl LC08_L1TP_009011
#     julia --project=tools/golden tools/golden/handoff.jl
#
# Between geogrid writing nine GeoTIFFs and `autoRIFT.autorift()` reading its grid sit two files of
# Python — `testautoRIFT.py`'s `runAutorift` and `autoRIFT.runAutorift` — which rewrite every array.
# `capture.py` intercepts that boundary and dumps what the correlator was handed, so `capture/in_*`
# holds the ground truth for the handoff, and this file checks the derivation against it.
#
# The rungs are fed the reference's own `window_*.tif` and diffed against the reference's own `in_*`,
# never against each other's output: this is a test of the *transform*, and chaining would rebuild
# the composed comparison it exists to take apart. Layer 3 is where this package's geometry enters.
#
# The derivation, and the line of Python each step is:
#
#   1. read band 1/2 of the GeoTIFF, and `permutedims` to the capture's (row = y, col = x)
#      orientation, since GDAL hands Julia (x, y) and numpy wrote (y, x);
#   2. `noDataMask = xGrid == nodata`, and zero every array there (`testautoRIFT.py:390-400`);
#   3. additionally zero a point the imagery marks unusable — read from the image samples themselves,
#      or from a `zero_mask` raster on a `wallis_fill` run (`testautoRIFT.py:337-349`);
#   4. truncate to `rlim × clim`, where `lim = floor(dim / chopFactor) * chopFactor` and
#      `chopFactor = max(ChipSizeMaxX) / ChipSize0X` (`autoRIFT.py:882-887`);
#   5. the grid alone becomes `round(v) + 0.5` as `Float32` (`autoRIFT.py:890-891`);
#   6. radar negates `Dy0`, optical does not (`testautoRIFT.py:404-406`).
#
# Steps 2 and 3 both zero, and they are separated deliberately. Step 2 is geogrid's own statement about
# where the grid falls outside the image; step 3 is a property of the *imagery* and is not reproducible
# from this capture, for a different reason in each of its two branches — see `captured_skip_mask`.
# Folding them together would let a wrong answer in either hide in the other.

using ImagePairGeometry: chip_size_pixels
using Printf
using Statistics

include(joinpath(@__DIR__, "cases.jl"))
include(joinpath(@__DIR__, "compare.jl"))

"""
    Capture

One run's captured correlator inputs: the arrays, and the scalars from `call1.json`.

Absent for a run whose `capture/` directory was never produced — several cases have geogrid output and
no capture, and those are reported as uncovered rather than skipped silently.
"""
struct Capture
    dir::String
    scalars::Dict{String,Any}
    json::Any
end

"""
    capture_of(r::GoldenRun) -> Union{Capture,Nothing}

Run `r`'s captured correlator inputs, or `nothing` when it has none.
"""
function capture_of(r::GoldenRun)
    dir = joinpath(r.dir, "capture")
    js = joinpath(dir, "call1.json")
    isfile(js) || return nothing
    j = JSON3.read(read(js, String))
    scalars = Dict{String,Any}(String(k) => v for (k, v) in pairs(j.scalars))
    return Capture(dir, scalars, j)
end

"""
    cap(k::Capture, name) -> Matrix

One captured array, by the name `call1.json` keys it under.

The `.abx` files carry their own shape and element type, so a transpose is an error rather than a
silent scramble — see `AutoRIFT.jl/tools/ab/xchg.jl`.
"""
cap(k::Capture, name::AbstractString) = xread(joinpath(k.dir, name))

# ---------------------------------------------------------------------------
# The derivation
# ---------------------------------------------------------------------------

"""
    geogrid_bands(r::GoldenRun) -> Dict{String,Matrix}

Every band of run `r`'s nine geogrid GeoTIFFs, in the capture's `(row = y, col = x)` orientation.

GDAL hands back `(x, y)` and numpy wrote `(y, x)`, so each band is transposed on the way in. Getting
this wrong is not subtle on a non-square grid and is invisible on a square one, which is why
`selftest.jl` injects a transpose on a non-square product.
"""
function geogrid_bands(r::GoldenRun)
    out = Dict{String,Matrix{Float64}}()
    for (file, fields) in (("window_location.tif", ("xGrid", "yGrid")),
                           ("window_offset.tif", ("Dx0", "Dy0")),
                           ("window_search_range.tif", ("SearchLimitX", "SearchLimitY")),
                           ("window_chip_size_min.tif", ("ChipSizeMinX", "ChipSizeMinY")),
                           ("window_chip_size_max.tif", ("ChipSizeMaxX", "ChipSizeMaxY")))
        path = joinpath(r.dir, file)
        isfile(path) || continue
        ArchGDAL.read(path) do ds
            for (b, name) in enumerate(fields)
                b <= ArchGDAL.nraster(ds) || continue
                out[name] = permutedims(Float64.(ArchGDAL.read(ds, b)))
            end
        end
    end
    return out
end

"""
    chop_limits(csmaxx, chip_size_0x) -> (rlim, clim)

The grid extent the correlator's nested pyramid can use, from `autoRIFT.py:882-887`.

`chopFactor` is the largest chip size in units of the base, and the grid is truncated to a whole number
of those in each direction. The truncation is at the far edge, so it removes rows and columns rather
than shifting anything.

**`csmaxx` must already be zeroed at the skipped points.** `testautoRIFT.py:396-398` zeroes
`ChipSizeMaxX` wherever the driver's mask applies, and `runAutorift` takes its maximum *after* that
(`autoRIFT.py:882-887`) — so a chip size that survives only on skipped points does not set the factor.
It is not a subtle difference: on `LC08_L1TP_060018` all 162,117 points holding 128 are zeroed, so the
maximum is 64 rather than 128 and the factor halves from 8 to 4, which changes the grid from
1760 × 2072 to 1760 × 2076. Four columns of a 2076-column grid, entering every array the correlator
receives.
"""
function chop_limits(csmaxx::AbstractMatrix, chip_size_0x::Real)
    chop = maximum(csmaxx) / chip_size_0x
    return (Int(floor(size(csmaxx, 1) / chop) * chop),
            Int(floor(size(csmaxx, 2) / chop) * chop))
end

"""
    chip_size_0x(r, bands) -> Int

`ChipSize0X`: the base chip extent in pixels, `ceil(chip_size_0 / pixel_size / 4) * 4`
(`testautoRIFT.py:355-370`).

`chip_size_pixels` is this package's own form of the same expression, so this is the one scalar of the
handoff that Layer 2 checks against a function rather than against a captured array.
"""
chip_size_0x(r::GoldenRun, pixel_size::Real) =
    chip_size_pixels(r.chip_size_0, pixel_size)

"""
    scale_chip_size_y(csminx, csminy, nodata) -> Float64

`ScaleChipSizeY`: the median of `CSMINy0 / CSMINx0` over points where neither is the sentinel
(`testautoRIFT.py:374`).

Over the *untruncated* grid, since the reference computes it before `runAutorift` chops.
"""
function scale_chip_size_y(csminx::AbstractMatrix, csminy::AbstractMatrix, nodata::Real)
    ratios = Float64[]
    for i in eachindex(csminx, csminy)
        (csminx[i] == nodata || csminy[i] == nodata) && continue
        push!(ratios, Float64(csminy[i]) / Float64(csminx[i]))
    end
    return median(ratios)
end

"""
    captured_skip_mask(k::Capture) -> BitMatrix

The points the reference's driver zeroed, read from the capture rather than derived.

Two rules zero a point (`testautoRIFT.py:337-349, 390-400`): geogrid's own nodata sentinel, and the
imagery marking the point unusable. The first is derivable from the geogrid output; the second is not,
from this capture, in either of the two branches it takes.

On an ordinary run the rule reads the image samples, and `capture.py`'s own header says why they are
unavailable: `in_I1` is taken at the `runAutorift` boundary, which is **downstream of the Wallis or FFT
prefilter**, while the rule runs on the image as read. The filter changes which samples are zero, so the
array on disk is not the one the rule was applied to.

On a `wallis_fill` run the rule instead looks up the `_zeroMask` raster the fill wrote beside the
filtered scene, which the capture does not include at all.

So the mask is read off the captured grid. A zeroed point is `0.5` in *both* axes — the driver wrote 0
to `xGrid` and `yGrid` together and `runAutorift` then added the half pixel — and both are required,
because `xGrid` alone is ambiguous: pixel column 0 is a real index, and geogrid writes it for a grid
point that lands on the image's left edge. On `S2B_MSIL1C_20200612` there are 79 such points, whose
`yGrid` is a large real value; treating them as zeroed removes them from the comparison and reports each
of six arrays as differing there.

That makes the mask an input to the remaining rungs rather than something they check, which is honest
about what this capture can establish: rungs 4 and 5 then test the grid convention, the truncation and
the pass-through of six arrays against the reference's own, on the reference's own set of searched
points.
"""
captured_skip_mask(k::Capture) =
    (cap(k, "in_xGrid") .== 0.5f0) .& (cap(k, "in_yGrid") .== 0.5f0)

"""
    uses_wallis_fill(r::GoldenRun) -> Bool

Whether this run's preprocessing fills nodata, which changes *how* the imagery marks a point unusable.

Not whether it does so at all. The container's `testautoRIFT.py:337-349` takes one of two branches: the
image-sample test on an ordinary run, and a lookup into the `_zeroMask` raster the Wallis fill wrote on
a `wallis_fill` one. Upstream autoRIFT 2.1.2 has only the first branch and a comment saying the test is
disabled here, so reading the upstream source alone predicts no extra zeroing — measured against the
captures, a `wallis_fill` run zeroes 1.59 million points beyond geogrid's own.

Read from the log's own `Using preprocessing methods` line rather than inferred from the platform: the
choice is per scene and a mixed pair takes the most stringent, so the log is the only record of what
the run actually did.
"""
function uses_wallis_fill(r::GoldenRun)
    text = read(joinpath(r.dir, "capture.log"), String)
    m = match(r"^Using preprocessing methods (.*)$"m, text)
    m === nothing && error("the log of $(short_name(r)) run $(r.run) names no preprocessing methods")
    return occursin("wallis_fill", m.captures[1])
end

# ---------------------------------------------------------------------------
# The rungs
# ---------------------------------------------------------------------------

"""
    check_handoff(r::GoldenRun) -> Vector{BandResult}

Every rung of the handoff for run `r`: the derived arrays against the captured ones.

The rungs run in order and each is reported whatever the ones before it did, because they test
independent steps of one transform rather than a chain — a shape rung failing tells you nothing about
the grid convention, and both are worth knowing at once. Only the shape rung short-circuits, since a
value comparison against a differently shaped array has nothing to say.
"""
function check_handoff(r::GoldenRun)
    k = capture_of(r)
    k === nothing && return BandResult[]
    out = BandResult[]

    bands = geogrid_bands(r)
    for need in ("xGrid", "yGrid", "Dx0", "Dy0", "SearchLimitX", "SearchLimitY",
                 "ChipSizeMinX", "ChipSizeMaxX")
        haskey(bands, need) || error(
            "$(short_name(r)) run $(r.run) has no geogrid band for `$need`, which the handoff needs")
    end
    nd = r.nodata

    # Rung 1: the truncation. `chopFactor` is the largest chip size *after* the driver's mask has zeroed
    # its own points (see `chop_limits`), and that mask is not reproducible here — so the shape is taken
    # from the capture and what is asserted is that it is a valid chop: a whole number of chips in each
    # direction, at a factor the chip sizes admit, losing fewer than one chip's worth of rows and
    # columns.
    #
    # That is weaker than deriving it, and it is what the capture supports. The factor is 8 on
    # `LC08_L1TP_009011` and 4 on `LC08_L1TP_060018`, and the difference is 162,117 points holding 128
    # that the mask zeroes on the second — so a derivation from the unmasked band gets 1760 × 2072 where
    # the run used 1760 × 2076.
    chip0x = Int(k.scalars["ChipSize0X"])
    full = size(bands["xGrid"])
    rlim, clim = size(cap(k, "in_xGrid"))
    factor = maximum(cap(k, "in_ChipSizeMaxX")) / chip0x
    valid = factor >= 1 && rlim % factor == 0 && clim % factor == 0 &&
            0 <= full[1] - rlim < factor && 0 <= full[2] - clim < factor
    push!(out, BandResult("1 shape", 1, :chop, "a valid chop of the grid", valid, 1,
                          valid ? 0 : 1, 0.0,
                          "$full chopped to $((rlim, clim)) at chopFactor $factor"))
    valid || return out

    S(a) = @view a[1:rlim, 1:clim]
    xg_full, yg_full = bands["xGrid"], bands["yGrid"]

    # Rung 2: geogrid's own nodata mask must be a *subset* of what the driver zeroed. That direction is
    # the whole of what is checkable — the driver zeroes for two reasons and only this one is derivable
    # from the geogrid output, per `captured_skip_mask`. A geogrid sentinel the driver did not zero would
    # be a real disagreement; the converse is the image-zero rule, which rung 3 counts.
    ndmask = S(xg_full) .== nd
    total = captured_skip_mask(k)
    escaped = count(ndmask .& .!total)
    push!(out, BandResult("2 nodata mask", 1, :noDataMask, "geogrid nodata ⊆ zeroed", escaped == 0,
                          length(total), escaped, 0.0,
                          @sprintf("geogrid nodata %d of %d, all within the %d the driver zeroed",
                                   count(ndmask), length(ndmask), count(total))))

    # Rung 3: how much of the driver's mask the imagery contributes, as a count. Reported rather than
    # gated in both branches, because neither is reproducible from this capture — see
    # `captured_skip_mask`. The count is worth printing anyway: it is a third of the grid on some cases,
    # so a comparison that silently folded it into the nodata mask would be describing that third
    # wrongly, and a *change* in it means the run or the capture moved.
    wallis = uses_wallis_fill(r)
    extra = count(total .& .!ndmask)
    src = wallis ? "the `_zeroMask` raster the Wallis fill wrote" : "the image before filtering"
    pct = @sprintf("%.1f%%", 100 * extra / length(total))
    push!(out, BandResult("3 image mask", 1, :imagemask, "reported (not in the capture)", true,
                          length(total), extra, 0.0,
                          "$extra of $(length(total)) points zeroed beyond geogrid's own ($pct), " *
                          "from $src"))

    # Rungs 4 and 5: the arrays themselves. The grid takes `round(v) + 0.5`; everything else passes
    # through. `Float32` on both sides, which is what the correlator receives.
    # Compared as `Float32` values rather than as bit patterns. The distinction matters in exactly one
    # place: negating a zeroed `Dy0` gives `-0.0`, which is the same number as the `0.0` the reference's
    # array holds and a different bit pattern. Every other value here is either an integer held in a
    # float or a half-integer, so value equality and bit equality coincide — and `compare_float_band`
    # with a zero bound reports any real difference exactly as a bitwise comparison would.
    function rung(name, arr, capname; grid = false, negate = false)
        v = Float64.(S(arr))
        v = ifelse.(total, 0.0, v)
        negate && (v = .-v)
        ours = grid ? Float32.(round.(v) .+ 0.5f0) : Float32.(v)
        push!(out, compare_float_band(name, 1, Symbol(capname), ours, cap(k, capname); bound = 0.0))
    end

    rung("4 grid x", xg_full, "in_xGrid"; grid = true)
    rung("4 grid y", yg_full, "in_yGrid"; grid = true)
    rung("5 searchx", bands["SearchLimitX"], "in_SearchLimitX")
    rung("5 searchy", bands["SearchLimitY"], "in_SearchLimitY")
    rung("5 csminx", bands["ChipSizeMinX"], "in_ChipSizeMinX")
    rung("5 csmaxx", bands["ChipSizeMaxX"], "in_ChipSizeMaxX")
    rung("5 Dx0", bands["Dx0"], "in_Dx0")
    # The azimuth offset is negated for radar and passed through for optical.
    rung("5 Dy0", bands["Dy0"], "in_Dy0"; negate = r.radar)

    # Rung 6: the scalars. `ChipSize0X` is derived here through this package's own
    # `chip_size_pixels`, which is the one part of the handoff that is not array bookkeeping.
    px = r.radar ? r.radar_params.ground_range_size : abs(r.printed_spacing[1])
    if px !== nothing
        ours0x = chip_size_0x(r, px)
        ok = ours0x == chip0x
        push!(out, BandResult("6 ChipSize0X", 1, :ChipSize0X, "derived == captured", ok, 1,
                              ok ? 0 : 1, 0.0,
                              "chip_size_pixels($(r.chip_size_0), $px) = $ours0x, capture $chip0x"))
    end

    # `GridSpacingX = ChipSize0X * gridspacing / chipsize0` (`testautoRIFT.py:371`), integer-truncated.
    gsx = Int(k.scalars["GridSpacingX"])
    ours_gsx = Int(chip0x * r.grid_spacing / r.chip_size_0)
    ok = ours_gsx == gsx
    push!(out, BandResult("6 GridSpacingX", 1, :GridSpacingX, "derived == captured", ok, 1,
                          ok ? 0 : 1, 0.0, "derived $ours_gsx, capture $gsx"))

    # `ScaleChipSizeY` is a median over the untruncated grid, so it is compared as a float.
    if haskey(bands, "ChipSizeMinY")
        ours_scy = scale_chip_size_y(bands["ChipSizeMinX"], bands["ChipSizeMinY"], nd)
        theirs_scy = Float64(k.scalars["ScaleChipSizeY"])
        ok = ours_scy == theirs_scy
        push!(out, BandResult("6 ScaleChipSizeY", 1, :ScaleChipSizeY, "bitwise", ok, 1,
                              ok ? 0 : 1, abs(ours_scy - theirs_scy),
                              "derived $ours_scy, capture $theirs_scy"))
    end

    return out
end

"""
    main_handoff(args) -> Bool

Check the handoff for every run named by `args` that has a capture, printing each rung.
"""
function main_handoff(args)
    name = isempty(args) || startswith(first(args), "--") ? nothing : first(args)
    runs = goldenruns(; name)
    isempty(runs) && error("no golden run on disk" * (name === nothing ? "" : " matching \"$name\""))
    allok = true
    ncov = 0
    for r in runs
        rs = check_handoff(r)
        if isempty(rs)
            @printf("\n=== %s run %d  — no capture, handoff not covered\n", short_name(r), r.run)
            continue
        end
        ncov += 1
        @printf("\n=== %s run %d  [%s]\n", short_name(r), r.run, r.radar ? "radar" : "optical")
        allok &= report(rs)
    end
    @printf("\n%d runs covered by a capture\n", ncov)
    println(allok ? "every rung of every covered run agrees" : "at least one rung disagrees")
    return allok
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main_handoff(ARGS) ? 0 : 1)
end
