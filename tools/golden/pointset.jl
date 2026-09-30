# Layer 3: this package's geometry, converted by AutoRIFT.jl, against what Python's correlator got.
#
#     julia --project=tools/golden -t 8 tools/golden/pointset.jl LC08_L1TP_009011
#     julia --project=tools/golden -t 8 tools/golden/pointset.jl
#
# The end-to-end claim, and the only rung where all three pieces are in play at once:
# `pairgeometry_blocked` computes the geometry, `AutoRIFT.pointset` converts it, and the result is
# diffed against `capture/in_*` — the arrays Python autoRIFT was handed. Layer 1 established that the
# geometry matches the reference's GeoTIFFs and Layer 2 that the derivation from those GeoTIFFs to the
# correlator's inputs is exact; this composes the two through the conversion that sits between them.
#
# So a difference here that Layers 1 and 2 both pass is a difference in `AutoRIFT.pointset`, which is
# what makes this worth running as its own layer rather than trusting the composition.
#
# The conversion's own header enumerates five ways it can silently misbehave — the index base, the half
# pixel counted twice, a nodata sentinel arriving as a negative radius, the y sign, a dropped chip bound
# — and every one produces a field that still reads as a search grid. Each has a rung below.
#
# **The x and y relations are measured, not assumed.** `AutoRIFT.jl/tools/golden/intermediate.jl`
# records that reasoning about the half pixel gave the wrong answer there and a scan gave the right one,
# and that the failure is invisible in a median: a half-pixel offset produces no residual under uniform
# motion and one proportional to the local velocity gradient. So `scan_offset` reports the whole profile
# over `{0, ±0.5, ±1}` before any relation is asserted, and the assertion names the offset it found.

using AutoRIFT
using ImagePairGeometry
using ImagePairGeometry: y_displacement_sign
using Printf

include(joinpath(@__DIR__, "cases.jl"))
include(joinpath(@__DIR__, "compare.jl"))
include(joinpath(@__DIR__, "scenes.jl"))
include(joinpath(@__DIR__, "params.jl"))
include(joinpath(@__DIR__, "handoff.jl"))
include(joinpath(@__DIR__, "geogrid_optical.jl"))

Base.get_extension(AutoRIFT, :AutoRIFTImagePairGeometryExt) === nothing &&
    error("""AutoRIFT's ImagePairGeometry extension did not load, so
             `AutoRIFT.pointset(::PairGeometry)` does not exist and every rung below would fail on a
             MethodError rather than on the value it is checking.""")

"""
    scan_offset(ours, theirs, mask; offsets = (-1.0, -0.5, 0.0, 0.5, 1.0)) -> Vector

For each candidate offset, how many of the compared points `ours + offset` matches `theirs` exactly.

The measurement that decides the index convention rather than an argument about it. Reported before any
relation is asserted, because the two half pixels in play — geogrid's zero-based index and the
correlator's `round(x) + 0.5` — do not cancel, and picking wrong is invisible in a summary statistic.
"""
function scan_offset(ours, theirs, mask; offsets = (-1.0, -0.5, 0.0, 0.5, 1.0))
    n = count(mask)
    return [(off, count(i -> mask[i] && Float64(ours[i]) + off == Float64(theirs[i]),
                        eachindex(ours, theirs)), n) for off in offsets]
end

"""
    scan_detail(scan) -> String

A scan profile as one line: each offset and the fraction it matched.
"""
scan_detail(scan) = join((@sprintf("%+.1f:%.4f", off, n == 0 ? 0.0 : hit / n)
                          for (off, hit, n) in scan), "  ")

"""
    check_pointset(r::GoldenRun; ntasks) -> Vector{BandResult}

Every rung of the `PointSet` conversion for run `r`.

Needs a capture, since the comparison is against the correlator's own inputs. A run without one returns
no rungs and is reported as uncovered.
"""
function check_pointset(r::GoldenRun; ntasks::Integer = max(1, Threads.nthreads()))
    k = capture_of(r)
    k === nothing && return BandResult[]
    r.radar && return BandResult[]          # the radar geometry is Layer 1's remaining work
    out = BandResult[]

    g = geogrid_result(r; ntasks)

    # The base chip extent the reference used, from the capture rather than derived — Layer 2 rung 6
    # already checked that `chip_size_pixels` reproduces it, so using it here keeps this layer about the
    # conversion.
    chip0x = Int(k.scalars["ChipSize0X"])
    pts = AutoRIFT.pointset(g; chip_size = chip0x)

    # The conversion works on the whole geogrid window; the correlator's arrays are truncated to the
    # chop. Compare over the region both cover, which is the capture's own extent.
    rlim, clim = size(cap(k, "in_xGrid"))
    # `pointset` returns arrays in this package's (x, y) orientation and the capture is (y, x).
    P(f) = permutedims(getfield(pts, f))[1:rlim, 1:clim]

    capx = cap(k, "in_xGrid")
    capy = cap(k, "in_yGrid")
    # The points the correlator actually searched: everywhere the driver did not zero the grid. The
    # remaining rungs are about those, since a skipped point's position is arbitrary on both sides.
    kept = .!captured_skip_mask(k)
    nkept = count(kept)

    # Rung 1: the index convention, measured. `pointset` makes geogrid's zero-based index one-based and
    # deliberately does *not* add the correlator's half pixel, so the expected relation is
    # `pts.x == in_xGrid + 0.5`: one for the base, less the half the capture already carries.
    for (name, ours, theirs) in (("1 x convention", P(:x), capx), ("1 y convention", P(:y), capy))
        scan = scan_offset(ours, theirs, kept)
        best = argmax(t -> t[2], scan)
        exact = best[2] == nkept
        push!(out, BandResult(name, 1, :position, "one offset matches every point", exact, nkept,
                              nkept - best[2], 0.0,
                              @sprintf("best offset %+.1f matches %d of %d; profile  %s",
                                       best[1], best[2], nkept, scan_detail(scan))))
    end

    # Every rung below compares only the searched points. `pointset` knows nothing of the driver's image
    # mask — it reads a `PairGeometry`, which carries geogrid's nodata and nothing else — so at a point
    # the driver zeroed for an image reason the two legitimately disagree: the capture holds 0 and the
    # conversion holds whatever geogrid computed. That is a property of the mask, established as
    # unreproducible in Layer 2, not of the conversion. Masking rather than reporting, because here the
    # disagreement is expected at every such point rather than at a few.
    #
    # The masking is what makes these rungs meaningful, so it is also what could hide a real difference:
    # `nkept` is printed on the convention rungs above, and Layer 2 rung 3 prints how many points the
    # mask removes.
    M(a) = [a[i] for i in eachindex(a) if kept[i]]

    # Rung 2: the search radii. A skipped point carries radius zero, which is how `PointSet` marks a
    # point not to search — and a sentinel passed through instead would make it negative, which
    # `gridpoints` would then size its margin from. The sign check runs over the *whole* grid, since a
    # negative radius anywhere is a defect whether or not that point is searched.
    for (name, f, capname) in (("2 radius x", :radius_x, "in_SearchLimitX"),
                               ("2 radius y", :radius_y, "in_SearchLimitY"))
        ours = P(f)
        # `BOUNDARY_POINTS` of one, for the same reason Layer 1 allows it on the integer bands: a search
        # extent whose float lands within the transform's noise of a `.5` boundary rounds the other way.
        # The one such point in the golden set reaches this rung through the geometry — `search_x` at
        # (1132, 1489) of `LC08_L1TP_062018`, 34 here and 35 in the capture — so allowing it here and not
        # there would make this layer stricter than the geometry it converts.
        push!(out, compare_int_band(name, 1, f, Int.(M(ours)), Int.(M(cap(k, capname)));
                                    allow_boundary = BOUNDARY_POINTS))
        whole = getfield(pts, f)
        neg = count(<(0), whole)
        push!(out, BandResult(name * " sign", 1, f, "no negative radius", neg == 0, length(whole),
                              neg, 0.0,
                              neg == 0 ? "every radius over the whole grid is zero or positive" :
                              "$neg radii are negative, which is a sentinel passed through as a value"))
    end

    # Rung 3: the priors, and the y sign. `y_displacement_sign` answers the sign from the coordinate
    # system, so a projected pair needs no negation and a radar one does; the capture is what says
    # whether the answer is right.
    push!(out, compare_float_band("3 dx prior", 1, :dx_prior, Float32.(M(P(:dx_prior))),
                                  Float32.(M(cap(k, "in_Dx0"))); bound = 0.0))
    push!(out, compare_float_band("3 dy prior", 1, :dy_prior, Float32.(M(P(:dy_prior))),
                                  Float32.(M(cap(k, "in_Dy0"))); bound = 0.0))
    sgn = y_displacement_sign(g.coordinate)
    want = r.radar ? -1.0 : 1.0
    push!(out, BandResult("3 y sign", 1, :dy_prior, "matches the coordinate system", sgn == want, 1,
                          sgn == want ? 0 : 1, 0.0,
                          "y_displacement_sign = $sgn, expected $want for a " *
                          (r.radar ? "radar" : "projected") * " pair"))

    # Rung 4: the chip-size bounds, which are per point and where zero means unbounded — the same thing
    # a sentinel means, so the conversion has to map one to the other rather than pass it through.
    push!(out, compare_float_band("4 chip min x", 1, :chip_size_min_x,
                                  Float32.(M(P(:chip_size_min_x))),
                                  Float32.(M(cap(k, "in_ChipSizeMinX"))); bound = 0.0))
    push!(out, compare_float_band("4 chip max x", 1, :chip_size_max_x,
                                  Float32.(M(P(:chip_size_max_x))),
                                  Float32.(M(cap(k, "in_ChipSizeMaxX"))); bound = 0.0))

    # Rung 5: the base chip extent, and its y counterpart through `ScaleChipSizeY`.
    scy = Float64(k.scalars["ScaleChipSizeY"])
    want_y = round(Int, chip0x * scy)
    okx = all(==(chip0x), pts.chip_size_x)
    oky = all(==(want_y), pts.chip_size_y)
    push!(out, BandResult("5 chip size", 1, :chip_size_x, "uniform and equal to the capture's",
                          okx && oky, length(pts.chip_size_x), (okx ? 0 : 1) + (oky ? 0 : 1), 0.0,
                          "x $(okx ? chip0x : "varies") against capture $chip0x; " *
                          "y $(oky ? want_y : "varies") against round($chip0x * $scy) = $want_y"))

    # Rung 6: blocking invariance. Points are independent, so a blocked run must give the same
    # `PointSet` — asserted here on a real window with a real nodata pattern rather than on a fixture.
    small = pairgeometry_blocked_like(r, g; blocksize = (256, 256), ntasks = 1)
    ps2 = AutoRIFT.pointset(small; chip_size = chip0x)
    same = all(f -> getfield(pts, f) == getfield(ps2, f),
               (:x, :y, :radius_x, :radius_y, :dx_prior, :dy_prior, :chip_size_x, :chip_size_y,
                :chip_size_min_x, :chip_size_max_x))
    push!(out, BandResult("6 blocking", 1, :all, "identical under a different tiling", same, 1,
                          same ? 0 : 1, 0.0,
                          same ? "a (256, 256) single-task run gives the same PointSet" :
                          "a (256, 256) single-task run gives a different PointSet"))

    return out
end

"""
    pairgeometry_blocked_like(r, g; blocksize, ntasks) -> PairGeometry

Run `r`'s geometry again at a different tiling, for the blocking-invariance rung.
"""
function pairgeometry_blocked_like(r::GoldenRun, g; blocksize, ntasks)
    paths = fetch_params(r)
    dem = Raster(paths["dem"]; lazy = true, missingval = nothing)
    ff = mapgrid(dem)
    grid = MapGrid(geotransform = ff.geotransform, size = ff.size, crs = r.epsg)
    coord = overlap_geometry(r)
    fp = ImageFootprint(origin = coord.origin, spacing = coord.spacing, size = coord.size)
    pair = coregister(fp, fp; dt = r.dt)
    win = CartesianIndices((1:ff.size[1], 1:ff.size[2]))
    lazy(f) = haskey(paths, f) ? Raster(paths[f]; lazy = true, missingval = nothing) : nothing
    src = RasterInputs(dem = dem, dhdx = lazy("dhdx"), dhdy = lazy("dhdy"),
                       vx = lazy("vx"), vy = lazy("vy"), srx = lazy("srx"), sry = lazy("sry"),
                       csminx = lazy("csminx"), csminy = lazy("csminy"),
                       csmaxx = lazy("csmaxx"), csmaxy = lazy("csmaxy"), ssm = lazy("ssm"))
    return pairgeometry_blocked(grid, pair, src;
                                transform = () -> fast_transform(r.epsg, image_crs(r)),
                                window = win, blocksize, ntasks,
                                params = GeometryParams(chip_size_0 = r.chip_size_0),
                                nodata = nodata_from(r.nodata))
end

"""
    main_pointset(args) -> Bool

Check the conversion for every covered run named by `args`.
"""
function main_pointset(args)
    name = isempty(args) || startswith(first(args), "--") ? nothing : first(args)
    runs = goldenruns(; name)
    isempty(runs) && error("no golden run on disk" * (name === nothing ? "" : " matching \"$name\""))
    allok = true
    ncov = 0
    for r in runs
        rs = check_pointset(r)
        if isempty(rs)
            @printf("\n=== %s run %d  — %s\n", short_name(r), r.run,
                    r.radar ? "radar, pending Layer 1's radar path" : "no capture")
            continue
        end
        ncov += 1
        @printf("\n=== %s run %d\n", short_name(r), r.run)
        allok &= report(rs)
    end
    @printf("\n%d runs covered\n", ncov)
    println(allok ? "every rung of every covered run agrees" : "at least one rung disagrees")
    return allok
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main_pointset(ARGS) ? 0 : 1)
end
