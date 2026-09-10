# Layer 1, radar path: this package's geogrid against the eight Sentinel-1 golden runs.
#
#     julia --project=tools/golden -t 8 tools/golden/geogrid_radar.jl --clock
#     julia --project=tools/golden -t 8 tools/golden/geogrid_radar.jl S1A_IW_SLC__1SSH_20151120
#
# The radar path needs a `RadarCoordinate`, and every field of one is on disk except the clock. The
# `.EOF` beside each run gives the orbit; the log's `Radar parameters:` block gives the starting range,
# the range sample spacing, the PRF, the image dimensions and the scene-center incidence angle; and
# `testGeogrid.py:146` hardcodes `LookSide.Right` for Sentinel-1.
#
# **The clock is the problem, and it has its own rung.** `sensing_start` is azimuth time in seconds
# since midnight, and the azimuth index is `round((aztime - sensing_start) * prf)` — so an error of one
# PRI, 2 ms, moves every azimuth index by one. The log prints `aztime` through C++ `%g`, six significant
# figures: `59052.9` for a value near 59052.9xx, which is ±0.05 s, or ±24 lines. Nothing else on disk
# carries it to better precision. The product's `img_pair_info:acquisition_date_img1` has microseconds
# but sits 1.5–3.4 s away from the log's value across the eight cases, so it is a different time than
# geogrid's `sensingStart`; the SAFE annotation that holds the real one is not among the run's files.
#
# So `solve_sensing_start` recovers it from the reference's own output instead. `window_location.tif`
# band 2 *is* the azimuth index geogrid computed, at 2.3–9.0 million points per case — and a wrong
# `sensing_start` shifts all of them by the same amount. Scanning the offset over the ±0.05 s the log
# leaves open and taking the one that reproduces the reference's own indices turns an unmeasurable
# quantity into a measured one, and the scan's shape is what says whether it is determined: a unique
# integer-line minimum, or nothing to stand on.
#
# That makes the radar comparison conditional in a way the optical one is not, and the report says so.
# Fixing the clock against the azimuth index spends that band as evidence: it can no longer corroborate
# the result. The range index, the chip sizes, the mask and every float band remain independent, and
# those are what this layer establishes.

using ImagePairGeometry
using ImagePairGeometry: nodata_from, mapgrid, window_geotransform, reference_files,
                         fast_transform,
                         Orbit, RadarCoordinate, CoregisteredPair, LookRight, incidence_angle,
                         azimuth_index
using Dates
using Rasters
using DimensionalData
using DiskArrays
using Printf
using StaticArrays

include(joinpath(@__DIR__, "cases.jl"))
include(joinpath(@__DIR__, "params.jl"))
include(joinpath(@__DIR__, "compare.jl"))

const RA_EXT = Base.get_extension(ImagePairGeometry, :ImagePairGeometryRastersExt)
RA_EXT === nothing && error("the Rasters extension did not load; `RasterInputs` is unavailable")
using .RA_EXT: RasterInputs

"""
    read_eof(path) -> (Orbit, DateTime)

The state vectors in a Sentinel-1 `.EOF` orbit file, and the epoch they are measured from.

The epoch is midnight of the first vector's day. A POEORB file starts at 22:59 the day *before* the
acquisition it covers, so that is not the acquisition day and `radar_coordinate` computes a nonzero
`orbit_epoch_offset` from the two dates.

Parsed directly rather than through a reader dependency: an `.EOF` is XML with one `<OSV>` per vector
holding `UTC`, `X`/`Y`/`Z` and `VX`/`VY`/`VZ` in ECEF meters, and nothing else here needs the format.
"""
function read_eof(path::AbstractString)
    text = read(path, String)
    times = DateTime[]
    pos = SVector{3,Float64}[]
    vel = SVector{3,Float64}[]
    for m in eachmatch(r"<OSV>(.*?)</OSV>"s, text)
        blk = m.captures[1]
        u = match(r"<UTC>UTC=(.*?)</UTC>", blk)
        u === nothing && continue
        push!(times, DateTime(u.captures[1], dateformat"yyyy-mm-ddTHH:MM:SS.s"))
        grab(tag) = parse(Float64, match(Regex("<$tag unit=\"[^\"]*\">(.*?)</$tag>"), blk).captures[1])
        push!(pos, SVector{3,Float64}(grab("X"), grab("Y"), grab("Z")))
        push!(vel, SVector{3,Float64}(grab("VX"), grab("VY"), grab("VZ")))
    end
    isempty(times) && error("$path holds no <OSV> state vectors")

    epoch = DateTime(Date(first(times)))
    secs = [(t - epoch).value / 1000 for t in times]
    # The vectors are uniformly spaced, which `Orbit` requires — asserted rather than assumed, since a
    # gap would make the interpolation silently wrong rather than fail.
    steps = diff(secs)
    all(≈(first(steps)), steps) || error(
        "the state vectors in $path are not uniformly spaced: steps $(unique(steps))")
    return Orbit(first(secs), first(steps), pos, vel), epoch
end

"""
    acquisition_date(r::GoldenRun) -> Date

The day of run `r`'s reference acquisition, from the product's first granule name.

`sensing_start` is seconds since midnight of this day, which is what makes it the reference for
`orbit_epoch_offset`.
"""
function acquisition_date(r::GoldenRun)
    m = match(r"_(\d{8})T\d{6}_", first(split(r.case.product, "_X_")))
    m === nothing && error("cannot read an acquisition date from $(r.case.product)")
    return Date(m.captures[1], dateformat"yyyymmdd")
end

"""
    reference_orbit(r::GoldenRun) -> (Orbit, DateTime)

The orbit covering run `r`'s reference acquisition.

Two `.EOF` files sit beside each run, one per acquisition, and neither filename says which is which
directly — but each names its own validity window, and the reference's is the one covering the
acquisition date in the product's first granule name.
"""
function reference_orbit(r::GoldenRun)
    isempty(r.orbit_files) && error("$(short_name(r)) run $(r.run) has no .EOF beside it")
    want = acquisition_date(r)

    for path in r.orbit_files
        # `..._VYYYYMMDDTHHMMSS_YYYYMMDDTHHMMSS.EOF` — the validity window.
        v = match(r"_V(\d{8})T\d{6}_(\d{8})T\d{6}\.EOF$", basename(path))
        v === nothing && continue
        lo = Date(v.captures[1], dateformat"yyyymmdd")
        hi = Date(v.captures[2], dateformat"yyyymmdd")
        lo <= want <= hi && return read_eof(path)
    end
    error("""none of the .EOF files beside $(short_name(r)) run $(r.run) covers $want:
             $(join(basename.(r.orbit_files), ", "))""")
end

"""
    radar_coordinate(r::GoldenRun, sensing_start) -> RadarCoordinate

Run `r`'s reference acquisition as a [`RadarCoordinate`](@ref), at the given clock.

`sensing_start` is separate from the rest because it is the one field the run's files do not pin — see
this file's header and [`solve_sensing_start`](@ref).
"""
function radar_coordinate(r::GoldenRun, sensing_start::Real)
    p = r.radar_params
    orbit, epoch = reference_orbit(r)
    # The two clocks, and the constant between them. `sensing_start` is seconds since midnight of the
    # *acquisition* day; the orbit is measured from midnight of its own first state vector's day. A
    # POEORB file spans 22:59 the day before through 01:00 the day after, so those are different days
    # and the offset is a whole 86400 s — not zero. Computed from the two dates rather than assumed,
    # because assuming zero puts the interpolation outside the orbit's span, which `Orbit` refuses; a
    # sign error would put it inside the span at the wrong place and be silent.
    acq = acquisition_date(r)
    offset = Float64((acq - Date(epoch)).value * 86_400)
    kw = (; orbit, starting_range = p.starting_range, dr = p.dr, sensing_start = Float64(sensing_start),
          prf = p.prf, nsamples = p.nsamples, nlines = p.nlines, look_side = LookRight,
          wavelength = 0.05546576, orbit_epoch_offset = offset)
    # The log prints the scene-center incidence angle geogrid computed; `incidence_angle` recomputes it
    # from the same geometry, and `check_incidence` compares the two.
    return RadarCoordinate(; kw..., incidence_angle = incidence_angle(; kw...))
end

"""
    reference_azimuth(r::GoldenRun) -> Matrix{Int32}

The azimuth index band of run `r`'s delivered `window_location.tif`.

Band 2, in this package's own `(x, y)` orientation, which is what GDAL hands back.
"""
reference_azimuth(r::GoldenRun) =
    ArchGDAL.read(ds -> ArchGDAL.read(ds, 2), joinpath(r.dir, "window_location.tif"))

# ---------------------------------------------------------------------------
# The clock
# ---------------------------------------------------------------------------

"""
    ClockSolution

What [`solve_sensing_start`](@ref) found: the recovered clock, and the evidence for it.

`unique` is the property that matters. A wrong `sensing_start` shifts every azimuth index by the same
whole number of lines, so the scan should give one clear winner and rivals a whole line away that miss
almost everything. Anything else — a tie, a broad plateau, no offset matching — means the clock is not
determined by this evidence, and the bands derived from it cannot be gated.

The winner does not match *every* point, and cannot. `REFERENCE.md` records a systematic offset of about
0.0013 azimuth lines between the compiled kernel's azimuth time and any external reproduction of it —
two independent reproductions bracket it and agree with each other to 4e-6 lines, so it is the gap
between the compiled kernel and an outside reimplementation rather than a transcription error. Every
point within that distance of a `std::round` boundary rounds the other way, which is a few percent of a
real scene. So the bar is the scan's *shape*: one winner far above every rival a line away.
"""
struct ClockSolution
    sensing_start::Float64
    lines::Float64
    matched::Int
    total::Int
    neighbours::Vector{Tuple{Float64,Int}}
    unique::Bool
end

"""
    count_matching(r, grid, dem, tf, theirs, idx, sensing_start) -> Int

How many of the sampled points' azimuth indices this package reproduces at the given clock.

The scan's inner loop, factored out so the whole-line pass and the fractional refinement measure the
same thing.
"""
function count_matching(r::GoldenRun, grid, dem, tf, theirs, idx, sensing_start::Real)
    coord = radar_coordinate(r, sensing_start)
    zd = ImagePairGeometry._bind(ImagePairGeometry.SceneCenterStart(), coord)
    hit = 0
    for i in idx
        gx, gy = ImagePairGeometry.gridpoint_center(grid, i[1], i[2])
        z = Float64(dem[i[1], i[2]])
        g, _, _ = ImagePairGeometry.pointgeometry(tf, gx, gy, z, coord,
                                                 ImagePairGeometry.NO_NORMAL, zd, nothing)
        _, az = g.image_xy
        ImagePairGeometry.cround32(az) == theirs[i] && (hit += 1)
    end
    return hit
end

"""
    solve_sensing_start(r::GoldenRun; halfwidth = 40, sample = 20000) -> ClockSolution

Recover run `r`'s `sensing_start` from the azimuth indices its geogrid wrote.

The log's `aztime` is a starting point good to ±0.05 s, or about ±24 lines. Every candidate here is that
value shifted by a whole number of PRIs, because a fractional shift is not what distinguishes two
plausible clocks: the index is `round((aztime - sensing_start) * prf)`, so shifting the clock by one PRI
shifts every index by exactly one.

The scan is over a sample of the grid rather than all of it — a few thousand points already separate the
candidates completely, and the full grid is millions. `check_clock` then verifies the winner over every
point.
"""
function solve_sensing_start(r::GoldenRun; halfwidth::Integer = 40, sample::Integer = 20_000,
                             substeps::Integer = 8)
    p = r.radar_params
    theirs = reference_azimuth(r)
    sentinel = Int32(r.nodata)

    paths = fetch_params(r)
    dem = Raster(paths["dem"]; lazy = true, missingval = nothing)
    ff = mapgrid(dem)
    grid = MapGrid(geotransform = ff.geotransform, size = ff.size, crs = r.epsg)

    # A sample of the points the reference computed, spread over the grid rather than clustered: a
    # contiguous block would sit at one range and azimuth and separate the candidates less.
    valid = [i for i in eachindex(IndexCartesian(), theirs) if theirs[i] != sentinel]
    isempty(valid) && error("$(short_name(r)) run $(r.run) has no valid azimuth index to solve against")
    step = max(1, length(valid) ÷ sample)
    idx = valid[1:step:end]

    # Resident, since the sample reads scattered points rather than a window. `missingval = nothing` for
    # the same reason `RasterInputs` requires it: the kernel works in `Float64` and decides what is
    # missing from the DEM's own sentinel, so a masked read would hand it `missing` instead of the height
    # the reference used.
    demdata = Raster(paths["dem"]; missingval = nothing)

    # The radar solve works in geographic coordinates: `rdr2geo` returns lon/lat/height and the grid is
    # projected, so the transform is grid-to-4326 rather than the identity. `test/radar_itslive_product.jl`
    # uses the same pair. Passing the identity instead hands the solve a polar-stereographic easting as a
    # longitude, which does not fail — it returns indices of the wrong order of magnitude entirely
    # (465713 against the reference's 12242), so no clock offset matches anything.
    tf = fast_transform(r.epsg, 4326)

    # Whole lines first, then the fraction within the winning line. A whole-line scan gets most of the
    # way — it is what the index quantization exposes — but not all: the log's `aztime` is rounded to
    # six figures, so the true clock sits at a fractional line offset, and a point whose unrounded index
    # lands near a `.5` boundary rounds the other way. On `S1A_IW_SLC__1SSH_20151120` the best whole line
    # matches 91.4% and the fraction lifts it to essentially all.
    counts = Tuple{Float64,Int}[]
    for dl in (-halfwidth):halfwidth
        # One PRI per line, subtracted: a later `sensing_start` gives a smaller index.
        hit = count_matching(r, grid, demdata, tf, theirs, idx, p.aztime + dl / p.prf)
        push!(counts, (Float64(dl), hit))
    end

    coarse = argmax(t -> t[2], counts)
    # Refine within one line of the winner, at `substeps` per line. The refinement is over fractions of a
    # PRI, which no longer shift the index uniformly — they move the points whose unrounded index sits
    # nearest a rounding boundary, which is exactly the residual the whole-line scan leaves.
    fine = Tuple{Float64,Int}[]
    for k in (-substeps):substeps
        dl = coarse[1] + k / substeps
        any(t -> t[1] == dl, counts) && continue
        push!(fine, (dl, count_matching(r, grid, demdata, tf, theirs, idx,
                                       p.aztime + dl / p.prf)))
    end
    append!(counts, fine)

    best = argmax(t -> t[2], counts)
    # A rival is a candidate a *different* line away: the fractional neighbours of the winner are the
    # same clock measured at slightly the wrong phase, and counting them as rivals would call every
    # determined clock ambiguous.
    others = filter(t -> abs(t[1] - best[1]) >= 1.0, counts)
    runner = isempty(others) ? (0.0, 0) : argmax(t -> t[2], others)
    # Determined when the winner matches the great majority and nothing a line away comes close. Both
    # halves are needed: the first alone would accept a broad plateau, and the second alone would accept
    # a scan where nothing matches. The winner's own bar is 0.9 rather than 0.99 because of the azimuth
    # residual above — 96.2% on `S1A_IW_SLC__1SSH_20151120`, where the nearest rival a full line away
    # takes 8.6%, so the two are separated by an order of magnitude and no threshold between them is
    # delicate.
    unique = best[2] > 0.9 * length(idx) && runner[2] < 0.2 * length(idx)
    return ClockSolution(p.aztime + best[1] / p.prf, best[1], best[2], length(idx),
                         sort(counts; by = t -> -t[2])[1:min(end, 5)], unique)
end

"""
    main_clock(args) -> Bool

Solve and report the clock for every radar run named by `args`, without running the geometry.

The rung the rest of the radar layer depends on, so it runs on its own: until the clock is determined
the azimuth index means nothing, and a band comparison against it would be reporting the scan's failure
as a kernel difference.
"""
function main_clock(args)
    name = nothing
    for a in args
        startswith(a, "--") || (name = a)
    end
    runs = filter(r -> r.radar, goldenruns(; name))
    isempty(runs) && error("no radar golden run on disk" *
                           (name === nothing ? "" : " matching \"$name\""))
    allok = true
    for r in runs
        s = solve_sensing_start(r)
        allok &= s.unique
        @printf("%-34s %s  sensing_start %.6f  (log %.6g %+.3f lines)  %d of %d\n",
                short_name(r), s.unique ? "determined" : "AMBIGUOUS ", s.sensing_start,
                r.radar_params.aztime, s.lines, s.matched, s.total)
        @printf("%-34s   top offsets: %s\n", "",
                join((@sprintf("%+.3f:%d", dl, hit) for (dl, hit) in s.neighbours), "  "))
    end
    println()
    println(allok ? "every radar run's clock is determined by its own azimuth indices" :
            "at least one clock is not determined; the bands derived from it cannot be gated")
    return allok
end



# ---------------------------------------------------------------------------
# The bands
# ---------------------------------------------------------------------------

"""
Bounds for the radar float bands, from `REFERENCE.md`'s real-data table.

Three groups, because they divide by different things and inherit different amounts of the azimuth
residual:

  * bands dividing by the range sample spacing (`off2v*_dx`) inherit none of it;
  * the scale factors inherit a little;
  * bands dividing by the along-track step (`off2vx_dy`, `off2vy_dy`, `off2v*_dr`) inherit it directly,
    because that step is measured between two solved ground points.

The along-track bound is the loosest for exactly that reason. `REFERENCE.md` measures the reference's
own `da` error at 1.07e-4 maximum relative and attributes the band error to it to five digits, so this
is a property of the reference's azimuth time rather than slack in the kernel.
"""
const RADAR_BOUNDS = Dict(:off2vx_dx => 1e-6, :off2vy_dx => 1e-6,
                          :scale_x => 1e-7, :scale_y => 1e-6,
                          :off2vx_dy => 1e-3, :off2vy_dy => 1e-3,
                          :off2vx_dr => 1e-3, :off2vy_dr => 1e-3)

"""
Points per index band allowed to differ by one on the radar path, as a fraction of the points the
reference computed.

Applied to `location_x` and `location_y` and to the extents derived from them. Both indices come from
the same range-Doppler solve, so both carry the gap between the compiled kernel's floating-point history
and any external reproduction of it — the ~0.0013-line azimuth offset `REFERENCE.md` documents, and its
range counterpart. Every point whose unrounded index lands within that distance of a `std::round`
boundary rounds the other way.

Set from measurement on this data rather than from the fixture bound. `REFERENCE.md` reports the
azimuth reach as ≤ 0.3% of points against a reference this repository runs itself, with the range index
bitwise. Neither holds against a delivered product, and the reason is the log rather than the kernel:
`dr` prints as `2.32956` — six figures, so ±5e-6, which is 0.14 index units at the far edge of a 66,000
sample swath. The range index inherits that directly and cannot be bitwise while `dr` is read from the
log.

Fitting `dr` was tried and rejected. Scanning it lifts the range agreement from 86.0% to a peak of
94.6%, but the peak sits at 2.329569, which renders as `2.32957` — outside the interval the log's own
`2.32956` admits. A fitted value that contradicts the printed one is absorbing some other error, so it
is not a refinement and the printed value stands.

So the bound is set at 15% of the computed points, which is what this evidence establishes: the range
index agrees on 86.4% of 6.7 million points and the azimuth on 97.7%, with *every* disagreement on
either exactly one. That combination — a large count, none of it larger than one index — is the
signature of a correct solve read through a coarsely printed sample spacing, and it is what the gate
checks. Tightening it needs `dr` to more digits, which means the SAFE annotation rather than the log.
"""
const RADAR_INDEX_FRACTION = 0.15

"""
    radar_result(r::GoldenRun, sensing_start; ntasks) -> PairGeometry

Run `r`'s geometry at the given clock, over the run's own window and inputs.
"""
function radar_result(r::GoldenRun, sensing_start::Real;
                      ntasks::Integer = max(1, Threads.nthreads()))
    paths = fetch_params(r)
    check_params(r, paths)

    dem = Raster(paths["dem"]; lazy = true, missingval = nothing)
    ff = mapgrid(dem)
    grid = MapGrid(geotransform = ff.geotransform, size = ff.size, crs = r.epsg)
    coord = radar_coordinate(r, sensing_start)
    # `testGeogrid.py:427-470` takes every radar parameter from image 1 and the secondary only for the
    # interval, so the pair is the reference coordinate plus `dt`. There is no radar `coregister`.
    pair = CoregisteredPair(coord; dt = r.dt)

    win = CartesianIndices((1:ff.size[1], 1:ff.size[2]))
    lazy(f) = haskey(paths, f) ? Raster(paths[f]; lazy = true, missingval = nothing) : nothing
    src = RasterInputs(dem = dem, dhdx = lazy("dhdx"), dhdy = lazy("dhdy"),
                       vx = lazy("vx"), vy = lazy("vy"), srx = lazy("srx"), sry = lazy("sry"),
                       csminx = lazy("csminx"), csminy = lazy("csminy"),
                       csmaxx = lazy("csmaxx"), csmaxy = lazy("csmaxy"), ssm = lazy("ssm"))

    return pairgeometry_blocked(grid, pair, src;
                                transform = () -> fast_transform(r.epsg, 4326),
                                window = win, ntasks,
                                params = GeometryParams(chip_size_0 = r.chip_size_0),
                                nodata = nodata_from(r.nodata))
end

"""
    check_radar(r::GoldenRun; ntasks) -> (Vector{BandResult}, ClockSolution)

Run `r`'s geogrid output against this package's, at the clock its own azimuth indices determine.

Returns the clock alongside the bands, because the bands are conditional on it: a run whose clock is not
determined gets no band comparison, since the azimuth index would then be reporting the scan's failure.
"""
function check_radar(r::GoldenRun; ntasks::Integer = max(1, Threads.nthreads()))
    clock = solve_sensing_start(r)
    clock.unique || return (BandResult[], clock)

    result = radar_result(r, clock.sensing_start; ntasks)
    out = BandResult[]

    # The azimuth-derived integer bands carry the residual; the rest do not. The allowance is a fraction
    # of the points the reference *computed*, not of the grid: most of a radar scene's grid is outside
    # the swath, and a fraction of the whole would be an allowance many times the documented reach.
    ncomputed = count(!=(Int32(r.nodata)), result.location_x)
    index_allow = ceil(Int, RADAR_INDEX_FRACTION * ncomputed)
    for (file, fields) in reference_files(result.coordinate)
        path = joinpath(r.dir, file)
        isfile(path) || continue
        ArchGDAL.read(path) do ds
            theirs = [ArchGDAL.read(ds, b) for b in 1:ArchGDAL.nraster(ds)]
            scale = _operator_scale(file, theirs, result)
            for (b, f) in enumerate(fields)
                ours = getfield(result, f)
                if eltype(ours) <: Integer
                    # Both index bands and the extents derived from them; the chip sizes and the mask
                    # come from the parameter rasters rather than the solve and are held bitwise.
                    allow = f in (:location_x, :location_y, :offset_x, :offset_y,
                                  :search_x, :search_y) ? index_allow : 0
                    push!(out, compare_int_band(file, b, f, ours, theirs[b];
                                                allow_boundary = allow,
                                                sentinel = Int(r.nodata)))
                else
                    push!(out, compare_float_band(file, b, f, ours, theirs[b];
                                                  bound = RADAR_BOUNDS[f], scale,
                                                  sentinel = r.nodata))
                end
            end
        end
    end
    # Coverage, with the same allowance the index bands get and for the same reason: a grid point whose
    # index lands within the six-figure noise of the image's own edge is in the swath on one side and out
    # of it on the other. Measured at 14 points of 6,678,195 on `S1A_IW_SLC__1SSH_20151120` — four orders
    # below the bound, so this is a statement that the footprint agrees rather than a tolerance doing work.
    cov = compare_coverage(r, result)
    pushfirst!(out, BandResult(cov.file, cov.band, cov.field, "counts within one part in 10,000",
                               cov.ndiff <= max(4, ceil(Int, 1e-4 * ncomputed)), cov.n, cov.ndiff,
                               cov.worst, cov.detail))

    # The scene-center incidence angle the log prints, against the one `incidence_angle` computes from
    # the same geometry. Printed to six figures, so that is the comparison's precision — and it is an
    # independent check on the orbit, the range and the clock all at once, since all three feed it.
    ours_inc = rad2deg(result.coordinate.incidence_angle)
    theirs_inc = r.radar_params.incidence_deg
    # One printed digit of tolerance, not zero. The log's value is itself rounded to six figures, and the
    # angle is computed from `starting_range` and `dr` — both read from that same six-figure printing —
    # so the last digit cannot be expected to agree. A whole unit in it is 1e-4 degrees, which at this
    # scene's slant range is millimetres of ground position, and a real error in the orbit or the range
    # would move the angle by far more than one digit.
    inc_ok = abs(ours_inc - theirs_inc) <= 1.5e-4
    push!(out, BandResult("incidence angle", 1, :incidence, "within one printed digit", inc_ok, 1,
                          inc_ok ? 0 : 1, abs(ours_inc - theirs_inc),
                          @sprintf("ours %.6g against the log's %.6g, differing by %.1e deg",
                                   ours_inc, theirs_inc, abs(ours_inc - theirs_inc))))
    return (out, clock)
end

"""
    main_radar(args) -> Bool

Compare every radar run named by `args`, or solve the clocks alone with `--clock`.
"""
function main_radar(args)
    "--clock" in args && return main_clock(args)
    name = nothing
    for a in args
        startswith(a, "--") || (name = a)
    end
    runs = filter(r -> r.radar, goldenruns(; name))
    isempty(runs) && error("no radar golden run on disk" *
                           (name === nothing ? "" : " matching \"$name\""))
    allok = true
    for r in runs
        rs, clock = check_radar(r)
        if isempty(rs)
            @printf("\n=== %s run %d  — clock not determined (%d of %d at %+.3f lines)\n",
                    short_name(r), r.run, clock.matched, clock.total, clock.lines)
            allok = false
            continue
        end
        @printf("\n=== %s run %d  [EPSG %d, clock %+.3f lines from the log, %d of %d]\n",
                short_name(r), r.run, r.epsg, clock.lines, clock.matched, clock.total)
        allok &= report(rs)
    end
    println()
    println(allok ? "every band of every radar run agrees within its gate" :
            "at least one band is outside its gate")
    return allok
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main_radar(ARGS) ? 0 : 1)
end
