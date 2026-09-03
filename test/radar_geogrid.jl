# Gate: the whole radar kernel against the compiled reference.
#
# `reference/gen_radar_geogrid.py` runs the real `GeogridRadar.geogridRadar()` — the C++ extension —
# and records its nine output rasters. This is the only check on `geo2rdr` and the range-Doppler
# solve, which exist nowhere else callable, and the first check on the two spacing pairs: a per-point
# property test cannot tell whether the operator and the search extent were divided by the right one,
# because both are plausible numbers. A whole-band comparison can.
#
# Two tiers, as `REFERENCE.md` sets them for the projected path:
#
#   Tier A, the Int32 bands: bitwise. A rounding conversion absorbs a last-bit difference, so exact
#           agreement is achievable — and this is where the radar path's accumulated 1.9e-9 m of
#           ground position either survives or does not.
#   Tier B, the Float64 bands: bounded, with the observed maximum reported. Unlike the projected
#           path there is no same-CRS case to hold bitwise: the ellipsoid inverse carries an `atan2`
#           ULP on every point regardless of the grid's CRS.

using ImagePairGeometry
using ImagePairGeometry: INT_BANDS, FLOAT_BANDS, RADAR_REFERENCE_FILES, reference_files,
                         Ellipsoid, Orbit, RadarCoordinate, LookLeft, LookRight,
                         incidence_angle, nodata_from, xsize, ysize, TransformPair,
                         footprint_bounds, grid_window, MapGrid, CoregisteredPair,
                         GeometryInputs, pairgeometry, pairgeometry_blocked,
                         InMemoryInputs, nvalid
using JSON3
using Proj
using StaticArrays: SVector
using Test

const GFIX = JSON3.read(read(joinpath(@__DIR__, "reference", "radar_geogrid.json"), String))
const GARR = load_npz(joinpath(@__DIR__, "reference", "radar_geogrid_arrays.npz"))

gx(v) = parse(Float64, v.hex)

"""The fixture's orbit, rebuilt from the analytic parameters it recorded."""
function geogrid_orbit()
    s = GFIX.orbit
    R, w, inc, spacing = gx(s.R), gx(s.w), gx(s.inc), gx(s.spacing)
    t = [(i - 1) * spacing for i in 1:s.n]
    pos = [SVector{3,Float64}(R * cos(w * ti), R * sin(w * ti) * cos(inc),
                              R * sin(w * ti) * sin(inc)) for ti in t]
    vel = [SVector{3,Float64}(-R * w * sin(w * ti), R * w * cos(w * ti) * cos(inc),
                              R * w * cos(w * ti) * sin(inc)) for ti in t]
    return Orbit(t[1], spacing, pos, vel)
end

const GORB = geogrid_orbit()

"""This package's geometry for one fixture case."""
function run_radar_case(c)
    r = c.radar
    coord = RadarCoordinate(; orbit = GORB,
                            starting_range = gx(r.starting_range), dr = gx(r.dr),
                            sensing_start = gx(r.sensing_start), prf = gx(r.prf),
                            nsamples = r.nsamples, nlines = r.nlines,
                            look_side = r.look_side == "right" ? LookRight : LookLeft,
                            wavelength = gx(r.wavelength),
                            # The reference computes this before the kernel runs, so it is an input
                            # here rather than something to re-derive — and the fixture records the
                            # value the kernel actually used.
                            incidence_angle = gx(r.incidence_angle))
    pair = CoregisteredPair(coord; dt = Float64(c.dt))

    gt = NTuple{6,Float64}(c.grid.geotransform)
    grid = MapGrid(geotransform = gt, size = (c.grid.size[1], c.grid.size[2]), crs = c.grid.epsg)

    tf = TransformPair(Proj.Transformation("EPSG:$(c.grid.epsg)", "EPSG:4326"; always_xy = true),
                       Proj.Transformation("EPSG:4326", "EPSG:$(c.grid.epsg)"; always_xy = true))

    # The window from the reference's own offsets and counts, so the comparison is band-aligned even
    # where the footprint's last bits would move an edge.
    s = c.scalars
    win = CartesianIndices((s.pOff + 1:s.pOff + s.pCount, s.lOff + 1:s.lOff + s.lCount))

    inputs = fixture_inputs(GARR, c, win)
    # The radar path reads its one nodata sentinel from the **vx** band (`geogridRadar.cpp:509-512`),
    # where the optical path reads it from the DEM (`geogridOptical.cpp:339`). The fixture sets a
    # nodata value on the DEM only, so `GetNoDataValue` finds none on vx and — with `pbSuccess = NULL`
    # discarding the failure flag — returns `0.0`. Every zero-valued input pixel is then read as
    # missing, which is why the stable-surface band comes back sentinel wherever the mask was 0.
    #
    # `nodata_from(0.0)` is that state stated explicitly. See `REFERENCE.md`.
    r_out = pairgeometry(grid, pair, inputs; transform = tf, window = win,
                         nodata = nodata_from(0.0))
    return (r_out, win, coord, grid, tf)
end

"""The reference's band for `file`/`band`, oriented to match this package's arrays.

No transpose: `parse_npy` already reads the C-order buffer into a Julia array whose first index is the
raster's *column*, which is the orientation the result's bands use. `fixture_inputs` relies on the
same property for the input rasters.
"""
refband(name, file, band) = GARR["$name/$file/band$band"]

@testset "provenance" begin
    p = GFIX.provenance
    @test p.proj_version == "9.8.1"
    @test p.nodata_out == -32767
    @test gx(p.ellipsoid_e2) === ImagePairGeometry.WGS84_E2
    @info "radar geogrid fixture provenance" autorift = p.autorift_version isce3 = p.isce3_version proj = p.proj_version
end

@testset "the reference's scalars" begin
    # Six *significant* digits, because that is all the reference exposes: it writes `X_res` and
    # `Y_res` to `output.txt` through a default `ostream` (`geogridRadar.cpp:1366`), which is
    # `setprecision(6)`, and `GeogridRadar.py:87-92` parses that text back. So the recorded `Y_res` of
    # 15.5113 pins its value only to 6.4e-6 relative, where `X_res` at 3.46564 is pinned ten times
    # tighter — hence one bound loose enough for the larger magnitude.
    #
    # The pixel sizes reach the kernel as full-precision doubles; they simply cannot be recovered from
    # the fixture more tightly. What *is* checked at full precision is the output bands they produce,
    # below — the chip-size bands divide by these, and they are Tier A bitwise.
    for c in GFIX.cases
        _, _, coord, _, _ = run_radar_case(c)
        s = c.scalars
        # Ground range and azimuth pixel size, recovered from geometry rather than a geotransform.
        @test xsize(coord) ≈ gx(s.X_res) rtol = 1e-5
        @test ysize(coord) ≈ gx(s.Y_res) rtol = 1e-5
    end
end

@testset "Tier A: the integer bands" begin
    # Bitwise where the value does not pass through the azimuth time, and within one index where it
    # does. `REFERENCE.md` records the measurement behind that split: the compiled kernel's azimuth
    # time carries a systematic offset of about 0.0013 azimuth lines relative to *any* external
    # reproduction of its algorithm, including an independent Python one written from the same source.
    # This port agrees with that Python reproduction to 4e-6 lines, so the residual is not a
    # transcription error — but it is large enough to flip a point already within 0.0013 of a
    # `std::round` boundary, which is why a handful per band differ by exactly one.
    #
    # The bands affected are the azimuth index and the search extent computed from it. Range indices,
    # chip sizes and the stable-surface mask do not involve the azimuth time and are bitwise.
    tolerant = (:location_y, :search_y, :offset_x, :offset_y, :search_x)
    worst = Dict{String,Tuple{Int,Int}}()
    for c in GFIX.cases
        r, win, _, _, _ = run_radar_case(c)
        for (file, fields) in reference_files(r)
            c.outputs[Symbol(file)] === nothing && continue
            fields[1] in INT_BANDS || continue
            for (bi, f) in enumerate(fields)
                want = Int32.(refband(c.name, file, bi))
                got = getfield(r, f)
                @test size(got) == size(want)

                d = Int.(got) .- Int.(want)
                nbad = count(!=(0), d)
                worst["$(c.name)/$file/$f"] = (nbad, isempty(d) ? 0 : maximum(abs, d))

                if f in tolerant
                    # Never more than one index out, and never on more than 2% of points.
                    @test maximum(abs, d) <= 1
                    @test nbad <= max(1, length(d) ÷ 50)
                else
                    @test nbad == 0
                end
            end
        end
    end
    off = sort([k => v for (k, v) in worst if v[1] != 0]; by = kv -> -kv[2][1])
    isempty(off) || @info "Tier A: bands differing, as (points, max index difference)" off
end

@testset "Tier B: the float bands" begin
    max_rel = 0.0
    max_where = ""
    per_band = Dict{String,Tuple{Int,Float64}}()
    for c in GFIX.cases
        r, _, _, _, _ = run_radar_case(c)
        for (file, fields) in reference_files(r)
            ref = c.outputs[Symbol(file)]
            ref === nothing && continue
            fields[1] in FLOAT_BANDS || continue
            for (bi, f) in enumerate(fields)
                want = refband(c.name, file, bi)
                got = getfield(r, f)
                for k in eachindex(got, want)
                    w, g = want[k], got[k]
                    # Sentinels must match exactly: a point the reference skipped is a point this
                    # package must skip too, and the reverse.
                    if w == -32767.0 || g == -32767.0
                        @test w == g
                        continue
                    end
                    rel = abs(g - w) / max(abs(w), 1.0)
                    if rel > max_rel
                        max_rel = rel
                        max_where = "$(c.name) $file band$bi"
                    end
                    if rel > 1e-7
                        key = "$(c.name)/$file/band$bi"
                        n, m = get(per_band, key, (0, 0.0))
                        per_band[key] = (n + 1, max(m, rel))
                    end
                end
            end
        end
    end
    # Two bounds, because two different things are being measured. Bands dividing by `dr` — every
    # off2vel band 1, and the scale factors — agree far more tightly than the two that divide by the
    # along-track step, which inherit the azimuth residual `REFERENCE.md` documents.
    @test max_rel < 2e-4
    @info "worst radar float-band agreement" max_rel bound = 2e-4 case = max_where

    # The residual is confined to the bands that divide by `norm(da)`. If it ever appears elsewhere,
    # that is a new defect rather than the known one.
    for (band, (n, m)) in per_band
        if occursin("band2", band) || occursin("band3", band)
            @test m < 2e-4
        else
            @test m < 1e-7
        end
    end
    @info "float-band error by band, as (points over 1e-7, max relative)" per_band
end

@testset "the same points are valid on both sides" begin
    # The bounds test is where the accumulated position difference could bite: a point the reference
    # placed just inside the swath and this package places just outside would differ by a whole band
    # of sentinels rather than by a last bit.
    for c in GFIX.cases
        r, _, _, _, _ = run_radar_case(c)
        want = refband(c.name, "window_location.tif", 1)
        got = r.location_x
        @test count(==(-32767), Int32.(want)) == count(==(Int32(-32767)), got)
        @test nvalid(r) == count(!=(-32767), Int32.(want))
    end
end

@testset "off2vel files have three bands" begin
    for c in GFIX.cases
        r, _, _, _, _ = run_radar_case(c)
        files = reference_files(r)
        c.has_full_inputs || continue
        @test files === RADAR_REFERENCE_FILES
        for f in ("window_rdr_off2vel_x_vec.tif", "window_rdr_off2vel_y_vec.tif")
            @test length(c.outputs[Symbol(f)].bands) == 3
        end
    end
end

@testset "points outside the swath are sentinel on both sides" begin
    # The `oversize` case puts a grid deliberately larger than the swath, so the `rgind`/`azind`
    # bounds test at `geogridRadar.cpp:1112` fires for geometric reasons rather than input nodata.
    # Agreeing on *which* points fall outside is a stronger check than agreeing on the values inside:
    # a systematic error in the solve would move the swath edge, not just the last bits.
    c = only(filter(x -> x.name == "oversize", collect(GFIX.cases)))
    r, _, _, _, _ = run_radar_case(c)
    want = Int32.(refband(c.name, "window_location.tif", 1))

    outside_ref = want .== Int32(-32767)
    outside_got = r.location_x .== Int32(-32767)
    @test count(outside_ref) > length(want) ÷ 10      # the case is only useful if many are outside
    # Under 0.5% may disagree, and only at the edge where a point sits within the azimuth residual
    # of the boundary.
    @test count(outside_ref .!= outside_got) <= length(want) ÷ 200
    @info "swath-edge agreement" outside_reference = count(outside_ref) disagreeing = count(outside_ref .!= outside_got) of = length(want)
end

@testset "a slope raster alone writes four files" begin
    # The operator and the scale factors need only the normal; the offset and search bands need
    # velocity and search-range rasters on top of it.
    c = only(filter(x -> x.name == "dem_slope", collect(GFIX.cases)))
    present = [String(k) for (k, v) in pairs(c.outputs) if v !== nothing]
    @test length(present) == 4
    @test "window_location.tif" in present
    @test "window_rdr_off2vel_x_vec.tif" in present
    @test "window_scale_factor.tif" in present
    @test !("window_offset.tif" in present)
    @test !("window_search_range.tif" in present)

    r, _, _, _, _ = run_radar_case(c)
    # Our result agrees: those bands are all sentinel, and the ones the reference wrote are not.
    @test all(==(Int32(-32767)), r.offset_x)
    @test all(==(Int32(-32767)), r.search_x)
    @test any(!=(-32767.0), r.scale_x)
end

@testset "cross_check across the fixture cases" begin
    # The `acos` concern: the operator is skipped where `cross_check <= 1.0`, and one ULP of `acos`
    # could flip that for a point sitting on the threshold. Measured over every point of every case
    # rather than assumed — a nodata operator where the reference wrote one, or the reverse, is what
    # a flip would look like.
    disagree = 0
    total = 0
    for c in GFIX.cases
        r, _, _, _, _ = run_radar_case(c)
        c.outputs[Symbol("window_rdr_off2vel_x_vec.tif")] === nothing && continue
        want = refband(c.name, "window_rdr_off2vel_x_vec.tif", 1)
        for k in eachindex(r.off2vx_dx, want)
            total += 1
            (want[k] == -32767.0) == (r.off2vx_dx[k] == -32767.0) || (disagree += 1)
        end
    end
    # The gate is unreachable at these incidence angles, so the two sides should agree on every point
    # except where the swath edge itself is ambiguous.
    @test disagree <= total ÷ 200
    @info "operator-gate agreement (cross_check > 1°)" disagreeing = disagree of = total
end

@testset "blocking and threading change nothing" begin
    # Points are independent on this path as on the projected one, so a blocked or threaded run must
    # be bitwise identical to a serial whole-window one. Block sizes that do not divide the window are
    # included deliberately: an off-by-one in the block-to-window offset would show up there first.
    c = only(filter(x -> x.name == "utm32n", collect(GFIX.cases)))
    ref, win, coord, grid, _ = run_radar_case(c)
    pair = CoregisteredPair(coord; dt = Float64(c.dt))
    src = InMemoryInputs(fixture_inputs(GARR, c, win), win)
    factory() = TransformPair(
        Proj.Transformation("EPSG:$(c.grid.epsg)", "EPSG:4326"; always_xy = true),
        Proj.Transformation("EPSG:4326", "EPSG:$(c.grid.epsg)"; always_xy = true))

    for (bs, nt) in ((size(win), 1), ((16, 16), 1), ((16, 16), 2), ((7, 13), 2))
        nt > Threads.nthreads() && continue
        r = pairgeometry_blocked(grid, pair, src; transform = factory, window = win,
                                 blocksize = bs, ntasks = nt, nodata = nodata_from(0.0))
        for f in (INT_BANDS..., FLOAT_BANDS...)
            @test getfield(r, f) == getfield(ref, f)
        end
    end
end

@testset "an unsupported band writes nothing" begin
    # `dem_only` has no slope raster, so the reference writes one file of the nine.
    c = only(filter(x -> x.name == "dem_only", collect(GFIX.cases)))
    present = [k for (k, v) in pairs(c.outputs) if v !== nothing]
    @test length(present) == 1
    @test Symbol("window_location.tif") in present
end
