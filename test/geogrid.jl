# Gate: the whole kernel against the compiled reference.
#
# `reference/gen_geogrid.py` runs the real `GeogridOptical.runGeogrid()` — the C++ extension — and
# records its nine output rasters. Two tiers, per REFERENCE.md:
#
#   Tier A, the Int32 bands: bitwise, in every case. A rounding or truncating conversion absorbs
#           any last-bit difference, so exact agreement is achievable and required — and it holds on
#           every platform, which is what establishes that the kernel arithmetic is portable.
#   Tier B, the Float64 bands: bitwise where the grid and image share a CRS, and bounded in relative
#           error where they do not.
#
# Two separate reasons the floats cannot be bitwise under a reprojection.
#
# Contraction: the reference is compiled with floating-point contraction, so it may evaluate
# `a*b + c` as one `fma` where Julia rounds twice. Checked directly — inserting `fma` into the
# determinant cuts the summed ULP error over a whole case by 62% (1148 to 436). Matching it exactly
# would mean matching its compiler flags.
#
# PROJ is not bit-reproducible across platforms. The same PROJ 9.8.1 returns a projected easting
# differing by 2 ULP on x86-64 Linux and Windows from aarch64 macOS. The kernel then divides a
# difference of two such coordinates by the pixel spacing, and at ITS_LIVE scale `|x|/spacing` is
# around 8e4, so a 2-ULP input difference emerges as ~1e4 ULP in a unit vector — 2e-8 absolute on a
# value near 12. Large in ULP, negligible in magnitude: about 2e-9 relative, which is nanometers
# per pixel of ground distance.
#
# Hence a relative bound rather than a ULP bound for the cross-CRS cases. A ULP count is the right
# metric for a value both sides compute from identical inputs, and the wrong one when the inputs
# themselves differ in their last bits. The exactness that does the real work is Tier A plus the
# same-CRS bitwise floats.

using ImagePairGeometry
using ImagePairGeometry: INT_BANDS, FLOAT_BANDS, REFERENCE_FILES, nodata_from, xsize, ysize
using JSON3
using Proj
using Test

const GFIX = JSON3.read(read(joinpath(@__DIR__, "reference", "geogrid.json"), String))
const GARR = load_npz(joinpath(@__DIR__, "reference", "geogrid_arrays.npz"))

"""Reference geometry for one fixture case, computed by this package."""
function run_case(c)
    s = fixture_scene(c)
    tf = s.transform()
    # `footprint_bounds` takes the image-to-grid direction from the pair itself.
    win = grid_window(s.grid, footprint_bounds(tf, s.coord))
    r = pairgeometry(s.grid, s.pair, fixture_inputs(GARR, c, win);
                     transform = tf, window = win, params = s.params,
                     nodata = nodata_from(-32767.0))
    return r, win, s.coord, s.grid, tf
end

"""The reference's band for `file`/`band`, oriented to match this package's arrays."""
refband(name, file, band) = GARR["$name/$file/band$band"]

@testset "provenance" begin
    p = GFIX.provenance
    @test p.proj_version == "9.8.1"
    @test p.nodata_out == -32767
    @info "geogrid fixture provenance" autorift = p.autorift_version gdal = p.gdal_version proj = p.proj_version
end

"""Relative difference between two Float64s, scaled by the larger magnitude. `0` when equal."""
function reldist(a::Float64, b::Float64)
    a === b && return 0.0
    (isnan(a) && isnan(b)) && return 0.0
    (isfinite(a) && isfinite(b)) || return Inf
    return abs(a - b) / max(abs(a), abs(b), 1.0)
end

"""
Largest relative difference tolerated on a float band under a reprojection.

Set well above the measured worst case (~2e-9) and far below anything that could matter: the float
bands convert a pixel displacement to a velocity, so this is nanometers per year against values in
meters per year.
"""
const FLOAT_REL_BOUND = 1e-7

const MAXREL = Ref(0.0)
const RELWORST = Ref(("", "", 0.0))

@testset "$(c.name)" for c in GFIX.cases
    r, win, coord, grid, tf = run_case(c)
    s = c.scalars

    @testset "window and scalars" begin
        @test first(win).I .- 1 == (Int(s.pOff), Int(s.lOff))
        @test size(win) == (Int(s.pCount), Int(s.lCount))
        @test xsize(coord) == parse(Float64, s.X_res_hex)
        @test ysize(coord) == parse(Float64, s.Y_res_hex)
    end

    for (file, fields) in REFERENCE_FILES
        rec = get(c.outputs, Symbol(file), nothing)
        if rec === nothing
            # The reference writes no file when its inputs are absent; every such band must be
            # uniformly nodata here rather than partly filled.
            @testset "$file absent" begin
                for f in fields
                    band = getfield(r, f)
                    @test all(==(eltype(band)(-32767)), band)
                end
            end
            continue
        end

        @test Int(rec.size[1]) == size(r)[1] && Int(rec.size[2]) == size(r)[2]

        for (bi, f) in enumerate(fields)
            got = getfield(r, f)
            want = refband(c.name, file, bi)
            @test size(got) == size(want)

            if eltype(got) === Int32
                # Tier A: bitwise. Compared as Int32 after checking the reference's value is
                # integral, so a non-integral reference value fails loudly instead of rounding.
                @testset "$file band $bi ($f) bitwise" begin
                    @test all(w -> w == trunc(w), want)
                    @test got == Int32.(want)
                end
            elseif Int(c.image.epsg) == Int(c.dem.epsg)
                # Tier B, same CRS: no PROJ call, so the only inputs are the fixture's own numbers
                # and agreement is exact. Contraction cannot bite here because the identity path
                # computes the axis vectors in closed form rather than summing products.
                @testset "$file band $bi ($f) bitwise" begin
                    @test all(reinterpret(UInt64, got) .== reinterpret(UInt64, want))
                end
            else
                # Tier B, reprojected: relative, tracking the worst case across every band and case.
                d = maximum(reldist.(got, want))
                if d > MAXREL[]
                    MAXREL[] = d
                    RELWORST[] = (String(c.name), "$file band $bi ($f)", d)
                end
                @testset "$file band $bi ($f) within tolerance" begin
                    @test d <= FLOAT_REL_BOUND
                end
            end
        end
    end
end

@testset "identity dispatch matches the PROJ noop path" begin
    # For a same-CRS case the reference goes through a PROJ noop pipeline while this package
    # dispatches on `IdentityTransform`. The two must agree bitwise on every band, or the fast path
    # is a different computation rather than a cheaper one.
    c = only(filter(x -> x.name == "same_crs", collect(GFIX.cases)))
    fast, win, coord, grid, _ = run_case(c)

    e = Int(c.image.epsg)
    proj_tf = TransformPair(Proj.Transformation("EPSG:$e", "EPSG:$e"),
                            Proj.Transformation("EPSG:$e", "EPSG:$e"))
    arr(kind) = GARR["$(c.name)/input/$kind"][win]
    inputs = GeometryInputs(dem = arr("dem"), dhdx = arr("dhdx"), dhdy = arr("dhdy"),
                            vx = arr("vx"), vy = arr("vy"), srx = arr("srx"), sry = arr("sry"),
                            csminx = arr("csminx"), csminy = arr("csminy"),
                            csmaxx = arr("csmaxx"), csmaxy = arr("csmaxy"), ssm = arr("ssm"))
    fp = ImageFootprint(origin = coord.origin, spacing = coord.spacing, size = coord.size)
    pair = coregister(fp, fp; dt = Float64(c.dt))
    slow = pairgeometry(grid, pair, inputs; transform = proj_tf, window = win,
                        nodata = nodata_from(-32767.0))

    for f in INT_BANDS
        @test getfield(fast, f) == getfield(slow, f)
    end
    for f in FLOAT_BANDS
        a, b = getfield(fast, f), getfield(slow, f)
        @test all(reinterpret(UInt64, a) .== reinterpret(UInt64, b))
    end
end

@testset "float agreement summary" begin
    # Reported, not just asserted: the margin is the evidence for the bound rather than a number
    # chosen in advance, and it is what would show a platform drifting further than expected.
    @info "worst reprojected float-band agreement" max_rel = MAXREL[] bound = FLOAT_REL_BOUND case = RELWORST[][1] band = RELWORST[][2]
    @test MAXREL[] <= FLOAT_REL_BOUND
    # The bound is not slack to grow into. Same-CRS float bands are asserted bitwise above, so a
    # regression in the arithmetic itself fails there rather than eating into this margin — what
    # remains here is PROJ's platform variation, which is orders of magnitude below the bound.
    @test MAXREL[] >= 0
end
