# `RadarCoordinate` and the two scene-level quantities derived from it, against the reference.
#
# `reference/gen_radar_coordinate.py` runs `GeogridRadar.determineBbox` and
# `GeogridRadar.getIncidenceAngle` as the reference defines them — the same `isce3.geometry.rdr2geo`
# calls in the same order over the same sample grid — and records the bounding box and the angle.
#
# The bound here is the one `REFERENCE.md` establishes for `rdr2geo`: 1.9e-9 m of ground position,
# carried by the `atan2` ULP and the Hermite velocity contraction. A bounding box is a min/max over
# 160 solves, so it inherits that and no more — the box cannot be bitwise while its corners are not,
# and the quantity that actually has to be exact is the *grid window* the box produces, which is
# asserted separately below.

using ImagePairGeometry
using ImagePairGeometry: Ellipsoid, Orbit, RadarCoordinate, LookLeft, LookRight,
                         incidence_angle, footprint_bounds, grid_window, MapGrid,
                         nsamples, nlines, xsize, ysize, midtime, orbit_midtime, sensing_stop,
                         chip_size_pixels, interpolate, norm3, CoregisteredPair,
                         AbstractImageCoordinate, ProjectedCoordinate, ImageFootprint, coregister,
                         FOOTPRINT_RANGE_SAMPLES, FOOTPRINT_AZIMUTH_SAMPLES
using Extents: Extent
using JSON3
using StaticArrays: SVector
using Test

const CFIX = JSON3.read(read(joinpath(@__DIR__, "reference", "radar_coordinate.json"), String))

cx(v) = parse(Float64, v.hex)

"""The fixture's orbit, rebuilt from the analytic parameters it recorded."""
function coordinate_orbit()
    s = CFIX.orbit
    n, spacing, t0 = s.n, cx(s.spacing), cx(s.t0)
    R, w, inc = cx(s.R), cx(s.w), cx(s.inc)
    t = [t0 + (i - 1) * spacing for i in 1:n]
    pos = [SVector{3,Float64}(R * cos(w * ti), R * sin(w * ti) * cos(inc),
                              R * sin(w * ti) * sin(inc)) for ti in t]
    vel = [SVector{3,Float64}(-R * w * sin(w * ti), R * w * cos(w * ti) * cos(inc),
                              R * w * cos(w * ti) * sin(inc)) for ti in t]
    return Orbit(t0, spacing, pos, vel)
end

const CORB = coordinate_orbit()
const R = CFIX.radar

"""The fixture's radar parameters as keyword arguments."""
radar_kwargs() = (starting_range = cx(R.starting_range), dr = cx(R.dr),
                  sensing_start = cx(R.sensing_start), prf = cx(R.prf),
                  nsamples = R.nsamples, nlines = R.nlines,
                  look_side = R.look_side == "right" ? LookRight : LookLeft,
                  wavelength = cx(R.wavelength))

const IA = incidence_angle(; orbit = CORB, radar_kwargs()...)
const RC = RadarCoordinate(; orbit = CORB, incidence_angle = IA, radar_kwargs()...)

@testset "provenance" begin
    @info "radar coordinate fixture provenance" isce3 = CFIX.provenance.isce3_version
    # The reference samples 80 positions at two elevations each.
    @test CFIX.bbox.npoints == 160
    @test 2 * (2 * FOOTPRINT_RANGE_SAMPLES + 2 * (FOOTPRINT_AZIMUTH_SAMPLES - 2)) == 160
end

@testset "incidence angle matches the reference" begin
    # Nanoradian agreement: this is two `rdr2geo` solves and an `acos`, so it carries the same
    # ground-position bound and nothing more.
    @test IA ≈ cx(CFIX.incidence_angle) rtol = 1e-12
    # A side-looking radar sits well away from both nadir and the horizon.
    @test deg2rad(20) < IA < deg2rad(70)
end

@testset "scene-center sampling is one sample short, as the reference computes it" begin
    # `floor(n / 2) - 1`, not `n / 2` — reproduced rather than corrected.
    kw = radar_kwargs()
    @test kw.starting_range + (floor(kw.nsamples / 2) - 1) * kw.dr === cx(CFIX.midrange)
    @test kw.sensing_start + (floor(kw.nlines / 2) - 1) / kw.prf === cx(CFIX.midsensing)
end

@testset "footprint bounds match the reference" begin
    # The transform is the identity on (lon, lat, h) in degrees, matching the fixture, which uses a
    # lon/lat "projection" so the comparison isolates the radar solve from PROJ.
    ext = footprint_bounds((x, y, z) -> (x, y, z), RC)

    @test ext.X[1] ≈ cx(CFIX.bbox.lon[1]) rtol = 1e-12
    @test ext.X[2] ≈ cx(CFIX.bbox.lon[2]) rtol = 1e-12
    @test ext.Y[1] ≈ cx(CFIX.bbox.lat[1]) rtol = 1e-12
    @test ext.Y[2] ≈ cx(CFIX.bbox.lat[2]) rtol = 1e-12

    # Ordered, and non-degenerate.
    @test ext.X[1] < ext.X[2]
    @test ext.Y[1] < ext.Y[2]
end

@testset "the grid window is exact despite the box being bounded" begin
    # What the box is *for*. The corners agree to 1e-12 relative, so a grid whose spacing is far
    # larger than that produces a bitwise-identical window — which is the quantity a shift would
    # corrupt, and the same standard `REFERENCE.md` holds the projected path's footprint to.
    ext = footprint_bounds((x, y, z) -> (x, y, z), RC)
    ref = (lon = (cx(CFIX.bbox.lon[1]), cx(CFIX.bbox.lon[2])),
           lat = (cx(CFIX.bbox.lat[1]), cx(CFIX.bbox.lat[2])))

    # A degree-spaced grid covering the swath, in the same lon/lat space.
    for spacing in (0.01, 0.05, 0.1)
        grid = MapGrid(geotransform = (0.0, spacing, 0.0, 25.0, 0.0, -spacing),
                       size = (400, 800))
        ours = grid_window(grid, ext)
        theirs = grid_window(grid, Extent(X = ref.lon, Y = ref.lat))
        @test ours == theirs
    end
end

@testset "pixel sizes come from geometry, not a geotransform" begin
    # `grd_res = dr / sin(incidence)`, `azm_res = |v| / prf` (`geogridRadar.cpp:684-686`).
    @test xsize(RC) === cx(R.dr) / sin(IA)

    _, vel = interpolate(CORB, orbit_midtime(RC))
    @test ysize(RC) === norm3(vel) / cx(R.prf)

    # Ground range exceeds slant range spacing, since the beam is oblique.
    @test xsize(RC) > cx(R.dr)
    # Both are plausible SAR pixel sizes.
    @test 1.0 < xsize(RC) < 100.0
    @test 1.0 < ysize(RC) < 100.0
end

@testset "chip size conversion consumes the radar sizes unchanged" begin
    # The point of `xsize`/`ysize` being the interface: the projected path's conversion works on a
    # `RadarCoordinate` with no radar-specific code.
    @test chip_size_pixels(240.0, xsize(RC)) % 4 == 0
    @test chip_size_pixels(240.0, ysize(RC)) % 4 == 0
    @test chip_size_pixels(240.0, xsize(RC)) > chip_size_pixels(240.0, ysize(RC))
end

@testset "the two clocks differ by one pulse interval" begin
    # Not a defect: `midtime` is `geogridRadar.cpp:328`'s expression and `orbit_midtime` is
    # `GeogridRadar.py:346`'s, and they feed different clocks. For an even line count they differ by
    # exactly one line. See REFERENCE.md.
    @test iseven(R.nlines)
    lines = (midtime(RC) - orbit_midtime(RC)) * cx(R.prf)
    @test lines ≈ 1.0 rtol = 1e-9

    # `sensing_stop` is the last line, one interval short of `nlines / prf`.
    @test sensing_stop(RC) ≈ cx(R.sensing_start) + (R.nlines - 1) / cx(R.prf)
    @test midtime(RC) > cx(R.sensing_start)
    @test midtime(RC) < sensing_stop(RC) + 1 / cx(R.prf)
end

@testset "size accessors" begin
    @test nsamples(RC) == R.nsamples
    @test nlines(RC) == R.nlines
    @test RC isa AbstractImageCoordinate
end

@testset "input validation" begin
    kw = radar_kwargs()
    bad(; over...) = RadarCoordinate(; orbit = CORB, incidence_angle = IA, kw..., over...)

    @test_throws "starting_range must be positive" bad(starting_range = -1.0)
    @test_throws "dr must be positive" bad(dr = 0.0)
    @test_throws "prf must be positive" bad(prf = -10.0)
    @test_throws "size must be positive" bad(nsamples = 0)
    @test_throws "size must be positive" bad(nlines = -5)
    @test_throws "wavelength must be positive" bad(wavelength = 0.0)
    # Zero would divide by zero in `xsize`; a right angle is looking straight down.
    @test_throws "incidence_angle must be in" bad(incidence_angle = 0.0)
    @test_throws "incidence_angle must be in" bad(incidence_angle = pi / 2)
    @test_throws "incidence_angle must be in" bad(incidence_angle = -0.5)
end

@testset "both look sides produce a valid footprint on opposite sides" begin
    left = RadarCoordinate(; orbit = CORB, incidence_angle = IA, radar_kwargs()...,
                           look_side = LookLeft)
    el = footprint_bounds((x, y, z) -> (x, y, z), left)
    er = footprint_bounds((x, y, z) -> (x, y, z), RC)

    @test el.X[1] < el.X[2]
    @test el.Y[1] < el.Y[2]
    # The two swaths sit on opposite sides of the ground track, so their boxes do not coincide.
    @test el.X != er.X || el.Y != er.Y
end

# ------------------------------------------------------------------- CoregisteredPair widening

@testset "CoregisteredPair carries any coordinate" begin
    p = CoregisteredPair(RC; dt = 6 * 86400.0)
    @test p.coordinate === RC
    @test p.dt === 6 * 86400.0
    # Zero because the coordinate *is* the window — there is no offset into a larger image.
    @test p.reference_offset == (0, 0)
    @test p.secondary_offset == (0, 0)
    @test p isa CoregisteredPair{<:RadarCoordinate}
end

@testset "a projected coordinate needs no coregistration either" begin
    # The case the widening is really for: a caller who already sliced the overlap passes a view's
    # coordinate directly, rather than round-tripping through `coregister`.
    c = ProjectedCoordinate(origin = (300000.0, 7800000.0), spacing = (30.0, -30.0),
                            size = (400, 400))
    p = CoregisteredPair(c; dt = 86400.0)
    @test p.coordinate === c
    @test p.reference_offset == (0, 0)
    @test p isa CoregisteredPair{<:ProjectedCoordinate}
end

@testset "coregister still produces the offsets it always did" begin
    # The widening must not have changed the computed path.
    a = ImageFootprint(origin = (100.0, 900.0), spacing = (10.0, -10.0), size = (50, 50))
    b = ImageFootprint(origin = (150.0, 950.0), spacing = (10.0, -10.0), size = (50, 50))
    p = coregister(a, b; dt = 86400.0)
    @test p.coordinate.origin == (150.0, 900.0)
    @test p.coordinate.size == (45, 45)
    @test p.reference_offset == (5, 0)
    @test p.secondary_offset == (0, 5)
    @test p isa CoregisteredPair{<:ProjectedCoordinate}
end
