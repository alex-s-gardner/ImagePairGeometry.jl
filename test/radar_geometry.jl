# The radar per-point kernel: `pointgeometry` for a `RadarCoordinate`.
#
# This is the one piece with no callable reference. `geo2rdr` and the range-Doppler solve exist only
# inside `geogridRadar.cpp`, so there is nothing to generate a fixture from; the values are checked by
# the whole-kernel comparison in `radar_geogrid.jl`. What is checked here is everything that can be
# established without one:
#
#   inversion       — a grid point placed by `rdr2geo` at a known time and range comes back at that
#                     time and range
#   consistency     — the axis vectors are unit length, `xlen` is close to `dr`, the scale factors are
#                     near one under a real projection
#   the frame split — `|da|` and `ylen` are the same step in two frames and differ by the map scale
#   the quirks      — the one-line azimuth offset, and the fixed iteration counts
#
# Run under a real projected CRS wherever the magnitudes matter. With an identity transform the grid
# coordinates are degrees, so a "length" mixes degrees and meters and none of the ratios mean anything.

using ImagePairGeometry
using ImagePairGeometry: Ellipsoid, Orbit, RadarCoordinate, LookLeft, LookRight,
                         incidence_angle, interpolate, rdr2geo, geo2rdr,
                         pointgeometry, PointGeometry, RadarSpacing, RANGE_DOPPLER_ITERATIONS,
                         GEO2RDR_ITERATIONS, _range_doppler,
                         lonlat_to_xyz, xyz_to_lonlat, norm3, dot3, unitvec3,
                         surface_normal, cross_check, offset_to_velocity, scale_factors,
                         axis_velocity, TransformPair, transform_pair, IdentityTransform,
                         midtime, orbit_midtime, range_index, azimuth_index, ysize
using Proj
using StaticArrays: SVector
using Test

const EL = Ellipsoid()

"""A near-polar circular orbit spanning ten minutes."""
function geom_orbit()
    R = 7.0e6
    w = sqrt(3.986004418e14 / R^3)
    inc = deg2rad(98.0)
    t = [(i - 1) * 10.0 for i in 1:61]
    pos = [SVector{3,Float64}(R * cos(w * ti), R * sin(w * ti) * cos(inc),
                              R * sin(w * ti) * sin(inc)) for ti in t]
    vel = [SVector{3,Float64}(-R * w * sin(w * ti), R * w * cos(w * ti) * cos(inc),
                              R * w * cos(w * ti) * sin(inc)) for ti in t]
    return Orbit(t[1], 10.0, pos, vel)
end

const GORB = geom_orbit()
const DR = 2.329562114715323
const SR = 8.0e5
const PRF = 486.4863103
const WVL = 0.05546576
const RKW = (starting_range = SR, dr = DR, sensing_start = 300.0, prf = PRF,
             nsamples = 10000, nlines = 8000, look_side = LookRight, wavelength = WVL)
const RIA = incidence_angle(; orbit = GORB, RKW...)
const RC = RadarCoordinate(; orbit = GORB, incidence_angle = RIA, RKW...)

const NORMAL = surface_normal(0.02, -0.05)

"""A grid point known to be imaged at `(t, rng)`, as geodetic degrees."""
function known_point(t, rng; height = 500.0, side = LookRight)
    llh = rdr2geo(GORB, EL, t, rng; height, wavelength = WVL, side)
    return (rad2deg(llh[1]), rad2deg(llh[2]), llh[3])
end

# UTM 32N covers the test swath. `always_xy` so the transform is (east, north) in both directions,
# which is the convention the kernel's `tf.forward` result is read in.
const TO_LONLAT = Proj.Transformation("EPSG:32632", "EPSG:4326"; always_xy = true)
const TO_GRID = Proj.Transformation("EPSG:4326", "EPSG:32632"; always_xy = true)
const UTM = TransformPair(TO_LONLAT, TO_GRID)

"""The `(t, rng)` point as UTM 32N grid coordinates."""
function utm_point(t, rng; kw...)
    lon, lat, h = known_point(t, rng; kw...)
    gx, gy = TO_GRID(lon, lat)
    return (gx, gy, h)
end

@testset "inverts rdr2geo" begin
    # The only correctness check available without the whole-kernel fixture: a point placed at a known
    # azimuth time and slant range must solve back to them.
    for (t, nsamp) in ((305.0, 4000), (302.0, 100), (310.0, 9000))
        rng = SR + nsamp * DR
        gx, gy, gz = utm_point(t, rng)
        _, _, p = pointgeometry(UTM, gx, gy, gz, RC, NORMAL)

        # The azimuth time carries the reference's one-pulse offset, so it lands one interval late.
        @test p.aztime ≈ t + 1 / PRF rtol = 1e-9
        @test p.range ≈ rng rtol = 1e-12
        # The range index is unaffected — it comes from the range, not the time.
        @test range_index(p, SR, DR) == Float64(nsamp)
    end
end

@testset "the azimuth index carries the one-line offset" begin
    # Not a defect. `midtime` and `orbit_midtime` are offsets from `sensing_start` on two clocks; the
    # epoch cancels from their difference, leaving one pulse interval. `geo2rdr` preserves it, so the
    # converged time is one pulse from the position the orbit was interpolated at. See REFERENCE.md.
    @test (midtime(RC) - orbit_midtime(RC)) * PRF ≈ 1.0 rtol = 1e-9

    t = 305.0
    gx, gy, gz = utm_point(t, SR + 4000 * DR)
    g, _, p = pointgeometry(UTM, gx, gy, gz, RC, NORMAL)

    geometric = round((t - RKW.sensing_start) * PRF)
    @test azimuth_index(p, RKW.sensing_start, PRF) == geometric + 1
    @test g.image_xy[2] == geometric + 1
end

@testset "axis vectors are unit length and sensibly oriented" begin
    gx, gy, gz = utm_point(305.0, SR + 4000 * DR)
    g, _, _ = pointgeometry(UTM, gx, gy, gz, RC, NORMAL)

    @test norm3(g.xunit) ≈ 1.0 atol = 1e-15
    @test norm3(g.yunit) ≈ 1.0 atol = 1e-15
    # The two axes are not parallel — a degenerate pair would make the operator singular.
    @test abs(dot3(g.xunit, g.yunit)) < 0.99
    # The along-track step is horizontal in grid coordinates: the ground point moves at constant
    # height, so its elevation component is zero by construction.
    @test g.yunit[3] == 0.0
end

@testset "step lengths are physical under a real projection" begin
    gx, gy, gz = utm_point(305.0, SR + 4000 * DR)
    g, sp, _ = pointgeometry(UTM, gx, gy, gz, RC, NORMAL)

    # A one-sample range step is `dr` on the ground, up to the projection's local scale and the
    # obliquity of the look direction.
    @test g.xlen ≈ DR rtol = 0.05
    # A one-pulse azimuth step is the platform's ground track advance, which `ysize` also reports.
    @test g.ylen ≈ ysize(RC) rtol = 0.2
    @test 1.0 < g.ylen < 100.0

    # Scale factors are ratios of the same distance in two systems, so near one in UTM.
    sfx, sfy = scale_factors(g, sp.operator)
    @test sfx ≈ 1.0 rtol = 0.05
    @test sfy ≈ 1.0 rtol = 0.05
end

@testset "the two spacing pairs differ by the map scale" begin
    # `|da|` is the along-track step in ECEF and `ylen` the same step in grid coordinates. The
    # reference divides by the first for the operator and the scale factor (`geogridRadar.cpp:1166`,
    # `:1191`) and by the second for the search extent (`:1207`), so both are carried.
    gx, gy, gz = utm_point(305.0, SR + 4000 * DR)
    g, sp, _ = pointgeometry(UTM, gx, gy, gz, RC, NORMAL)

    @test sp.operator[1] === DR
    @test sp.search[1] === DR
    @test sp.search[2] === g.ylen
    @test sp.operator[2] != sp.search[2]
    # UTM's scale distortion is well under a percent at this distance from the central meridian.
    @test sp.search[2] / sp.operator[2] ≈ 1.0 rtol = 0.02
end

@testset "the shared outputs consume it" begin
    # `outputs.jl` works on a radar `PointGeometry` with no radar-specific code, given the right
    # spacing pair. That is why those functions take spacings as values.
    gx, gy, gz = utm_point(305.0, SR + 4000 * DR)
    g, sp, _ = pointgeometry(UTM, gx, gy, gz, RC, NORMAL)
    dt = 6 * 86400.0

    o = offset_to_velocity(g, sp.operator, dt)
    @test length(o) == 4
    @test all(isfinite, o)

    @test all(isfinite, scale_factors(g, sp.operator))

    # Band 3: velocity per pixel along each image axis, positive and ordered as the spacings are.
    b3x = axis_velocity(sp.operator[1], dt)
    b3y = axis_velocity(sp.operator[2], dt)
    @test b3x > 0 && b3y > 0
    @test b3y > b3x        # the azimuth step is larger than one range sample
end

@testset "cross_check stays clear of its gate on radar geometry" begin
    # The concern `REFERENCE.md` records: the operator is skipped where `cross_check <= 1.0`, and radar
    # geometry is oblique enough to approach it where the projected path never does. Measured across
    # the swath rather than assumed.
    worst = Inf
    for nsamp in (100, 2000, 4000, 7000, 9000), t in (302.0, 305.0, 310.0)
        gx, gy, gz = utm_point(t, SR + nsamp * DR)
        g, _, _ = pointgeometry(UTM, gx, gy, gz, RC, NORMAL)
        worst = min(worst, cross_check(g))
    end
    @test worst > 1.0
    @info "radar cross_check margin above the 1° gate" min_degrees = worst
end

@testset "both look sides solve" begin
    for side in (LookLeft, LookRight)
        c = RadarCoordinate(; orbit = GORB, incidence_angle = RIA, RKW..., look_side = side)
        gx, gy, gz = utm_point(305.0, SR + 4000 * DR; side)
        g, sp, p = pointgeometry(UTM, gx, gy, gz, c, NORMAL)
        @test p.range ≈ SR + 4000 * DR rtol = 1e-12
        @test norm3(g.xunit) ≈ 1.0 atol = 1e-15
        @test all(isfinite, sp.operator)
    end
end

@testset "the range-Doppler solve lands on the range sphere" begin
    # Its own property, independent of the surrounding kernel: the returned point is at the requested
    # slant range from the satellite and at the requested height.
    satx, satv = interpolate(GORB, 305.0)
    rng = SR + 4000 * DR
    for height in (0.0, 500.0, 4000.0)
        xyz, llh = _range_doppler(EL, RC, satx, satv, rng, height)
        @test norm3(satx - xyz) ≈ rng rtol = 1e-9
        @test llh[3] === height
        # And the ECEF and geodetic returns describe the same point.
        @test norm3(lonlat_to_xyz(EL, llh) - xyz) < 1e-6
    end
end

@testset "iteration counts are the reference's" begin
    @test RANGE_DOPPLER_ITERATIONS == 10
    @test GEO2RDR_ITERATIONS == 51
end

@testset "type stable and non-allocating" begin
    gx, gy, gz = utm_point(305.0, SR + 4000 * DR)
    @test @inferred(pointgeometry(UTM, gx, gy, gz, RC, NORMAL)) isa
        Tuple{PointGeometry,RadarSpacing,Any}

    satx, satv = interpolate(GORB, 305.0)
    @test @inferred(_range_doppler(EL, RC, satx, satv, SR, 500.0)) isa
        Tuple{SVector{3,Float64},SVector{3,Float64}}

    # Allocation is asserted on the identity transform: a PROJ call allocates inside PROJ, which is
    # not this kernel's doing and would mask its own behavior.
    idt = transform_pair(IdentityTransform())
    lon, lat, h = known_point(305.0, SR + 4000 * DR)
    pointgeometry(idt, lon, lat, h, RC, NORMAL)
    @test @allocated(pointgeometry(idt, lon, lat, h, RC, NORMAL)) == 0
end
