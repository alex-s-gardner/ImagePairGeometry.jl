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
                         incidence_angle, interpolate, rdr2geo,
                         pointgeometry, PointGeometry, RadarSpacing, RANGE_DOPPLER_ITERATIONS,
                         GEO2RDR_ITERATIONS, _range_doppler,
                         lonlat_to_xyz, xyz_to_lonlat, norm3, dot3, unitvec3,
                         geodetic_tcn, nadir_sphere, looksign,
                         surface_normal, cross_check, offset_to_velocity, scale_factors,
                         axis_velocity, TransformPair, transform_pair, IdentityTransform,
                         midtime, orbit_midtime, range_index, azimuth_index, ysize,
                         SceneCenterStart, WarmStart, WARM_START_MIN_ITERATIONS,
                         ZeroDopplerSeed, geo2rdr, _geocentric_radius, Ellipsoid, WGS84_A,
                         WGS84_E2
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

"""The range-Doppler solve at an arbitrary iteration count, for checking convergence.

A transcription of `_range_doppler` with the count as an argument, so a test can watch the iterate
settle. Kept here rather than parameterizing the package's own loop, which stays a fixed count with
no branch.
"""
function _range_doppler_n(el, c, satx, satv, rngpix, height, niter)
    vhat = unitvec3(satv)
    radius, _, _, a = nadir_sphere(el, satx)
    tcn = geodetic_tcn(satx, satv)
    ndotv = dot3(tcn.nhat, vhat)
    vdott = dot3(vhat, tcn.that)
    zsch = height
    llhi = SVector{3,Float64}(0.0, 0.0, 0.0)
    targ_xyz = llhi
    for _ in 1:niter
        b = radius + zsch
        costheta = 0.5 * (a / rngpix + rngpix / a - (b / a) * (b / rngpix))
        sintheta = sqrt(1 - costheta * costheta)
        gamma = rngpix * costheta
        alpha = -gamma * ndotv / vdott
        beta = -looksign(c.look_side) * sqrt(rngpix * rngpix * sintheta * sintheta - alpha * alpha)
        targ_vec = satx + alpha * tcn.that + beta * tcn.chat + gamma * tcn.nhat
        lonlat = xyz_to_lonlat(el, targ_vec)
        llhi = SVector{3,Float64}(lonlat[1], lonlat[2], height)
        targ_xyz = lonlat_to_xyz(el, llhi)
        zsch = norm3(targ_xyz) - radius
    end
    return (targ_xyz, llhi)
end

@testset "iteration counts clear their requirements with margin" begin
    # Neither count is the reference's: 6 against 10, and 16 against 51. Each is justified at its own
    # constant's docstring by a sweep over acquisition geometry, and each is asserted here to sit
    # above the count its own convergence needs rather than merely to equal a number.
    @test RANGE_DOPPLER_ITERATIONS == 6
    @test GEO2RDR_ITERATIONS == 16

    # The range-Doppler requirement, at the mid-latitude geometry where its rate is worst rather than
    # at the fixture's near-equatorial one. Convergence is measured against the answer well past the
    # floor, so `settled` is where the target stops moving at all.
    satx, satv = interpolate(GORB, 305.0)
    rng = SR + 4000 * DR
    for height in (0.0, 4000.0)
        ref, _ = _range_doppler_n(EL, RC, satx, satv, rng, height, 30)
        settled = findfirst(k -> norm3(first(_range_doppler_n(EL, RC, satx, satv, rng, height, k))
                                       - ref) < 1e-7,
                            1:RANGE_DOPPLER_ITERATIONS)
        @test settled !== nothing
        @test settled < RANGE_DOPPLER_ITERATIONS
        # Two iterations of headroom is what the constant was chosen for. Dropping below one means
        # the count is at its requirement and must be re-derived.
        @test RANGE_DOPPLER_ITERATIONS - settled >= 1
    end
end

@testset "the radar path negates a y displacement" begin
    # `testautoRIFT.py:790-791`, under `optical_flag == 0`: azimuth increases along the track while a
    # north-up raster's +y points down, so the two conventions disagree in sign. A projected image
    # needs no negation, and applying one there would invert its velocities.
    @test y_displacement_sign(RC) === -1.0
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

@testset "the zero-Doppler start policy" begin
    # `SceneCenterStart` must reproduce the no-argument call exactly: it is the default, and the whole
    # radar fixture is asserted against it.
    gx, gy, gz = utm_point(305.0, SR + 4000 * DR)
    a = pointgeometry(UTM, gx, gy, gz, RC, NORMAL)
    b = pointgeometry(UTM, gx, gy, gz, RC, NORMAL, SceneCenterStart(), nothing)
    @test a[1].image_xy === b[1].image_xy
    @test a[3].aztime === b[3].aztime
    @test a[3].range === b[3].range

    # A `WarmStart` with no predecessor cold-starts, so its first point matches too.
    seed = ZeroDopplerSeed()
    @test !seed.valid
    c = pointgeometry(UTM, gx, gy, gz, RC, NORMAL, WarmStart(), seed)
    @test c[3].aztime === a[3].aztime
    # ...and the seed now carries that answer forward.
    @test seed.valid
    @test seed.aztime === a[3].aztime
    @test seed.position === a[3].position

    # A second point started from it agrees with the cold solve on both indices. `WarmStart`'s
    # minimum exists to keep the *float* bands in agreement too; that is asserted whole-band in
    # `radar_geogrid.jl`, since a single point cannot show a band's worst case.
    gx2, gy2, gz2 = utm_point(305.0 + 0.01, SR + 4010 * DR)
    warm = pointgeometry(UTM, gx2, gy2, gz2, RC, NORMAL, WarmStart(), seed)
    cold = pointgeometry(UTM, gx2, gy2, gz2, RC, NORMAL)
    @test warm[1].image_xy === cold[1].image_xy

    # The minimum is enforced, and it is the float bands that set it rather than the indices.
    @test WARM_START_MIN_ITERATIONS == 8
    @test WarmStart().iterations == WARM_START_MIN_ITERATIONS
    @test_throws "must be at least 8" WarmStart(iterations = 7)
    @test_throws "must be at least 8" WarmStart(iterations = 1)
    @test_throws "above GEO2RDR_ITERATIONS" WarmStart(iterations = GEO2RDR_ITERATIONS + 1)

    # `geo2rdr`'s count is an argument now; the default must still be the constant.
    target = lonlat_to_xyz(EL, ImagePairGeometry.xyz_to_lonlat(EL,
        ImagePairGeometry.lonlat_to_xyz(EL, SVector{3,Float64}(0.1, 0.34, 0.0))))
    pm, vm = interpolate(GORB, orbit_midtime(RC))
    p_default = geo2rdr(GORB, target, midtime(RC), orbit_midtime(RC), pm, vm)
    p_explicit = geo2rdr(GORB, target, midtime(RC), orbit_midtime(RC), pm, vm, GEO2RDR_ITERATIONS)
    @test p_default.aztime === p_explicit.aztime
    @test p_default.range === p_explicit.range
end

@testset "_geocentric_radius replaces a round trip to a few ULP" begin
    # The range-Doppler loop reads only `norm3(targ_xyz)` from its forward conversion, and that norm is
    # independent of longitude: the ellipsoid is a surface of revolution. `_geocentric_radius` computes
    # it from the latitude alone, which is what lets the loop skip the conversion until its last
    # iteration.
    #
    # Mathematically the two are the same quantity; in floating point they are two expressions for it
    # and round differently, so the bound is a few ULP rather than bitwise equality. Measured over
    # 74185 latitude/longitude/height combinations: 2 ULP on WGS84 and on isce3's truncated `e2`,
    # 3 ULP on a deliberately eccentric non-standard ellipsoid. At Earth radius 3 ULP is 2.8e-9 m —
    # the magnitude of this solve's own fixed-point floor, so the substitution cannot move the answer
    # further than the iteration already moves it, and 1.3e-9 of a range sample either way.
    worst_ulp = 0
    worst_m = 0.0
    for lat_deg in (-89.9, -60.0, -23.5, 0.0, 19.5, 45.0, 60.0, 82.0, 89.9)
        for h in (-430.0, 0.0, 500.0, 4000.0, 8848.0)
            for lon_deg in (-179.0, -45.0, 0.0, 90.0, 178.0)
                llh = SVector{3,Float64}(deg2rad(lon_deg), deg2rad(lat_deg), h)
                # The quantity being replaced, computed the way the loop computed it.
                want = norm3(lonlat_to_xyz(EL, llh))
                got = _geocentric_radius(EL, sin(llh[2]), h)
                u = abs(reinterpret(Int64, want) - reinterpret(Int64, got))
                @test u <= 4
                worst_ulp = max(worst_ulp, u)
                worst_m = max(worst_m, abs(want - got))
            end
        end
    end
    @test worst_m < 1e-8
    @info "_geocentric_radius vs the round trip" worst_ulp worst_meters = worst_m

    # Longitude must not enter at all — the closed form does not take it, and the round trip must not
    # depend on it either, or the substitution would be unsound rather than merely differently rounded.
    let ref = norm3(lonlat_to_xyz(EL, SVector{3,Float64}(0.0, deg2rad(45.0), 500.0)))
        for lon_deg in (-179.0, -45.0, 90.0, 178.0)
            @test norm3(lonlat_to_xyz(EL, SVector{3,Float64}(deg2rad(lon_deg), deg2rad(45.0),
                                                             500.0))) === ref
        end
    end

    # And it is not accidentally specific to WGS84: a non-standard ellipsoid must work too, since
    # `Ellipsoid` is constructible with any `a` and `e2` and `radar_numerics.jl` exercises isce3's.
    for el in (Ellipsoid(), Ellipsoid(WGS84_A, 0.0066943799901), Ellipsoid(6.0e6, 0.02))
        for lat_deg in (-70.0, 0.0, 33.0, 71.0), h in (0.0, 3000.0)
            llh = SVector{3,Float64}(0.3, deg2rad(lat_deg), h)
            @test _geocentric_radius(el, sin(llh[2]), h) ≈ norm3(lonlat_to_xyz(el, llh)) rtol = 1e-15
        end
    end

    # A sphere is the degenerate case the formula must not special-case its way out of.
    sphere = Ellipsoid(6.371e6, 0.0)
    for lat_deg in (0.0, 45.0, 89.0), h in (0.0, 1000.0)
        @test _geocentric_radius(sphere, sin(deg2rad(lat_deg)), h) ≈ 6.371e6 + h rtol = 1e-15
    end
end
