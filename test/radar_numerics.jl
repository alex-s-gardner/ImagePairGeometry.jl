# The radar numerics against isce3: ellipsoid, orbit interpolation, rdr2geo.
#
# `reference/gen_radar_numerics.py` calls `isce3.core.Ellipsoid`, `isce3.core.Orbit` and
# `isce3.geometry.rdr2geo` — the same objects `geogridRadar.cpp` links against — and records every
# float as a hex literal so the comparison here is on bit patterns rather than on a decimal repr.
#
# The orbit and rdr2geo cases are asserted bitwise: they are pure arithmetic on values both sides
# receive identically, so the two reasons the projected path's floats cannot be bitwise under a
# reprojection (contraction, and PROJ's platform variance) do not apply.
#
# The ellipsoid conversions are not, because this package uses WGS84's `e2` at full precision where
# isce3 truncates it to eight digits. PROJ is the reference for those and the fixture a secondary
# check, at the tolerance that difference implies — see `ELLIPSOID_FIXTURE_ATOL`.
#
# Four independent checks, because a fixture alone cannot tell a correct implementation from a
# faithful transcription of a misreading:
#
#   PROJ                — the reference for the ellipsoid conversions
#   the fixture         — bitwise for orbit and rdr2geo, to a tolerance for the ellipsoid
#   Geodesy.jl          — an unrelated formulation, agreeing at 1e-9
#   internal properties — round trips, orthonormality, and the analytic orbit

using ImagePairGeometry
using ImagePairGeometry: Ellipsoid, WGS84_A, WGS84_E2, semiminor, r_east,
                         lonlat_to_xyz, xyz_to_lonlat, geodetic_tcn, nadir_sphere, TCNBasis,
                         Orbit, OrbitDomainError, interpolate, statetime, starttime, stoptime,
                         geo2rdr, RadarPoint, range_index, azimuth_index, GEO2RDR_ITERATIONS,
                         rdr2geo, rdr2geo_converged, LookSide, LookLeft, LookRight, looksign,
                         RDR2GEO_THRESHOLD, RDR2GEO_MAXITER, RDR2GEO_EXTRAITER,
                         dot3, cross3, norm3, unitvec3
using Geodesy: Geodesy, LLA, ECEF, wgs84
using JSON3
using Proj
using StaticArrays: SVector
using Test

const RFIX = JSON3.read(read(joinpath(@__DIR__, "reference", "radar_numerics.json"), String))

"""The fixture's exact value for a recorded float, from its hex literal."""
fx(v) = parse(Float64, v.hex)

"""A recorded three-vector, exactly."""
fv(v) = SVector{3,Float64}(fx(v[1]), fx(v[2]), fx(v[3]))

const EL = Ellipsoid()

"""
The isce3 truncation of `e2`, `0.0066943799901`, against which the fixture was generated.
"""
const ISCE3_E2 = 0.0066943799901

"""
How far a position computed on WGS84's `e2` may sit from the fixture's, which was computed on
isce3's eight-digit truncation of it.

The two constants differ by 6e-12 relative, which reaches an ECEF position as 1.5e-7 m. That is 6e-8
of a range sample, so it moves no rounded index: every band of the `radar geogrid vs reference`
fixture still matches. This is the size of the datum difference, not a tolerance on the
implementation — PROJ is what pins that, to 6e-9 m forward.
"""
const ELLIPSOID_FIXTURE_ATOL = 1e-6

# PROJ's own geographic-to-geocentric pipeline on full-precision WGS84: the reference for the
# ellipsoid conversions. Takes degrees and (lon, lat, h), where the functions under test take
# radians.
const PROJ_LL_TO_XYZ = Proj.Transformation("EPSG:4979", "EPSG:4978"; always_xy = true)
const PROJ_XYZ_TO_LL = Proj.Transformation("EPSG:4978", "EPSG:4979"; always_xy = true)

proj_ll_to_xyz(llh) =
    SVector{3,Float64}(PROJ_LL_TO_XYZ(rad2deg(llh[1]), rad2deg(llh[2]), llh[3])...)

function proj_xyz_to_ll(xyz)
    lon_d, lat_d, h = PROJ_XYZ_TO_LL(xyz[1], xyz[2], xyz[3])
    SVector{3,Float64}(deg2rad(lon_d), deg2rad(lat_d), h)
end

@testset "provenance" begin
    p = RFIX.provenance
    @test fx(p.ellipsoid_a) === WGS84_A
    # The fixture was generated on isce3's truncated `e2`, which is what makes every comparison
    # against it below a tolerance rather than a bitwise one.
    @test fx(p.ellipsoid_e2) === ISCE3_E2
    @info "radar numerics fixture provenance" isce3 = p.isce3_version numpy = p.numpy_version
end

@testset "ellipsoid constants" begin
    @test EL.a === WGS84_A
    @test EL.e2 === WGS84_E2
    @test WGS84_E2 === 6.69437999014e-3          # WGS84 at full precision, not isce3's truncation
    @test WGS84_E2 > ISCE3_E2
    # The truncation is small enough that the two ellipsoids share a semi-minor axis to well inside
    # a nanometer, which is why no downstream band moves.
    @test semiminor(EL) ≈ fx(RFIX.ellipsoid.b) atol = ELLIPSOID_FIXTURE_ATOL
end

@testset "lonlat_to_xyz agrees with PROJ" begin
    # PROJ resolves EPSG:4979 to EPSG:4978 on full-precision WGS84, so this is the accuracy
    # statement for the forward conversion; the fixture below is the secondary check.
    worst = 0.0
    for c in RFIX.ellipsoid.cases
        llh = fv(c.llh)
        worst = max(worst, norm3(lonlat_to_xyz(EL, llh) - proj_ll_to_xyz(llh)))
    end
    @test worst < 1e-8
    @info "lonlat_to_xyz vs PROJ" meters = worst
end

@testset "lonlat_to_xyz is within the datum difference of isce3" begin
    for c in RFIX.ellipsoid.cases
        llh = fv(c.llh)
        got = lonlat_to_xyz(EL, llh)
        @test norm3(got - fv(c.xyz)) < ELLIPSOID_FIXTURE_ATOL
        # ...and on isce3's own constant it is bitwise, which is what separates the datum from the
        # arithmetic: every operation agrees exactly, only the constant differs.
        @test lonlat_to_xyz(Ellipsoid(WGS84_A, ISCE3_E2), llh) === fv(c.xyz)
    end
end

@testset "xyz_to_lonlat closes on its own input exactly" begin
    # The accuracy statement for the inverse, and it is a round trip against an exact forward rather
    # than a comparison with PROJ. PROJ's inverse loses accuracy with height: at the fixture's
    # 700 km case it returns a height 4.0e-3 m out and a latitude 1.9e-8° out, where Vermeille's
    # closed form recovers the input to 6e-10 m. So PROJ cannot be the reference at satellite
    # altitude, and the strongest available check is that inverting an exactly-computed ECEF returns
    # the coordinates it was computed from.
    #
    # `lonlat_to_xyz` is exact for this purpose, being bitwise against isce3 on isce3's own constant
    # -- asserted below -- and within 6e-9 m of PROJ forward, where PROJ is accurate.
    worst_h = 0.0
    worst_ground = 0.0
    for c in RFIX.ellipsoid.cases
        llh = fv(c.llh)
        back = xyz_to_lonlat(EL, lonlat_to_xyz(EL, llh))
        worst_h = max(worst_h, abs(back[3] - llh[3]))
        # Angles as a ground distance, which is the quantity that matters and is comparable across
        # latitudes. A pole has no defined longitude, so comparing angles there would report a
        # spurious disagreement; re-projecting both compares only what is determined.
        worst_ground = max(worst_ground, norm3(lonlat_to_xyz(EL, back) - lonlat_to_xyz(EL, llh)))
    end
    @test worst_h < 1e-8
    @test worst_ground < 1e-8
    @info "xyz_to_lonlat round trip" height_meters = worst_h ground_meters = worst_ground
end

@testset "xyz_to_lonlat agrees with PROJ below satellite altitude" begin
    # Where PROJ is accurate the two agree to 1e-6 m, which is the independent check on the inverse.
    # Above the troposphere PROJ's own accuracy is the limit rather than this implementation's, so
    # the comparison is confined to heights where that is not the case; the round trip above covers
    # every case including the 700 km one.
    worst = 0.0
    for c in RFIX.ellipsoid.cases
        llh_in = fv(c.llh)
        llh_in[3] > 1e5 && continue
        xyz = fv(c.xyz)
        worst = max(worst,
                    norm3(proj_ll_to_xyz(xyz_to_lonlat(EL, xyz)) -
                          proj_ll_to_xyz(proj_xyz_to_ll(xyz))))
    end
    @test worst < 1e-6
    @info "xyz_to_lonlat vs PROJ, h <= 100 km" meters = worst
end

@testset "xyz_to_lonlat is within the datum difference of isce3" begin
    # The height carries the datum difference directly, with no inverse trigonometry in it; the
    # angles carry it plus one ULP of `atan2`, so both are compared as a position.
    for c in RFIX.ellipsoid.cases
        xyz = fv(c.xyz)
        got = xyz_to_lonlat(EL, xyz)
        want = fv(c.llh_roundtrip)

        @test got[3] ≈ want[3] atol = ELLIPSOID_FIXTURE_ATOL
        @test norm3(lonlat_to_xyz(EL, got) - lonlat_to_xyz(EL, want)) < ELLIPSOID_FIXTURE_ATOL

        # On isce3's constant the height is bitwise, localizing the difference to the datum: the
        # height is computed from `k`, `d` and `z` alone, so every operation upstream agrees exactly.
        @test xyz_to_lonlat(Ellipsoid(WGS84_A, ISCE3_E2), xyz)[3] === want[3]
    end
end

@testset "every libm involved is faithful, none is correctly rounded" begin
    # Why the ULP above is a library difference rather than a transcription error, and why switching
    # libraries would not close it.
    #
    # `atan2` is *itself* platform-dependent, which is the fact that governs this. The fixture's
    # angles were produced by the libm of the machine that generated it (aarch64 macOS), and the
    # system libm here may or may not be that one — on x86-64 Linux and Windows it agrees with
    # openlibm instead, so the fixture's last bit is not reproducible by either local implementation.
    # `REFERENCE.md` records the same phenomenon for PROJ.
    #
    # So the assertions below are the ones that hold on every platform: both implementations are
    # faithful — within one ULP of the true value, evaluated at 256 bits — and neither is correctly
    # rounded. A correctly-rounded `atan2` would make the choice obvious; two faithful ones mean
    # switching buys agreement with one machine rather than accuracy, and gives up the
    # cross-platform reproducibility openlibm exists to provide.
    setprecision(BigFloat, 256) do
        openlibm_closer = 0
        system_closer = 0
        agree = 0
        for c in RFIX.ellipsoid.cases
            xyz = fv(c.xyz)
            abs(rad2deg(fv(c.llh_roundtrip)[2])) > 89.999 && continue

            correct = Float64(atan(BigFloat(xyz[2]), BigFloat(xyz[1])))
            ours = atan(xyz[2], xyz[1])
            theirs = @ccall atan2(xyz[2]::Float64, xyz[1]::Float64)::Float64

            e_ours = abs(reinterpret(Int64, ours) - reinterpret(Int64, correct))
            e_theirs = abs(reinterpret(Int64, theirs) - reinterpret(Int64, correct))

            # Faithful rounding: the strongest property both libraries actually guarantee.
            @test e_ours <= 1
            @test e_theirs <= 1

            e_ours < e_theirs && (openlibm_closer += 1)
            e_theirs < e_ours && (system_closer += 1)
            ours === theirs && (agree += 1)
        end
        # Neither library dominates the other. Reported rather than asserted one way, because which
        # of the three counts is nonzero depends on the platform's libm.
        @test openlibm_closer + system_closer + agree > 0
        @info "atan2: openlibm vs this platform's libm" openlibm_closer system_closer bitwise_agreement = agree
    end
end

@testset "the arithmetic is exact; only the datum constant differs" begin
    # What separates the datum from the implementation without depending on any platform's libm. On
    # isce3's own `e2` the height is bitwise on every case and every platform — it is computed from
    # `k`, `d` and `z` with no inverse trigonometry — so every term upstream of the two `atan2` calls
    # agrees exactly. An error in the arithmetic would move the height on that constant too.
    el_isce = Ellipsoid(WGS84_A, ISCE3_E2)
    for c in RFIX.ellipsoid.cases
        @test xyz_to_lonlat(el_isce, fv(c.xyz))[3] === fv(c.llh_roundtrip)[3]
        @test lonlat_to_xyz(el_isce, fv(c.llh)) === fv(c.xyz)
    end
end

@testset "the datum difference is negligible in position" begin
    # What the difference costs where it is eventually spent, against a range sample of about 2.3 m.
    # It can only move a range or azimuth index for a point already this close to a rounding
    # boundary, which is why every band of the geogrid fixture still matches.
    dr = fx(RFIX.radar.dr)
    worst = 0.0
    for c in RFIX.ellipsoid.cases
        want = fv(c.llh_roundtrip)
        got = xyz_to_lonlat(EL, fv(c.xyz))
        worst = max(worst, norm3(lonlat_to_xyz(EL, got) - lonlat_to_xyz(EL, want)))
    end
    @test worst < ELLIPSOID_FIXTURE_ATOL
    @test worst / dr < 1e-6
    @info "ellipsoid inverse: worst-case position difference from the fixture" meters = worst range_samples = worst / dr
end

@testset "round trip closes" begin
    for c in RFIX.ellipsoid.cases
        llh = fv(c.llh)
        back = xyz_to_lonlat(EL, lonlat_to_xyz(EL, llh))
        # Height in meters: the loose one, since it is a difference of large radii.
        @test back[3] ≈ llh[3] atol = 1e-6
        abs(rad2deg(llh[2])) > 89.999 && continue
        @test back[1] ≈ llh[1] atol = 1e-12
        @test back[2] ≈ llh[2] atol = 1e-12
    end
end

# The independent check. Geodesy.jl uses GeographicLib's series on the full-precision WGS84 datum,
# so it cannot agree to the bit — but a genuinely different formulation landing within 1e-9 is
# evidence the transcription reads the reference correctly, which a fixture generated from that same
# reference cannot provide.
@testset "agrees with Geodesy.jl at 1e-9" begin
    for c in RFIX.ellipsoid.cases
        llh = fv(c.llh)
        lon_d, lat_d = rad2deg(llh[1]), rad2deg(llh[2])
        abs(lat_d) > 89.999 && continue

        ours = lonlat_to_xyz(EL, llh)
        theirs = ECEF(LLA(lat_d, lon_d, llh[3]), wgs84)

        # Relative, on a magnitude of order the Earth's radius. Both sides are on full-precision
        # WGS84, so the floor here is each implementation's own accuracy rather than a datum
        # difference: Geodesy.jl uses GeographicLib's series where this is Vermeille's closed form.
        scale = max(norm3(ours), 1.0)
        @test abs(ours[1] - theirs.x) / scale < 1e-9
        @test abs(ours[2] - theirs.y) / scale < 1e-9
        @test abs(ours[3] - theirs.z) / scale < 1e-9

        # And the inverse, in the other direction.
        back = xyz_to_lonlat(EL, ours)
        their_back = LLA(ECEF(ours[1], ours[2], ours[3]), wgs84)
        @test rad2deg(back[1]) ≈ their_back.lon atol = 1e-9
        @test rad2deg(back[2]) ≈ their_back.lat atol = 1e-9
        @test back[3] ≈ their_back.alt atol = 1e-4
    end
end

@testset "r_east" begin
    # At the equator the prime vertical radius is the semi-major axis exactly.
    @test r_east(EL, 0.0) === WGS84_A
    # At the pole it is a / sqrt(1 - e2), the largest it gets.
    @test r_east(EL, pi / 2) ≈ WGS84_A / sqrt(1 - WGS84_E2) rtol = 1e-15
    # Monotone in |lat|.
    @test r_east(EL, 0.3) < r_east(EL, 0.6) < r_east(EL, 0.9)
end

@testset "TCN basis is orthonormal and right-signed" begin
    for c in RFIX.tcn.cases
        pos = fv(c.position)
        vel = fv(c.velocity)
        b = geodetic_tcn(pos, vel)

        for u in (b.that, b.chat, b.nhat)
            @test norm3(u) ≈ 1.0 atol = 1e-15
        end
        @test abs(dot3(b.that, b.chat)) < 1e-15
        @test abs(dot3(b.chat, b.nhat)) < 1e-15
        @test abs(dot3(b.that, b.nhat)) < 1e-15

        # `nhat` points down — antiparallel to the position vector, and geocentric rather than
        # geodetic, which is what the range-Doppler solve assumes.
        @test dot3(b.nhat, unitvec3(pos)) ≈ -1.0 atol = 1e-15
        # `that` is the along-track direction, so it agrees with the velocity.
        @test dot3(b.that, unitvec3(vel)) > 0.99
        # Right-handed: t × c = n.
        @test cross3(b.that, b.chat) ≈ b.nhat atol = 1e-14
    end
end

@testset "nadir_sphere" begin
    for c in RFIX.tcn.cases
        pos = fv(c.position)
        radius, height, eta = nadir_sphere(EL, pos)
        # The local radius sits between the two axes, inclusively: it reaches the semi-major axis
        # exactly for a satellite in the equatorial plane, which the fixture's orbit is at t = 0.
        @test semiminor(EL) <= radius <= WGS84_A
        @test height > 0
        @test radius + height ≈ norm3(pos) rtol = 1e-15
        @test 0 < eta < 1
    end
end

# ------------------------------------------------------------------------------------ orbit

"""The fixture's orbit, built from the state vectors it recorded."""
function fixture_orbit()
    svs = RFIX.orbit.state_vectors
    t = [fx(s.t) for s in svs]
    pos = [fv(s.position) for s in svs]
    vel = [fv(s.velocity) for s in svs]
    return Orbit(; time = t, position = pos, velocity = vel)
end

const ORB = fixture_orbit()

"""The radar wavelength and look side the fixture cases were generated with."""
const WVL = fx(RFIX.rdr2geo.wavelength)
fixture_side(c) = c.side == "right" ? LookRight : LookLeft

# The scene-center start every `geo2rdr` solve below begins from, as the reference does — it uses the
# scene midpoint for every grid point rather than a per-point estimate.
const TMID = 0.5 * (starttime(ORB) + stoptime(ORB))
const PM, VM = interpolate(ORB, TMID)

@testset "orbit construction" begin
    @test length(ORB) == RFIX.orbit.spec.n
    @test ORB.spacing === fx(RFIX.orbit.spec.spacing)
    @test starttime(ORB) === fx(RFIX.orbit.spec.t0)

    svs = RFIX.orbit.state_vectors
    for (i, s) in enumerate(svs)
        @test statetime(ORB, i) ≈ fx(s.t) atol = 1e-12
    end
    @test stoptime(ORB) ≈ fx(svs[end].t) atol = 1e-12
end

@testset "orbit input validation" begin
    t = [fx(s.t) for s in RFIX.orbit.state_vectors]
    pos = [fv(s.position) for s in RFIX.orbit.state_vectors]
    vel = [fv(s.velocity) for s in RFIX.orbit.state_vectors]

    @test_throws "at least 4 state vectors" Orbit(; time = t[1:3], position = pos[1:3],
                                                 velocity = vel[1:3])
    @test_throws "each state vector needs both" Orbit(ORB.t0, ORB.spacing, pos, vel[1:(end - 1)])

    # A non-uniform axis has no Linspace representation, so it is refused rather than resampled.
    tbad = copy(t)
    tbad[5] += 0.5
    @test_throws "uniformly spaced" Orbit(; time = tbad, position = pos, velocity = vel)

    @test_throws "finite and nonzero" Orbit(0.0, 0.0, pos, vel)
end

@testset "Hermite position is bitwise or 1 ULP, velocity to 1e-13 relative" begin
    # Position is bitwise on every case. Velocity is not, and the cause is isce3's compiled
    # accumulation rather than the transcription: an independent NumPy implementation of the same
    # formula — no contraction, explicit evaluation order — disagrees with isce3 in exactly the same
    # cases and directions as this one does, and agrees with this one. Inserting `fma` into the
    # velocity accumulation moves the worst ULP count from 80896 to 43911 without closing it, which
    # identifies contraction as the mechanism while showing the pattern is not one `fma` site.
    #
    # The velocity terms are where the two differ because they carry a cancellation the position
    # terms do not: `g0` is a difference of two products (`f0*hdot - s*h`) whose magnitudes exceed
    # their difference, so a fused evaluation and a rounded one diverge at a relative scale the
    # position weights never reach.
    #
    # 4e-14 relative is 3e-10 m/s on an orbital speed of 7.5 km/s. What that costs downstream is
    # measured, not assumed: the `geo2rdr inverts rdr2geo` testset below closes the round trip to
    # 1e-9 s and 1e-6 m through this interpolator.
    #
    # Position is bitwise on 10 of the 11 cases; the exception is `clamp_high`, at 1 ULP in one
    # component, and the same NumPy implementation shows the same 1 ULP there. So the position
    # accumulation is affected by the same contraction as the velocity, just far less — one term of
    # `h[i]^2 * (...)` rather than a cancelling difference.
    exact_pos = 0
    worst_pos_ulp = 0
    worst_vel = 0.0
    for c in RFIX.orbit.cases
        t = fx(c.t)
        p, v = interpolate(ORB, t)
        pr, vr = fv(c.position), fv(c.velocity)

        p === pr && (exact_pos += 1)
        for k in 1:3
            u = abs(reinterpret(Int64, p[k]) - reinterpret(Int64, pr[k]))
            @test u <= 1
            worst_pos_ulp = max(worst_pos_ulp, u)
        end

        rel = norm3(v - vr) / norm3(vr)
        @test rel < 1e-13
        worst_vel = max(worst_vel, rel)
    end
    @test exact_pos >= 10
    @info "Hermite interpolation vs isce3" bitwise_position_cases = "$exact_pos/$(length(RFIX.orbit.cases))" worst_position_ulp = worst_pos_ulp velocity_relative = worst_vel
end

@testset "interpolation is exact at the state vectors" begin
    # An interpolant that matches value and derivative at its nodes must reproduce them exactly.
    # This is what distinguishes Hermite from a cubic fit through positions alone.
    for (i, s) in enumerate(RFIX.orbit.state_vectors)
        # Skip the endpoints: the index clamp makes those one-sided, so `statetime` and the
        # recorded time can differ in the last bit and the node is no longer hit exactly.
        (i <= 2 || i >= length(ORB) - 1) && continue
        p, v = interpolate(ORB, statetime(ORB, i))
        @test p ≈ fv(s.position) atol = 1e-6
        @test v ≈ fv(s.velocity) atol = 1e-9
    end
end

@testset "interpolation reproduces the analytic orbit" begin
    # The fixture's trajectory is an exact circle, so the interpolant can be checked against the
    # closed form rather than only against isce3. Hermite on four nodes of a smooth arc is good to
    # a few parts in 1e10 of the radius.
    s = RFIX.orbit.spec
    R, w, inc = fx(s.R), fx(s.w), fx(s.inc)
    for t in (37.5, 122.5, 155.0, 201.3)
        p, v = interpolate(ORB, t)
        a = w * t
        want = SVector{3,Float64}(R * cos(a), R * sin(a) * cos(inc), R * sin(a) * sin(inc))
        @test norm3(p - want) / R < 1e-10
        # Speed is constant on a circular orbit.
        @test norm3(v) ≈ R * w rtol = 1e-9
        # And the velocity is perpendicular to the position.
        @test abs(dot3(unitvec3(p), unitvec3(v))) < 1e-9
    end
end

@testset "out of domain throws" begin
    @test_throws OrbitDomainError interpolate(ORB, starttime(ORB) - 1e-6)
    @test_throws OrbitDomainError interpolate(ORB, stoptime(ORB) + 1e-6)
    @test_throws "outside the orbit's span" interpolate(ORB, stoptime(ORB) + 100.0)
    # The edges themselves are in the domain.
    @test interpolate(ORB, starttime(ORB)) isa Tuple
    @test interpolate(ORB, stoptime(ORB)) isa Tuple
end

# ------------------------------------------------------------------------------------ rdr2geo

@testset "rdr2geo defaults match isce3" begin
    @test RDR2GEO_THRESHOLD === fx(RFIX.rdr2geo.threshold)
    @test RDR2GEO_MAXITER == RFIX.rdr2geo.maxiter
    @test RDR2GEO_EXTRAITER == RFIX.rdr2geo.extraiter
end

@testset "looksign" begin
    # Inverted relative to how it reads: right-looking is -1, per geogridRadar.cpp:1048.
    @test looksign(LookRight) === -1.0
    @test looksign(LookLeft) === 1.0
end

@testset "rdr2geo agrees with isce3 to sub-micrometers on the ground" begin
    # Not bitwise, for three reasons upstream of this solve rather than anything in it: the datum
    # constant differs from the fixture's by 6e-12 relative, the ellipsoid inverse carries a 1-ULP
    # `atan2` difference (openlibm versus the platform libm), and the Hermite velocity carries a
    # 4e-14 contraction difference. This solve calls all three, repeatedly.
    #
    # The bound that matters is the one in meters. A ULP of longitude is not a unit anyone cares
    # about; where the result is spent is as a ground position feeding a range and azimuth index, so
    # that is what is bounded here — and it is dominated by the datum difference, so it sits at the
    # same 1e-6 m as every other fixture comparison.
    worst_ulp = 0
    worst_ground = 0.0
    dr = fx(RFIX.radar.dr)
    for c in RFIX.rdr2geo.cases
        side = fixture_side(c)
        got = rdr2geo(ORB, EL, fx(c.t), fx(c.range);
                      height = fx(c.height), wavelength = WVL, side)
        want = fv(c.llh)

        for k in 1:2
            worst_ulp = max(worst_ulp, abs(reinterpret(Int64, got[k]) - reinterpret(Int64, want[k])))
        end
        ground = norm3(lonlat_to_xyz(EL, got) - lonlat_to_xyz(EL, want))
        worst_ground = max(worst_ground, ground)

        @test ground < ELLIPSOID_FIXTURE_ATOL
        @test ground / dr < 1e-6
        # The height is what the DEM snap pins, so it tracks the requested value rather than
        # drifting with the angles. Relative, since a nanometer on 4000 m is below `eps`.
        @test got[3] ≈ want[3] rtol = 1e-12 atol = 1e-8
    end
    @info "rdr2geo vs isce3" worst_angle_ulp = worst_ulp worst_ground_meters = worst_ground range_samples = worst_ground / dr
end

@testset "rdr2geo lands on the range sphere at zero Doppler" begin
    for c in RFIX.rdr2geo.cases
        side = fixture_side(c)
        t, rng = fx(c.t), fx(c.range)
        llh = rdr2geo(ORB, EL, t, rng; height = fx(c.height), wavelength = WVL, side)

        target = lonlat_to_xyz(EL, llh)
        pos, vel = interpolate(ORB, t)

        # At the requested slant range.
        @test abs(norm3(target - pos) - rng) < 1e-6
        # And in the zero-Doppler plane.
        look = target - pos
        @test abs(dot3(look, vel)) / (norm3(look) * norm3(vel)) < 1e-12
        # At the requested height, which is what the DEM snap enforces.
        @test llh[3] ≈ fx(c.height) atol = 1e-6
    end
end

@testset "the two look sides land on opposite sides of the track" begin
    t, rng = 120.0, 8.6e5
    pos, vel = interpolate(ORB, t)
    b = geodetic_tcn(pos, vel)

    right = lonlat_to_xyz(EL, rdr2geo(ORB, EL, t, rng; height = 0.0, wavelength = WVL,
                                      side = LookRight))
    left = lonlat_to_xyz(EL, rdr2geo(ORB, EL, t, rng; height = 0.0, wavelength = WVL,
                                     side = LookLeft))

    # The cross-track component of the look vector flips sign, and nothing else does.
    @test dot3(right - pos, b.chat) * dot3(left - pos, b.chat) < 0
    @test norm3(right - pos) ≈ norm3(left - pos) rtol = 1e-9
    @test right != left
end

@testset "rdr2geo converges" begin
    for c in RFIX.rdr2geo.cases
        side = fixture_side(c)
        _, conv = rdr2geo_converged(ORB, EL, fx(c.t), fx(c.range);
                                    height = fx(c.height), wavelength = WVL, side)
        @test conv
    end
end

# ------------------------------------------------------------------------------------ geo2rdr

# `geo2rdr` exists only inside the compiled kernel, so it has no callable reference. It is verified
# as the inverse of `rdr2geo`, which does: a target placed at a known time and range must be
# recovered at that time and range.
@testset "geo2rdr inverts rdr2geo" begin
    # Start every solve from the same mid-orbit guess, as the reference does — it uses the scene
    # midpoint for every grid point rather than a per-point initial estimate.

    for c in RFIX.rdr2geo.cases
        side = fixture_side(c)
        t, rng = fx(c.t), fx(c.range)
        target = lonlat_to_xyz(EL, fv(c.llh))

        p = geo2rdr(ORB, target, TMID, TMID, PM, VM)

        # Nanosecond agreement on a time reached by Newton from 60 s away.
        @test abs(p.aztime - t) < 1e-9
        @test abs(p.orbittime - t) < 1e-9
        # Micron agreement on a range of 800 km.
        @test abs(p.range - rng) < 1e-6
        # `look` is the unnormalized satellite-to-target vector, so its length is the range.
        @test norm3(p.look) ≈ p.range rtol = 1e-15
    end
end

@testset "geo2rdr reaches zero Doppler" begin
    for c in RFIX.rdr2geo.cases
        target = lonlat_to_xyz(EL, fv(c.llh))
        p = geo2rdr(ORB, target, TMID, TMID, PM, VM)
        # The condition the solve is driving to zero, normalized so the tolerance is dimensionless.
        @test abs(dot3(p.look, p.velocity)) / (norm3(p.look) * norm3(p.velocity)) < 1e-11
    end
end

@testset "the two time scales stay offset by their initial difference" begin
    # Both receive the same Newton increment each iteration, which is what lets the reference carry
    # an image time scale and an orbit time scale independently and read the azimuth index off the
    # former while interpolating on the latter.
    #
    # The offset is preserved to rounding, not to the bit: `a - s` and `b - s` for the same `s` are
    # each rounded to their own exponent, so the difference drifts. The drift is bounded by an ULP of
    # the *times*, not of the offset — around 600 s here, so ~1e-13 s per iteration — which is why a
    # small offset like one pulse interval shows the largest relative error while being the most
    # accurate in absolute terms.
    #
    # So the bound is stated absolutely, and in the unit the offset is eventually spent in: fractions
    # of an azimuth line.
    target = lonlat_to_xyz(EL, fv(RFIX.rdr2geo.cases[1].llh))
    prf = fx(RFIX.radar.prf)

    worst_lines = 0.0
    for offset in (0.0, 1.0 / prf, -3.5, 1e5)
        p = geo2rdr(ORB, target, TMID + offset, TMID, PM, VM)
        drift = abs((p.aztime - p.orbittime) - offset)
        # An ULP of `TMID` per iteration, at most.
        @test drift < GEO2RDR_ITERATIONS * eps(max(abs(TMID) + abs(offset), 1.0))
        @test drift * prf < 1e-8
        worst_lines = max(worst_lines, drift * prf)
    end
    # A zero offset is the production case, and it is preserved exactly.
    p0 = geo2rdr(ORB, target, TMID, TMID, PM, VM)
    @test p0.aztime === p0.orbittime
    @info "geo2rdr time-scale drift" worst_azimuth_lines = worst_lines
end

@testset "convergence is linear, so the iteration count is load-bearing" begin
    # Measured rather than assumed, because the answer is not what a Newton solve would suggest.
    # `fnprime` drops the acceleration term of the true derivative (`geogridRadar.cpp:959`), so this
    # is not Newton's method but a fixed-point iteration with a slightly wrong slope — and it
    # converges *linearly*, at a factor of about 11 per step, not quadratically.
    #
    # That is why 51 iterations is a plausible choice and not obvious overkill: from a scene-center
    # start about 60 s from the answer it takes roughly 15 iterations to reach machine precision,
    # after which the estimate oscillates at the 1e-12 level rather than settling. Both facts make
    # `GEO2RDR_ITERATIONS` unlowerable without changing outputs.
    c1 = RFIX.rdr2geo.cases[1]
    target = lonlat_to_xyz(EL, fv(c1.llh))
    t_true = fx(c1.t)

    function solve_n(n)
        satx, satv, az = PM, VM, TMID
        look = target - satx
        rng = norm3(look)
        for _ in 1:n
            look = target - satx
            rng = norm3(look)
            az -= dot3(look, satv) / -dot3(satv, satv)
            satx, satv = interpolate(ORB, az)
        end
        return az, rng
    end

    az51, rng51 = solve_n(GEO2RDR_ITERATIONS)
    p = geo2rdr(ORB, target, TMID, TMID, PM, VM)
    @test p.aztime === az51
    @test p.range === rng51
    @test GEO2RDR_ITERATIONS == 51

    # Linear, not quadratic: each step cuts the error by a roughly constant factor. A quadratic
    # method would square it, reaching machine precision by iteration 4 from this start.
    errs = [abs(solve_n(n)[1] - t_true) for n in 1:10]
    ratios = [errs[i] / errs[i + 1] for i in 1:9]
    @test all(r -> 5 < r < 25, ratios)
    @test errs[4] > 1e-4        # still milliseconds off where Newton would be exact
    @test errs[10] < 1e-8

    # And it has not settled by iteration 15 — it oscillates, so truncating anywhere past
    # convergence still changes the last bits.
    az15, _ = solve_n(15)
    az20, _ = solve_n(20)
    @test abs(az15 - az51) < 1e-10
    @test az15 !== az51
    @test az20 !== az51
    @info "geo2rdr convergence" mean_error_ratio_per_iteration = sum(ratios) / length(ratios) error_at_4 = errs[4] error_at_10 = errs[10]
end

@testset "range and azimuth indices" begin
    r = RFIX.radar
    dr, prf, sr = fx(r.dr), fx(r.prf), fx(r.starting_range)

    c = RFIX.rdr2geo.cases[1]
    target = lonlat_to_xyz(EL, fv(c.llh))
    p = geo2rdr(ORB, target, TMID, TMID, PM, VM)

    ri = range_index(p, sr, dr)
    ai = azimuth_index(p, 0.0, prf)
    # Integral values, since both round.
    @test ri === round(ri)
    @test ai === round(ai)
    # And they invert: index times spacing recovers the range and time.
    @test abs((sr + ri * dr) - p.range) <= dr
    @test abs((ai / prf) - p.aztime) <= 1.0 / prf

    # Ties go away from zero, as `std::round` does and Julia's `round` does not.
    tie = RadarPoint(0.0, 0.0, sr + 0.5 * dr, SVector{3,Float64}(1, 0, 0),
                     SVector{3,Float64}(0, 0, 0), SVector{3,Float64}(0, 0, 0))
    @test range_index(tie, sr, dr) === 1.0
end
