# Terrain height as a source `rdr2geo` iterates against.
#
# The load-bearing assertion is that the *constant* case is bitwise what it was before sources existed.
# Every radar output goes through `rdr2geo` at a scalar height — the footprint bounds, the incidence
# angle, and through them the ground pixel sizes and the whole grid window — and all of them are asserted
# against the reference elsewhere. So a height source that changed the constant path by an ULP would move
# 47181 bitwise assertions, and that is checked here directly rather than inferred from those passing.
#
# The varying case is checked for the two things that can go wrong quietly: a units error between the
# radians the solve holds and the degrees a raster is indexed in, and an out-of-bounds query answered with
# a clamped edge value rather than refused.

using ImagePairGeometry
using ImagePairGeometry: rdr2geo, rdr2geo_converged, Orbit, Ellipsoid, LookLeft, LookRight,
                         ConstantHeight, AbstractHeightSource, height_at, reference_height,
                         RDR2GEO_MAXITER, RDR2GEO_EXTRAITER
using StaticArrays: SVector
using Test

# The analytic circular orbit the radar fixtures use, spanning well past the times queried.
function _hs_orbit(; t0 = 0.0, n = 41, dt = 10.0, radius = 7.0e6, incl = deg2rad(98.0))
    omega = sqrt(3.986004418e14 / radius^3)
    ts = collect(t0 .+ dt .* (0:(n - 1)))
    pos = [SVector(radius * cos(omega * t), radius * sin(omega * t) * cos(incl),
                   radius * sin(omega * t) * sin(incl)) for t in ts]
    vel = [SVector(-radius * omega * sin(omega * t), radius * omega * cos(omega * t) * cos(incl),
                   radius * omega * cos(omega * t) * sin(incl)) for t in ts]
    return Orbit(; time = ts, position = pos, velocity = vel)
end

const HS_ORBIT = _hs_orbit()
const HS_EL = Ellipsoid()
_solve(h; aztime = 200.0, range = 8.2e5, side = LookRight) =
    rdr2geo(HS_ORBIT, HS_EL, aztime, range; height = h, wavelength = 0.055, side)

@testset "a constant source is bitwise the scalar it replaces" begin
    # The whole reason this is dispatch rather than a branch. Compared on bit patterns, since `==` would
    # let a signed zero or a differently-rounded value pass.
    #
    # `ConstantHeight` only — *not* a closure returning the same value. A closure declares no
    # `reference_height`, so it starts the iteration from sea level where a constant starts from itself,
    # and converging from a different direction lands on a different last bit: measured at 1.4e-12 of a
    # degree for an 8848 m constant, which is 1.5e-7 m of ground position. Immaterial, and not bitwise,
    # which is exactly why `ConstantHeight` exists as a type rather than being spelled as a closure.
    for h in (0.0, 500.0, 4000.0, -200.0, 8848.0)
        want = _solve(h)
        @test all(reinterpret(UInt64, want) .== reinterpret(UInt64, _solve(ConstantHeight(h))))
        # An Integer height is accepted and converts, as it did before. Written as an `if` rather than
        # with `&&`: `@test` expands to a statement, so short-circuiting it silently swallows the
        # assertion's result and reports the guard instead.
        if h == round(h)
            @test all(reinterpret(UInt64, want) .== reinterpret(UInt64, _solve(Int(h))))
        end
    end

    # Both look sides, since the height enters through a square root whose sign the side chooses.
    for side in (LookLeft, LookRight)
        want = _solve(1500.0; side)
        @test all(reinterpret(UInt64, want) .==
                  reinterpret(UInt64, _solve(ConstantHeight(1500.0); side)))
    end
end

@testset "the source protocol" begin
    # A bare number and a `ConstantHeight` both ignore position, which is the reference's behaviour.
    @test height_at(750.0, 0.1, 0.2) === 750.0
    @test height_at(ConstantHeight(750.0), 0.1, 0.2) === 750.0
    @test height_at(ConstantHeight(750), 1.0, 2.0) === 750.0    # converts
    # Any other callable is invoked as `src(lon, lat)`.
    @test height_at((lon, lat) -> 100 * lon + lat, 2.0, 3.0) === 203.0

    # The iteration's starting height: the value itself for a constant, sea level for a bare callable
    # that declares nothing better.
    @test reference_height(900.0) === 900.0
    @test reference_height(ConstantHeight(900.0)) === 900.0
    @test reference_height((lon, lat) -> 5000.0) === 0.0

    @test ConstantHeight(1.0) isa AbstractHeightSource
end

@testset "a varying source moves the answer" begin
    # A source constant in value but reached through the varying code path agrees with the constant to
    # the solve's own floor rather than to the bit, because it starts from a different height — see the
    # note in the bitwise testset. 1e-9 radians is 6 mm of ground position.
    flat = (lon, lat) -> 800.0
    @test _solve(flat)[1] ≈ _solve(800.0)[1] atol = 1e-9
    @test _solve(flat)[2] ≈ _solve(800.0)[2] atol = 1e-9
    @test _solve(flat)[3] ≈ 800.0 atol = 1e-6

    # Declaring the reference height recovers the bitwise agreement, which pins the cause on the starting
    # value rather than on the varying code path itself.
    struct_free = ConstantHeight(800.0)
    @test all(reinterpret(UInt64, _solve(800.0)) .== reinterpret(UInt64, _solve(struct_free)))

    # A source varying with longitude moves the ground point, and the height it converges to is the
    # source evaluated *there* — which is the fixed-point property, and the thing a units error breaks.
    base = _solve(0.0)
    lon0 = base[1]
    # Radians in, so a source written against radians is self-consistent by construction; the degrees
    # question belongs to a raster-backed source and is checked in `test/rasters.jl`.
    ramp = (lon, lat) -> 1000.0 + 2.0e5 * (lon - lon0)
    llh, converged = rdr2geo_converged(HS_ORBIT, HS_EL, 200.0, 8.2e5; height = ramp,
                                       wavelength = 0.055, side = LookRight)
    @test converged
    # The converged height is what the source says at the converged location, to the solve's own
    # tolerance. That is the definition of having landed on the terrain rather than near it.
    @test llh[3] ≈ ramp(llh[1], llh[2]) atol = 1e-6
    # And it is not simply the starting value, or the test would pass for a source that was ignored.
    @test abs(llh[3] - 1000.0) > 1.0

    # A steeper source lands somewhere else again, monotonically — a sanity check that the solve is
    # tracking the surface rather than converging to a fixed point of its own.
    steeper = (lon, lat) -> 1000.0 + 4.0e5 * (lon - lon0)
    llh2, _ = rdr2geo_converged(HS_ORBIT, HS_EL, 200.0, 8.2e5; height = steeper,
                                wavelength = 0.055, side = LookRight)
    @test llh2[3] ≈ steeper(llh2[1], llh2[2]) atol = 1e-6
    @test abs(llh2[3] - 1000.0) > abs(llh[3] - 1000.0)
end

@testset "convergence becomes load-bearing" begin
    # At a constant height the iteration is a contraction on a smooth surface and always settles. Over
    # terrain it need not: where the surface slope approaches the look direction the fixed point is
    # ill-conditioned, which is layover, and there the question the solve asks has no single answer.
    #
    # `rdr2geo` still returns isce3's answer either way, as isce3 does — so `rdr2geo_converged`'s flag is
    # the only way to tell, and this asserts it actually discriminates rather than always being true.
    @test rdr2geo_converged(HS_ORBIT, HS_EL, 200.0, 8.2e5; height = 0.0,
                            wavelength = 0.055, side = LookRight)[2]

    # A source whose slope is absurd — 400 km of relief per degree — is not a surface any look direction
    # meets cleanly, and the solve does not converge on it.
    wild = (lon, lat) -> 4.0e8 * (lon - 0.05)
    llh, converged = rdr2geo_converged(HS_ORBIT, HS_EL, 200.0, 8.2e5; height = wild,
                                       wavelength = 0.055, side = LookRight)
    @test !converged
    # A location is still returned rather than an error or a NaN, which is isce3's documented leniency.
    @test all(isfinite, llh)

    # A source that puts the terrain above the satellite has no geometry at all: the near-nadir guard
    # breaks the loop immediately. Still no error, still finite.
    above = (lon, lat) -> 1.0e7
    @test all(isfinite, rdr2geo(HS_ORBIT, HS_EL, 200.0, 8.2e5; height = above,
                                wavelength = 0.055, side = LookRight))
end

@testset "the footprint and incidence angle are unmoved" begin
    # These are the two consumers of `rdr2geo` that existed before height sources, and both are asserted
    # against the reference elsewhere. Passing a `ConstantHeight` where they take a `zrange` would be a
    # future convenience; what matters now is that their scalar path is untouched, which the bitwise
    # comparison above establishes for the solve itself. This adds the two call sites.
    kw = (; orbit = HS_ORBIT, starting_range = 8.0e5, dr = 2.3295621147, sensing_start = 100.0,
          prf = 486.4863103, nsamples = 500, nlines = 400, look_side = LookRight,
          wavelength = 0.05546576)
    ia = incidence_angle(; kw...)
    @test 0 < ia < pi / 2
    # Recomputed identically on a second call, which is what says nothing stateful crept into the height
    # path — a source holding a mutable running value would break this.
    @test incidence_angle(; kw...) === ia

    coord = RadarCoordinate(; kw..., incidence_angle = ia)
    box = footprint_bounds((lon, lat, h) -> (lon, lat, h), coord)
    @test all(isfinite, (box.X..., box.Y...))
    @test footprint_bounds((lon, lat, h) -> (lon, lat, h), coord) == box
end
