# Building the radar types from a read SAR product.
#
# What matters here is that the conversion carries every quantity across unchanged and computes the two
# derived ones — the epoch offset and the incidence angle — correctly, and that a product whose orbit
# cannot support the geometry is refused rather than extrapolated from.
#
# The product is rebuilt from a fixture SLCDatasets commits, so this needs no granule and no network.

using ImagePairGeometry
using ImagePairGeometry: interpolate
using SLCDatasets
using Dates
using HDF5
using JSON3
using Test

# The fixture and its writer live in SLCDatasets, next to the reader they describe.
const SARD_TEST = joinpath(dirname(dirname(pathof(SLCDatasets))), "test")
include(joinpath(SARD_TEST, "fixture.jl"))

mktempdir() do dir
    path = write_fixture_product(joinpath(dir, "fixture_rslc.h5"))
    s = open_slc(path)
    coord = RadarCoordinate(s)

    @testset "every scalar crosses unchanged" begin
        g = s.geometry
        @test coord.starting_range === g.starting_range
        @test coord.dr === g.range_pixel_spacing
        @test coord.sensing_start === g.sensing_start
        @test coord.prf === g.prf
        @test coord.nsamples == g.nsamples
        @test coord.nlines == g.nlines
        @test coord.wavelength === g.wavelength
        # Each package has its own `LookSide`, so both are named through their module: the whole point
        # of the conversion is mapping one to the other, and an unqualified name would compare a value
        # to itself.
        @test coord.look_side == (g.look_side == SLCDatasets.LookLeft ? ImagePairGeometry.LookLeft :
                                  ImagePairGeometry.LookRight)
        @test coord.look_side isa ImagePairGeometry.LookSide
    end

    @testset "the epoch offset is the epoch's seconds past midnight" begin
        # NISAR's epoch is midnight of the acquisition day, so the offset is zero — but it is computed
        # rather than assumed, since a product on another epoch would silently shift every azimuth
        # index.
        @test coord.orbit_epoch_offset == 0.0
        @test s.geometry.epoch == DateTime(Date(s.geometry.epoch))
    end

    @testset "the orbit is the product's state vectors" begin
        sv = orbit(s)
        orb = coord.orbit
        @test length(orb) == length(sv.time)
        @test orb.position == sv.position
        @test orb.velocity == sv.velocity
        # The interpolant needs a uniform axis; the product supplies one, so the spacing is exact.
        @test orb.spacing === sv.time[2] - sv.time[1]
    end

    @testset "the incidence angle is a plausible SAR geometry" begin
        # Asserted against isce3 bitwise in the radar numerics; here only that it is in range, since a
        # transposed orbit or a wrong look side lands far outside it.
        @test 0 < coord.incidence_angle < pi / 2
        @test 20 < rad2deg(coord.incidence_angle) < 60
    end

    @testset "the interpolated trajectory matches the product" begin
        # At a state vector time the interpolant must return that state vector, which catches a
        # transposed or misordered position array.
        sv = orbit(s)
        i = 3
        p, v = interpolate(coord.orbit, sv.time[i])
        @test all(isapprox.(p, sv.position[i]; rtol = 1e-12))
        @test all(isapprox.(v, sv.velocity[i]; rtol = 1e-12))
    end

    @testset "chebyshev is opt-in and does not change the scalars" begin
        fast = RadarCoordinate(s; chebyshev = true)
        @test fast.starting_range === coord.starting_range
        @test fast.sensing_start === coord.sensing_start
        # The incidence angle is solved for through the orbit, so a different interpolant moves it —
        # by 4e-12 relative here, far below the 1e-8 the package bounds its own `rdr2geo` at.
        @test fast.incidence_angle ≈ coord.incidence_angle rtol = 1e-10
        @test typeof(fast.orbit) !== typeof(coord.orbit)

        # The two interpolants agree to about a centimeter on a real orbit, which is 1e-2 of a range
        # sample and so cannot move an integer output. The bound is 0.1 m rather than the 1.2e-8 m
        # `ImagePairGeometry`'s own documentation reports, because that figure is measured against an
        # analytically circular orbit: a real one is perturbed — this granule's radius varies by a
        # kilometer over 340 s — and an 8-term series fits a perfect circle far better than a real
        # trajectory. Tightening this bound would be asserting a property of synthetic data.
        sv = orbit(s)
        worst = 0.0
        for k in 0:200
            t = first(sv.time) + (last(sv.time) - first(sv.time)) * k / 200
            p1, _ = interpolate(coord.orbit, t)
            p2, _ = interpolate(fast.orbit, t)
            worst = max(worst, maximum(abs, p1 .- p2))
        end
        @test worst < 0.1
        @test worst < coord.dr / 100
    end

    @testset "a pair takes its geometry from the reference alone" begin
        # `testGeogrid.py:427-470` takes every radar parameter from image 1 and the secondary only for
        # the interval, so a pair of one acquisition with itself has dt zero — which is refused.
        @test_throws "acquisition order" CoregisteredPair(s, s)
        @test repeat_interval(s, s) == 0.0
    end
end

@testset "a pair carries the reference's geometry and the interval" begin
    # `repeat_interval` itself is SLCDatasets'; what matters here is that the pair takes it and takes its
    # geometry from image 1 alone.
    mktempdir() do dir
        a = open_slc(write_fixture_product(joinpath(dir, "a.h5")))
        later = override(FIXTURE,
                         (:geometry => :epoch) => "seconds since 2025-12-15T00:00:00",
                         (:orbit => :epoch) => "seconds since 2025-12-15T00:00:00")
        b = open_slc(write_fixture_product(joinpath(dir, "b.h5"), later))

        dt = repeat_interval(a, b)
        @test dt ≈ 48 * 86400 atol = 1.0
        pair = CoregisteredPair(a, b)
        @test pair.dt === dt
        # The pair's geometry is the reference's, not a blend of the two.
        @test pair.coordinate.starting_range === a.geometry.starting_range
    end
end

@testset "a product whose orbit misses its acquisition is refused" begin
    # An out-of-range solve extrapolates rather than failing, so this is caught at construction.
    mktempdir() do dir
        early = override(FIXTURE,
                         (:geometry => :sensing_start) => hx(0.0),
                         (:geometry => :sensing_stop) => hx(1.0))
        s = open_slc(write_fixture_product(joinpath(dir, "early.h5"), early))
        @test_throws "does not cover it" RadarCoordinate(s)
    end
end

@testset "a product splitting its two clocks is refused" begin
    # Every product measured puts the azimuth times and the state vectors on one epoch. One that did
    # not would need a conversion this does not implement, so it says so rather than differencing two
    # scales.
    mktempdir() do dir
        split = override(FIXTURE, (:orbit => :epoch) => "seconds since 2025-10-27T00:00:00")
        s = open_slc(write_fixture_product(joinpath(dir, "split.h5"), split))
        @test_throws "Converting between them is not implemented" RadarCoordinate(s)
    end
end
