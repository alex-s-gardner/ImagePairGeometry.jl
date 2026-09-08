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
using SLCDatasets: nbursts
using Dates
using HDF5
using JSON3
using Test

# The fixtures and their writers live in SLCDatasets, next to the readers they describe.
const SLCD_TEST = joinpath(dirname(dirname(pathof(SLCDatasets))), "test")
include(joinpath(SLCD_TEST, "fixture.jl"))

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

    @testset "the azimuth times are already on the orbit's scale" begin
        # `RadarGeometry` reports both clocks against one epoch, so the offset between them is zero.
        # NISAR cannot distinguish that from the epoch's time of day, because its epoch *is* midnight —
        # the Sentinel-1 testset below is what separates the two.
        @test coord.orbit_epoch_offset == 0.0
        @test s.geometry.epoch == DateTime(Date(s.geometry.epoch))
        sv = orbit(s)
        @test first(sv.time) <= coord.sensing_start <= last(sv.time)
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

        # The secondary's own coordinate comes across too, which is what `pixel_offset` needs and what a
        # pair built from a single acquisition cannot supply. It must be the *secondary's* geometry:
        # filling it from the reference would make the offset identically zero and look like agreement.
        sec = pair.secondary_offset_coordinate
        @test sec isa RadarCoordinate
        @test sec.sensing_start === b.geometry.sensing_start
        @test sec.starting_range === b.geometry.starting_range
        @test sec.prf === b.geometry.prf
        # These two products differ only by epoch, so their `sensing_start` values agree on their own
        # clocks and the 48 days live in the epochs. That is exactly the case where reading one clock for
        # the other passes unnoticed, so the orbits are asserted distinct rather than the times.
        @test sec.orbit !== pair.coordinate.orbit
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


# Sentinel-1, which is the case that distinguishes a correct epoch conversion from a lucky one. NISAR
# puts its epoch at midnight, so `sensing_start` on the azimuth-index scale and on the orbit scale are
# numerically equal and any offset between zero and the epoch's time of day passes. Sentinel-1's epoch is
# the middle of the day, so the two differ by hours and only the right answer lands on the orbit.
@testset "Sentinel-1" begin
    # The sensor that distinguishes a correct epoch conversion from a lucky one. NISAR puts its epoch at
    # midnight, so `sensing_start` on the azimuth-index scale and on the orbit scale are numerically
    # equal and any offset between zero and the epoch's time of day passes. Sentinel-1's epoch is the
    # middle of the day, so the two differ by hours and only the right answer lands on the orbit.
    #
    # Built by SLCDatasets' own fixture writer, so this depends on files in another repository. Both the
    # writer and the inputs it takes are checked for rather than assumed: a version of that package
    # without them should skip this, not error.
    writer = joinpath(SLCD_TEST, "sentinel1_fixture.jl")
    inputs_a = joinpath(SLCD_TEST, "reference", "sentinel1_inputs.json")
    inputs_b = joinpath(SLCD_TEST, "reference", "sentinel1_inputs_s1b.json")
    if !isfile(writer) || !isfile(inputs_a)
        @info """skipping the Sentinel-1 conversion; this SLCDatasets has no committed S1 fixture
                 writer and inputs""" writer = isfile(writer) inputs = isfile(inputs_a)
    else
        include(writer)
        mktempdir() do dir
            safe_a, eof_a = write_s1_fixture(mkpath(joinpath(dir, "a")),
                                             JSON3.read(read(inputs_a, String)))
            s1 = open_slc(safe_a; orbit = eof_a)
            g = s1.geometry
            sv = orbit(s1)
            coord = RadarCoordinate(s1)
            mosaic_epoch = g.epoch

            # The premise that makes this a test rather than a repeat of the NISAR case.
            @testset "the premise: this epoch is not midnight" begin
                @test g.epoch != DateTime(Date(g.epoch))
                @test SLCDatasets.epoch_offset(g) > 3600
            end

            # The same checks the NISAR case makes, so neither sensor is covered more thinly than the
            # other. What differs is only the expected values.
            @testset "every scalar crosses unchanged" begin
                @test coord.starting_range === g.starting_range
                @test coord.dr === g.range_pixel_spacing
                @test coord.sensing_start === g.sensing_start
                @test coord.prf === g.prf
                @test coord.nsamples == g.nsamples
                @test coord.nlines == g.nlines
                @test coord.wavelength === g.wavelength
                # Right-looking, where NISAR is left: the one field whose value differs by sensor rather
                # than by acquisition, and the reason it is read rather than assumed.
                @test coord.look_side == ImagePairGeometry.LookRight
                @test coord.look_side isa ImagePairGeometry.LookSide
            end

            @testset "the azimuth times land on the orbit" begin
                # The bug this guards against added the epoch's time of day to `sensing_start`, pushing
                # every solve hours outside the state vectors. It threw rather than returning a wrong
                # answer, but only for a product whose epoch is not midnight.
                @test coord.orbit_epoch_offset == 0.0
                t0 = coord.sensing_start + coord.orbit_epoch_offset
                t1 = coord.sensing_start + (coord.nlines - 1) / coord.prf + coord.orbit_epoch_offset
                @test first(sv.time) <= t0 <= last(sv.time)
                @test first(sv.time) <= t1 <= last(sv.time)
            end

            @testset "the orbit is the product's state vectors" begin
                @test length(coord.orbit) == length(sv.time)
                @test coord.orbit.position == sv.position
                @test coord.orbit.velocity == sv.velocity
                @test coord.orbit.spacing === sv.time[2] - sv.time[1]
            end

            @testset "the interpolated trajectory matches the product" begin
                # At a state vector time the interpolant must return that state vector, which catches a
                # transposed or misordered position array.
                i = 3
                p, v = interpolate(coord.orbit, sv.time[i])
                @test all(isapprox.(p, sv.position[i]; rtol = 1e-12))
                @test all(isapprox.(v, sv.velocity[i]; rtol = 1e-12))
            end

            @testset "the ground pixel sizes are IW-like" begin
                # C-band IW: a few metres in ground range, around fifteen along track. A wrong look side
                # or a transposed orbit puts the incidence angle far outside this.
                @test 0 < coord.incidence_angle < pi / 2
                @test 25 < rad2deg(coord.incidence_angle) < 50
                @test 2 < ImagePairGeometry.xsize(coord) < 8
                @test 10 < ImagePairGeometry.ysize(coord) < 25
            end

            @testset "chebyshev is opt-in and does not change the scalars" begin
                fast = RadarCoordinate(s1; chebyshev = true)
                @test fast.starting_range === coord.starting_range
                @test fast.sensing_start === coord.sensing_start
                @test typeof(fast.orbit) !== typeof(coord.orbit)
                @test fast.incidence_angle ≈ coord.incidence_angle rtol = 1e-10
            end

            @testset "a pair takes its geometry from the reference alone" begin
                @test_throws "acquisition order" CoregisteredPair(s1, s1)
                @test repeat_interval(s1, s1) == 0.0
            end

            # A single burst. SLCDatasets asserts every burst's metadata against isce3, but nothing
            # there builds geometry from one, and a burst is not a smaller mosaic: it carries its own
            # epoch — ten seconds later than the mosaic's in this product — its own trimmed sensing
            # window, and its own slant range origin. So the epoch conversion takes a different input
            # here than in any test above.
            @testset "a single burst" begin
                nsw = nbursts(safe_a; swath = 2)
                @test nsw > 1
                mosaic = coord

                for i in (1, nsw ÷ 2 + 1, nsw)
                    bs = open_slc(safe_a; orbit = eof_a, swath = 2, burst = i)
                    bg = bs.geometry
                    bsv = orbit(bs)
                    bc = RadarCoordinate(bs)

                    # The burst's own epoch, not the mosaic's, and not midnight either — so this
                    # exercises the conversion rather than inheriting a value that happens to work.
                    @test bg.epoch != mosaic_epoch
                    @test bg.epoch != DateTime(Date(bg.epoch))
                    @test bc.orbit_epoch_offset == 0.0

                    # A burst is a slice of the swath in azimuth, so it is shorter and its sensing
                    # window sits inside the mosaic's.
                    @test bg.nlines < mosaic.nlines
                    @test bg.sensing_stop - bg.sensing_start < mosaic.sensing_start +
                          (mosaic.nlines - 1) / mosaic.prf - mosaic.sensing_start

                    # The state vectors bracket this burst, which is what makes a burst usable alone.
                    t0 = bc.sensing_start
                    t1 = bc.sensing_start + (bc.nlines - 1) / bc.prf
                    @test first(bsv.time) <= t0 <= last(bsv.time)
                    @test first(bsv.time) <= t1 <= last(bsv.time)
                    @test interpolate(bc.orbit, t0) isa Tuple
                    @test interpolate(bc.orbit, t1) isa Tuple

                    # And the geometry is still Sentinel-1's, at the same look side and a plausible
                    # incidence angle — a burst read with a neighbour's timing would move this.
                    @test bc.look_side == ImagePairGeometry.LookRight
                    @test 25 < rad2deg(bc.incidence_angle) < 50
                    @test 2 < ImagePairGeometry.xsize(bc) < 8
                    @test 10 < ImagePairGeometry.ysize(bc) < 25
                end
            end

            # A second acquisition, so the pair path is exercised on this sensor too rather than only on
            # NISAR. The S1B fixture is a different platform on a different date, which is what a real
            # pair is.
            if isfile(inputs_b)
                @testset "a pair of two acquisitions" begin
                    safe_b, eof_b = write_s1_fixture(mkpath(joinpath(dir, "b")),
                                                     JSON3.read(read(inputs_b, String)))
                    s1b = open_slc(safe_b; orbit = eof_b)
                    dt = repeat_interval(s1, s1b)
                    @test dt != 0
                    early, late = dt > 0 ? (s1, s1b) : (s1b, s1)
                    pair = CoregisteredPair(early, late)
                    @test pair.dt > 0
                    @test pair.dt === abs(dt)
                    @test pair.coordinate.starting_range === early.geometry.starting_range
                    @test pair.coordinate.look_side == ImagePairGeometry.LookRight
                end
            else
                @info "skipping the Sentinel-1 pair; no S1B inputs committed"
            end
        end
    end
end
