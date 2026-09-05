# The FastGeoProjections transform, against PROJ and against the reference fixtures.
#
# PROJ is what the fixtures are generated and asserted against, so the question here is not whether
# this transform is correct in isolation but whether substituting it changes a result the reference
# pins. Two things are asserted separately, because they are true to different degrees:
#
#   integer bands — bitwise. They pass through a rounding conversion, and the 2.4e-8 m that separates
#                   the two transforms is far too small to move one.
#   float bands   — relative, to the ≤1e-7 `REFERENCE.md` sets for the cross-CRS group. These carry
#                   the difference directly, amplified: the axis vectors are differences of nearby
#                   transformed points, so a coordinate error enters them relative to a ~10 m step
#                   rather than to a ~1e6 m coordinate. Measured worst case is 3.7e-9.
#
# The same-CRS cases never reach this code: `fast_transform` returns an `IdentityTransform` for equal
# codes, as `fixture_scene` does, so they are exercised here only to assert that.

using ImagePairGeometry
using ImagePairGeometry: INT_BANDS, FLOAT_BANDS, IdentityTransform, TransformPair,
                         nodata_from, transform_pair, inverse
using FastGeoProjections
using Proj
using Test

const FGPExt = Base.get_extension(ImagePairGeometry, :ImagePairGeometryFastGeoProjectionsExt)
using .FGPExt: FastTransform

# The cross-CRS fixture cases. The same-CRS ones dispatch to `IdentityTransform` on both paths, so
# they hold nothing to compare.
const CROSS_CASES = ("cross_crs", "cross_crs_3031", "cross_crs_nodata")

@testset "fast_transform is a TransformPair over EPSG codes" begin
    tf = fast_transform(3413, 32624)
    @test tf isa TransformPair
    @test tf.forward isa FastTransform
    @test tf.inverse isa FastTransform

    # Accepts EPSG objects as well as integers, and gives the same transform either way.
    p = fast_transform(EPSG(3413), EPSG(32624))
    @test p.forward(-1.5e5, -2.2e6, 1500.0) == tf.forward(-1.5e5, -2.2e6, 1500.0)

    # A CRS given any other way has no code to look up.
    @test_throws "takes EPSG codes as integers or EPSG objects" fast_transform("EPSG:3413", "EPSG:32624")
    @test_throws "use proj_transform for those" fast_transform("EPSG:3413", "EPSG:32624")

    # `inverse` is an involution, so a pair cannot be assembled inconsistently.
    @test inverse(inverse(tf.forward)) == tf.forward
end

@testset "the third coordinate is carried, not transformed" begin
    tf = fast_transform(3413, 32624)
    # Both CRSs share a datum, so a height is not a quantity either projection changes.
    for z in (0.0, 1500.0, -30.0)
        @test tf.forward(-1.5e5, -2.2e6, z)[3] === z
        @test tf.inverse(5.0e5, 7.5e6, z)[3] === z
    end
    # And x, y do not depend on it.
    a = tf.forward(-1.5e5, -2.2e6, 0.0)
    b = tf.forward(-1.5e5, -2.2e6, 9999.0)
    @test a[1] == b[1] && a[2] == b[2]
end

@testset "equal codes give the identity, not a projection round trip" begin
    # PROJ resolves a same-CRS pair to `+proj=noop` and returns its input unchanged, and the
    # same-CRS fixtures asserted against it are bitwise on every band. Composing a projection with
    # its own inverse instead lands tens of nanometres away, which is why this is special-cased.
    for code in (32624, 3413, 3031)
        tf = fast_transform(code, code)
        @test tf.forward isa IdentityTransform
        @test tf.inverse isa IdentityTransform
        for p in ((1.0, 2.0, 3.0), (-1.5e5, -2.2e6, 1500.0))
            @test tf.forward(p...) === p
        end
    end
end

@testset "agrees with PROJ on the transform itself: $a -> $b" for (a, b) in
        ((3413, 32624), (32624, 3413), (3031, 32719), (32719, 3031))
    fast = fast_transform(a, b).forward
    slow = Proj.Transformation("EPSG:$a", "EPSG:$b")
    pts = a in (3413, 3031) ? ((-1.5e5, -2.2e6), (0.0, -2.0e6), (3.0e5, -1.5e6)) :
                              ((5.0e5, 7.0e6), (4.0e5, 6.5e6), (6.0e5, 7.5e6))
    for (x, y) in pts
        fx, fy, _ = fast(x, y, 0.0)
        sx, sy = slow(x, y)
        # Absolute metres, since these are projected coordinates of order 1e6 m and the quantity
        # that matters downstream is a distance on the ground, not a ratio.
        @test abs(fx - sx) < 1e-6
        @test abs(fy - sy) < 1e-6
    end
end

@testset "fixture case $name: bands against the PROJ path" for name in CROSS_CASES
    s = setup_case(name)
    nd = nodata_from(-32767.0)

    slow = pairgeometry(s.grid, s.pair, s.inputs; transform = s.makepair(),
                        window = s.win, params = s.params, nodata = nd)
    fast = pairgeometry(s.grid, s.pair, s.inputs;
                        transform = fast_transform(s.grid_epsg, s.image_epsg),
                        window = s.win, params = s.params, nodata = nd)

    @test size(fast) == size(slow)
    # Every point the PROJ path found in bounds, this one finds too: a 2.4e-8 m shift cannot move a
    # point across an image edge that a whole pixel is 10 m or more wide.
    @test nvalid(fast) == nvalid(slow)

    # Rounding absorbs the difference, so these stay exact.
    for f in INT_BANDS
        @test getfield(fast, f) == getfield(slow, f)
    end

    # The Float64 bands carry it. `REFERENCE.md` bounds the cross-CRS group at 1e-7 relative.
    for f in FLOAT_BANDS
        x, y = getfield(fast, f), getfield(slow, f)
        finite = findall(k -> isfinite(x[k]) && isfinite(y[k]), eachindex(x, y))
        # A band left entirely at its sentinel has nothing to compare, and `isapprox` on an empty
        # selection would pass vacuously either way.
        isempty(finite) && continue
        @test x[finite] ≈ y[finite] rtol = 1e-7
        # NaN and Inf must land in the same places, which `≈` over the finite subset cannot see.
        @test map(isfinite, x) == map(isfinite, y)
    end
end

@testset "one transform serves every thread" begin
    # The PROJ path needs a factory because a `PJ*` is single-thread-only. A native transformation is
    # immutable and holds no PROJ state, so a blocked run can pass one object directly — this is what
    # makes that safe, and it is why `fast_transform` returns a pair rather than a factory.
    s = setup_case("cross_crs")
    nd = nodata_from(-32767.0)
    tf = fast_transform(s.grid_epsg, s.image_epsg)

    whole = pairgeometry(s.grid, s.pair, s.inputs; transform = tf, window = s.win,
                         params = s.params, nodata = nd)
    src = InMemoryInputs(s.inputs, s.win)

    # Shared across tasks, at block sizes that divide the window and ones that do not.
    for bs in ((7, 13), (16, 16), (1000, 1000))
        blocked = pairgeometry_blocked(s.grid, s.pair, src; transform = tf, window = s.win,
                                       blocksize = bs, params = s.params, nodata = nd)
        for f in INT_BANDS
            @test getfield(blocked, f) == getfield(whole, f)
        end
        for f in FLOAT_BANDS
            # Bit patterns: blocking must change nothing at all, and `-0.0 == 0.0` would hide a
            # difference that a later comparison against the reference would not.
            @test reinterpret(UInt64, getfield(blocked, f)) ==
                  reinterpret(UInt64, getfield(whole, f))
        end
    end
end

@testset "a shared transform is called identically from many threads" begin
    tf = fast_transform(3413, 32624).forward
    xs = collect(range(-3.0e5, 1.0e5; length = 20_000))
    serial = [tf(x, -2.0e6, 1500.0) for x in xs]
    threaded = Vector{NTuple{3,Float64}}(undef, length(xs))
    Threads.@threads for k in eachindex(xs)
        threaded[k] = tf(xs[k], -2.0e6, 1500.0)
    end
    @test threaded == serial
end
