# Gate: what the lattice approximation costs in accuracy.
#
# Compared against the *exact* path, not against the reference. The exact path is already gated
# against the reference by `test/geogrid.jl`, so comparing to it isolates what the interpolation
# itself does and keeps the reference tolerances measuring only what they were set for — PROJ's
# platform variation and compiler contraction.
#
# The two modes carry different contracts, and both are asserted:
#
# `:hybrid` keeps the forward transform exact, so every integer band must be bitwise identical. That
# is not a tolerance to be tuned; it is the property that makes the mode worth having, since
# `location_x`/`location_y` are Tier A bands.
#
# `:full` interpolates both directions, so a point whose exact fractional pixel index sits within the
# interpolation error of a `.5` boundary can round the other way. The bound asserted is one pixel on
# the location bands and nothing on the rest.

using ImagePairGeometry
using ImagePairGeometry: INT_BANDS, FLOAT_BANDS, nodata_from, build_lattice, latticehalo,
                         _resolve_transform
using Extents
using Proj
using Test

const KERNELS = (NearestNode(), Bilinear(), Bicubic())

function interp_run(k, tf)
    return pairgeometry(k.grid, k.pair, k.inputs; transform = tf, window = k.win,
                        params = k.params, nodata = nodata_from(-32767.0))
end

"""Worst absolute difference over a band, divided by the band's own scale.

Normalized by the band maximum rather than per element: `off2vx_dy` passes through zero, so a
pointwise relative error there reports how near zero the denominator got, not how wrong the value is.
Points that are nodata on either side are excluded — a sentinel is not a magnitude.
"""
function band_error(got::PairGeometry, want::PairGeometry, f::Symbol)
    g, w = getfield(got, f), getfield(want, f)
    m = (w .!= -32767.0) .& (g .!= -32767.0)
    any(m) || return 0.0
    scale = maximum(abs, w[m])
    iszero(scale) && return maximum(abs, g[m] .- w[m])
    return maximum(abs.(g[m] .- w[m])) / scale
end

@testset "lattice kernels are exact on an affine transform" begin
    # An affine map is reproduced exactly by linear and cubic interpolation alike, so this pins the
    # kernels' arithmetic at any spacing with no PROJ involved and no reference to compare against —
    # the same role `AffineTransform` plays for the geometry kernel.
    t = AffineTransform(a = 0.6, b = -0.8, c = 1000.0, d = 0.8, e = 0.6, f = -500.0)
    b = Extent(X = (0.0, 1000.0), Y = (0.0, 1000.0))
    query = [(x, y) for x in 0.0:37.0:1000.0, y in 0.0:41.0:1000.0]

    for m in (Bilinear(), Bicubic()), spacing in ((120.0, 120.0), (480.0, 480.0))
        L = build_lattice(t, b, spacing, m)
        worst = 0.0
        for (x, y) in query
            gx, gy, gz = L(x, y, 300.0)
            ex, ey, _ = t(x, y, 300.0)
            worst = max(worst, abs(gx - ex), abs(gy - ey))
            @test gz == 300.0
        end
        # Bounded by rounding on coordinates of order 1e3, not by the interpolation.
        @test worst < 1e-9
    end

    # `NearestNode` must *not* reproduce it: a kernel that passed this test would mean the query
    # points all landed on nodes, making the other two assertions vacuous.
    L = build_lattice(t, b, (120.0, 120.0), NearestNode())
    @test maximum(abs(L(x, y, 300.0)[1] - t(x, y, 300.0)[1]) for (x, y) in query) > 1.0
end

@testset "lattice passes z and NaN through" begin
    t = AffineTransform(a = 1.0, b = 0.0, c = 0.0, d = 0.0, e = 1.0, f = 0.0, scale_z = 3.0)
    L = build_lattice(t, Extent(X = (0.0, 500.0), Y = (0.0, 500.0)), (100.0, 100.0), Bilinear())
    # The kernel takes `z` from the point it started with, so the lattice returns the queried
    # elevation rather than the transform's scaled one.
    @test L(250.0, 250.0, 42.0)[3] == 42.0
    # A `NaN` coordinate is a value the exact path propagates to a `NaN` pixel index, not a bounds
    # error: `REFERENCE.md` records that the reference reports such a point in bounds.
    @test all(isnan, L(NaN, 250.0, 0.0)[1:2])
    @test all(isnan, L(250.0, NaN, 0.0)[1:2])

    # This transform's horizontal part ignores `z`, so both levels tabulate identically and a query
    # interpolates one of them. The result must not depend on the elevation asked for.
    @test L.flat
    @test L(250.0, 250.0, -200.0)[1:2] == L(250.0, 250.0, 4000.0)[1:2]

    # Only one level is tabulated for it, since a second would duplicate the first: that halves the
    # transform calls the lattice costs to build, which for a same-datum pair is every real case.
    @test size(L.x, 3) == 1

    # A transform whose horizontal result does move with elevation gets both levels, and then a query
    # blends them: at the midpoint it sits halfway between.
    shear(x, y, z) = (x + z, y, z)
    S = build_lattice(shear, Extent(X = (0.0, 500.0), Y = (0.0, 500.0)), (100.0, 100.0), Bilinear();
                      zrange = (0.0, 1000.0))
    @test size(S.x, 3) == 2
    @test !S.flat
    @test S(250.0, 250.0, 0.0)[1] ≈ 250.0
    @test S(250.0, 250.0, 1000.0)[1] ≈ 1250.0
    @test S(250.0, 250.0, 500.0)[1] ≈ 750.0

    # A shift that vanishes at the origin and grows with position still counts as z-dependent: the
    # probe samples the lattice's corners and center, not one point.
    skew(x, y, z) = (x + z * (x / 500.0), y, z)
    K = build_lattice(skew, Extent(X = (0.0, 500.0), Y = (0.0, 500.0)), (100.0, 100.0), Bilinear();
                      zrange = (0.0, 1000.0))
    @test size(K.x, 3) == 2
    @test !K.flat

    # A degenerate range has nothing to interpolate along, so one level regardless of the transform.
    D = build_lattice(shear, Extent(X = (0.0, 500.0), Y = (0.0, 500.0)), (100.0, 100.0), Bilinear();
                      zrange = (500.0, 500.0))
    @test size(D.x, 3) == 1
    @test D.flat
    @test D(250.0, 250.0, 500.0)[1] ≈ 750.0
end

@testset "a query outside the lattice throws" begin
    t = AffineTransform(a = 1.0, b = 0.0, c = 0.0, d = 0.0, e = 1.0, f = 0.0)
    for m in KERNELS
        L = build_lattice(t, Extent(X = (0.0, 500.0), Y = (0.0, 500.0)), (100.0, 100.0), m)
        # Clamping instead would return a plausible coordinate for an out-of-region query, which is
        # the failure mode the bitwise gates exist to catch.
        @test_throws "outside the lattice" L(1e6, 250.0, 0.0)
        @test_throws "outside the lattice" L(250.0, -1e6, 0.0)
    end
end

@testset "build_lattice validation" begin
    t = AffineTransform(a = 1.0, b = 0.0, c = 0.0, d = 0.0, e = 1.0, f = 0.0)
    b = Extent(X = (0.0, 500.0), Y = (0.0, 500.0))
    @test_throws "spacing must be positive" build_lattice(t, b, (0.0, 100.0), Bilinear())
    @test_throws "spacing must be positive" build_lattice(t, b, (-100.0, 100.0), Bilinear())
    @test_throws "bounds are empty" build_lattice(t, Extent(X = (500.0, 0.0), Y = (0.0, 1.0)),
                                                 (100.0, 100.0), Bilinear())
    # Bicubic's wider stencil needs a wider margin, so it builds more nodes over the same bounds.
    @test latticehalo(Bicubic()) > latticehalo(Bilinear())
    @test all(latticesize(build_lattice(t, b, (100.0, 100.0), Bicubic())) .>
              latticesize(build_lattice(t, b, (100.0, 100.0), Bilinear())))
end

@testset "InterpolatedTransform validation" begin
    k = setup_case("cross_crs")
    @test_throws "lattice must be at least 1" InterpolatedTransform(
        k.makepair, k.grid, k.pair; lattice = 0, window = k.win)
    @test_throws "mode must be one of" InterpolatedTransform(
        k.makepair, k.grid, k.pair; mode = :approximate, window = k.win)
    @test_throws "zrange must be ordered" InterpolatedTransform(
        k.makepair, k.grid, k.pair; zrange = (100.0, -100.0), window = k.win)
end

@testset ":hybrid keeps the location bands bitwise" begin
    # The mode's justification, and it is specific to these two bands: `location_x`/`location_y` come
    # from the forward transform alone, so leaving that exact keeps them bitwise however coarse the
    # lattice, and they are the Tier A bands that gate agreement with the reference.
    #
    # The distinction matters. `offset` and `search` are *not* in that position — both go through the
    # displacement-to-velocity conversion, which is built from the axis unit vectors the inverse
    # transform produces. So they carry the interpolation error, and at a coarse lattice a point whose
    # exact value sits near a rounding boundary can tip by one. Measured on these fixtures that first
    # appears at 8x with `Bilinear`, and not at all through 16x with `Bicubic`; a scene with more
    # points near a tie would show it sooner, so the assertion is a bound of one rather than equality.
    for name in ("cross_crs", "cross_crs_3031", "cross_crs_nodata", "ratio_15m")
        k = setup_case(name)
        exact = interp_run(k, k.makepair)
        for lattice in (1, 2, 4, 8), interpolation in (Bilinear(), Bicubic())
            tf = InterpolatedTransform(k.makepair, k.grid, k.pair;
                                       lattice, mode = :hybrid, interpolation, window = k.win)
            got = interp_run(k, tf)
            @testset "$name lattice=$lattice $(typeof(interpolation))" begin
                @test got.location_x == exact.location_x
                @test got.location_y == exact.location_y
                # Which points are computed at all follows from the location, so it cannot move.
                @test nvalid(got) == nvalid(exact)
                for f in (:stable_surface, :chip_min_x, :chip_min_y, :chip_max_x, :chip_max_y)
                    @test getfield(got, f) == getfield(exact, f)
                end
                for f in (:offset_x, :offset_y, :search_x, :search_y)
                    g, w = getfield(got, f), getfield(exact, f)
                    @test maximum(abs.(Int.(g) .- Int.(w))) <= 1
                end
            end
        end
    end
end

@testset ":hybrid float bands at a stated bound" begin
    # A bound of its own rather than the reference's Tier B: Tier B absorbs PROJ's platform variation
    # on the exact path and should keep measuring only that. This one measures the lattice.
    #
    # The error falls as the square of the spacing, as bilinear interpolation of a smooth function
    # must, so the bound is stated per spacing rather than as one number that would be slack at 2x
    # and tight at 8x. `docs/interpolated-transform.md` tabulates the measured values.
    bounds = Dict(1 => 1e-4, 2 => 4e-4, 4 => 1.6e-3, 8 => 6.4e-3)
    k = setup_case("cross_crs")
    exact = interp_run(k, k.makepair)
    for lattice in (1, 2, 4, 8)
        tf = InterpolatedTransform(k.makepair, k.grid, k.pair;
                                   lattice, mode = :hybrid, window = k.win)
        got = interp_run(k, tf)
        for f in FLOAT_BANDS
            @test band_error(got, exact, f) < bounds[lattice]
        end
    end
end

@testset ":full shifts a location band by at most one pixel" begin
    # What rules `:full` out as the default, asserted as a bound rather than left as prose. The
    # positional error is far below a pixel; what changes is which side of a rounding boundary a
    # point falls on, so the difference is exactly one pixel where it appears at all.
    k = setup_case("cross_crs")
    exact = interp_run(k, k.makepair)
    sentinel = Int32(-32767)
    for lattice in (2, 4, 8)
        tf = InterpolatedTransform(k.makepair, k.grid, k.pair;
                                   lattice, mode = :full, window = k.win)
        got = interp_run(k, tf)
        @testset "lattice=$lattice" begin
            for f in (:location_x, :location_y)
                g, w = getfield(got, f), getfield(exact, f)
                # A point valid on one side and sentinel on the other is a validity flip, not a
                # shift, and would otherwise register as a difference of 32767.
                both = (g .!= sentinel) .& (w .!= sentinel)
                @test maximum(abs.(Int.(g[both]) .- Int.(w[both]))) <= 1
                # A point may cross the image edge, but only where it sat on that edge already.
                @test count((g .!= sentinel) .!= (w .!= sentinel)) <= 8
            end
            # The chip-size bounds and the mask are read from their own rasters and converted with the
            # image's pixel size, touching neither transform, so no lattice can move them.
            for f in (:stable_surface, :chip_min_x, :chip_min_y, :chip_max_x, :chip_max_y)
                @test getfield(got, f) == getfield(exact, f)
            end
            # `offset` and `search` go through the inverse transform, so they tip by at most one for
            # the same reason the location bands do.
            for f in (:offset_x, :offset_y, :search_x, :search_y)
                g, w = getfield(got, f), getfield(exact, f)
                both = (g .!= sentinel) .& (w .!= sentinel)
                @test maximum(abs.(Int.(g[both]) .- Int.(w[both]))) <= 1
            end
        end
    end
end

@testset "a lattice at the grid spacing is exact where nodes coincide" begin
    # At `lattice = 1` a node sits at every grid *point*, but the kernel queries the inverse a pixel
    # off those points, so only the forward lattice is node-aligned. Checking it directly keeps the
    # assertion about node alignment rather than about the whole pipeline.
    k = setup_case("cross_crs")
    tf = _resolve_transform(k.makepair)
    dx, dy = abs.(ImagePairGeometry.gridspacing(k.grid))
    L = build_lattice(tf.forward, ImagePairGeometry._grid_bounds(k.grid, k.win), (dx, dy),
                      Bilinear())
    for idx in k.win[1:37:end]
        gx, gy = ImagePairGeometry.gridpoint_center(k.grid, idx.I...)
        ex, ey, _ = tf.forward(gx, gy, 0.0)
        gx′, gy′, _ = L(gx, gy, 0.0)
        # The node's own value, up to the rounding of locating the cell it sits on the corner of.
        @test abs(gx′ - ex) < 1e-6
        @test abs(gy′ - ey) < 1e-6
    end
end

@testset "an identity pair needs no lattice" begin
    # A same-CRS case dispatches to the closed-form `pointgeometry`, which is exact and makes no PROJ
    # call, so wrapping it in a lattice can only lose precision for no gain. It still has to work.
    k = setup_case("same_crs")
    @test k.same
    exact = interp_run(k, k.makepair)
    tf = InterpolatedTransform(k.makepair, k.grid, k.pair;
                               lattice = 4, mode = :full, window = k.win)
    got = interp_run(k, tf)
    for f in INT_BANDS
        @test getfield(got, f) == getfield(exact, f)
    end
end

@testset "blocking and threading change nothing" begin
    # The invariant `pairgeometry_blocked` documents: a window computed in pieces gives the same bits
    # as the same window computed whole. A lattice sized from a *block* would break it, so this is
    # what asserts the lattice is built from the whole window instead.
    k = setup_case("cross_crs")
    source = InMemoryInputs(k.inputs, k.win)
    ref = nothing
    for mode in (:hybrid, :full), blocksize in ((512, 512), (97, 211)), ntasks in (1, 4)
        tf = InterpolatedTransform(k.makepair, k.grid, k.pair;
                                   lattice = 4, mode, window = k.win)
        got = pairgeometry_blocked(k.grid, k.pair, source; transform = tf, window = k.win,
                                   blocksize, ntasks, params = k.params,
                                   nodata = nodata_from(-32767.0))
        if blocksize == (512, 512) && ntasks == 1
            ref = got
            continue
        end
        for f in INT_BANDS
            @test getfield(got, f) == getfield(ref, f)
        end
        for f in FLOAT_BANDS
            @test reinterpret(UInt64, getfield(got, f)) == reinterpret(UInt64, getfield(ref, f))
        end
    end
end

@testset "type stability" begin
    t = AffineTransform(a = 0.6, b = -0.8, c = 1000.0, d = 0.8, e = 0.6, f = -500.0)
    b = Extent(X = (0.0, 1000.0), Y = (0.0, 1000.0))
    for m in KERNELS
        L = build_lattice(t, b, (120.0, 120.0), m)
        @test @inferred(L(500.0, 500.0, 300.0)) isa NTuple{3,Float64}
        # An `Int` query must not widen the result: the kernel passes whatever the grid holds.
        @test @inferred(L(500, 500, 300)) isa NTuple{3,Float64}
    end
    # The factory itself is deliberately not required to be inferrable: `mode` is a runtime field, so
    # the two modes give two different `TransformPair` types. It is called once per task, and what the
    # per-point loop specializes on is the concrete pair it returns — so that is what is asserted.
    k = setup_case("cross_crs")
    for mode in (:hybrid, :full)
        pair = InterpolatedTransform(k.makepair, k.grid, k.pair;
                                     lattice = 4, mode, window = k.win)()
        @test isconcretetype(typeof(pair))
        @test pair isa TransformPair
        @test pair.inverse isa CoordLattice{Bilinear}
    end
    @test InterpolatedTransform(k.makepair, k.grid, k.pair;
                                lattice = 4, mode = :full, window = k.win)().forward isa
          CoordLattice{Bilinear}
end
