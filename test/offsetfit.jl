# The offset field as a polynomial, and the order selection that fixes its degree.
#
# The claim under test is that the *order is chosen by measurement*, not named in advance. So the cases
# that matter are: a field of known degree is recovered at that degree and not below it; a target that no
# order meets is refused rather than fitted around; and a sampling too sparse to constrain the terms is
# refused rather than silently interpolated.
#
# The real pair is the case the plan was written to answer. First order — the "affine model" this work
# started from — leaves 0.03 samples on the smallest-baseline pair available, and second order reaches
# 1.4e-4. That is why the order is a fit parameter.

using ImagePairGeometry
using ImagePairGeometry: MapGrid, CoregisteredPair, RadarCoordinate, Orbit, LookRight,
                         incidence_angle, Ellipsoid, xyz_to_lonlat, pixel_offset, fit_offset,
                         offset_fit_terms, OffsetFit, OFFSET_FIT_MIN_NODES_PER_TERM,
                         OFFSET_FIT_MAX_ORDER
using StaticArrays: SVector
using JSON3
using Test

_grid() = MapGrid(geotransform = (0.0, 100.0, 0.0, 5000.0, 0.0, -100.0), size = (60, 60))
_win() = CartesianIndices((1:40, 1:40))

@testset "the term enumeration is pinned" begin
    # A fit of order n must share its leading terms with one of order n+1, so the two are comparable and
    # a coefficient vector means the same thing at either order.
    @test offset_fit_terms(0) == [(0, 0, 0)]
    @test length.(offset_fit_terms.(0:4)) == [1, 4, 10, 20, 35]
    @test offset_fit_terms(2)[1:4] == offset_fit_terms(1)
    # Total degree never exceeds the order.
    @test all(sum(t) <= 3 for t in offset_fit_terms(3))

    # `zorder` caps the elevation exponent, which is what keeps two sampled heights from asking for a z²
    # coefficient no data determines.
    @test all(t[3] <= 1 for t in offset_fit_terms(3; zorder = 1))
    @test length(offset_fit_terms(2; zorder = 0)) == 6   # no z terms at all
    @test_throws "must be non-negative" offset_fit_terms(-1)
end

@testset "a polynomial field is recovered at its own order" begin
    g, w = _grid(), _win()
    # Exactly quadratic in x, linear in y and z. Order 2 must be essentially exact; order 1 must not be,
    # or the test would pass for a fit that ignored the quadratic term.
    f = (x, y, z) -> (1.0 + 1e-3x + 2e-4y + 1e-8x^2 + 3e-9 * x * y + 1e-4z,
                      -0.5 + 5e-5x - 1e-4y)
    hs = (0.0, 1000.0, 2000.0)

    fit = fit_offset(f, g, w; heights = hs, target = 1e-6)
    @test fit.order == 2
    @test fit.residual[1] < 1e-10
    @test fit.residual[2] < 1e-10

    # And it evaluates away from the fitted nodes, which is the whole purpose — a fit that matched only
    # where it was sampled would be a lookup table with extra steps.
    for (x, y, z) in ((1234.0, 3456.0, 500.0), (77.0, 4900.0, 1750.0))
        ds, dl = fit(x, y, z)
        @test ds ≈ f(x, y, z)[1] atol = 1e-9
        @test dl ≈ f(x, y, z)[2] atol = 1e-9
    end

    # First order cannot describe it, and says so through its residual rather than by failing.
    one = fit_offset(f, g, w; heights = hs, order = 1)
    @test one.order == 1
    @test one.residual[1] > 1e-3

    # A higher order than needed is not worse — it is not *better* either, which is the check that the
    # extra terms are being determined rather than absorbing noise.
    three = fit_offset(f, g, w; heights = hs, order = 3)
    @test three.residual[1] < 1e-9
end

@testset "an unmeetable target is refused" begin
    g, w = _grid(), _win()
    # A field with a discontinuity no polynomial of any order describes.
    f = (x, y, z) -> (x > 2000 ? 5.0 : -5.0, 0.0)
    err = try
        fit_offset(f, g, w; heights = (0.0, 1000.0), target = 1e-3)
        nothing
    catch e
        sprint(showerror, e)
    end
    @test err !== nothing
    # The message must carry the residual actually reached and point at the alternative, since the
    # caller's next move is to use a lattice rather than to raise the order.
    @test occursin("no polynomial up to order", err)
    @test occursin("LatticeOffsetField", err)
    @test occursin("height_sensitivity", err)

    # Fixing the order sidesteps the target entirely: the residual is reported, not required.
    forced = fit_offset(f, g, w; heights = (0.0, 1000.0), order = 2)
    @test forced.order == 2
    @test forced.residual[1] > 1e-3
end

@testset "a fit needs more nodes than terms" begin
    g = _grid()
    # 3x3 positions x 2 heights = 18 samples. Order 2 has 10 terms with z capped at 1, needing 40.
    tiny = CartesianIndices((1:3, 1:3))
    f = (x, y, z) -> (1.0 + 1e-3x, 0.0)
    err = try
        fit_offset(f, g, tiny; heights = (0.0, 1000.0), order = 2)
        nothing
    catch e
        sprint(showerror, e)
    end
    @test err !== nothing
    # Named for what it means rather than as a dimension mismatch: a fit with as many terms as samples
    # interpolates, and its zero residual would be read as a perfect fit.
    @test occursin("interpolates rather than fits", err)
    # The count it names is derived, not hardcoded: two heights cap the elevation exponent at 1, so an
    # order-2 fit has 9 terms rather than 10 — which is the `zorder` capping doing its job.
    nterms = length(offset_fit_terms(2; zorder = 1))
    @test nterms == 9
    @test occursin(string(OFFSET_FIT_MIN_NODES_PER_TERM * nterms), err)

    # Selecting an order stops at the last one the sampling supports rather than throwing, since a
    # coarser fit is a real answer where a finer one is not.
    sel = fit_offset(f, g, tiny; heights = (0.0, 1000.0), target = 1e-9)
    @test sel.order == 1
end

@testset "the arguments are checked" begin
    g, w = _grid(), _win()
    f = (x, y, z) -> (1.0, 2.0)
    # One height cannot determine an elevation coefficient, and a fit that reported a residual anyway
    # would be claiming to have tested something it never varied.
    @test_throws "at least two heights" fit_offset(f, g, w; heights = (0.0,))
    @test_throws "must be distinct" fit_offset(f, g, w; heights = (500.0, 500.0))
    @test_throws "target must be positive" fit_offset(f, g, w; heights = (0.0, 1.0), target = 0.0)
end

@testset "the fit on real radar geometry" begin
    # The measurement this plan was written to obtain: how far the polynomial form holds, and at what
    # order, on the smallest-baseline pair available.
    fx = JSON3.read(read(joinpath(@__DIR__, "reference", "rdr2rdr.json"), String))
    mkc(o) = let kw = (; orbit = Orbit(; time = collect(Float64.(o.time)),
                                      position = [SVector{3,Float64}(p...) for p in o.pos],
                                      velocity = [SVector{3,Float64}(v...) for v in o.vel]),
                       starting_range = Float64(o.r0), dr = Float64(o.dr),
                       sensing_start = Float64(o.t0), prf = Float64(o.prf), nsamples = 24845,
                       nlines = 12244, look_side = LookRight, wavelength = Float64(o.wavelength))
        RadarCoordinate(; kw..., incidence_angle = incidence_angle(; kw...))
    end
    pair = CoregisteredPair(mkc(fx.ref); dt = 24 * 86400.0, secondary = mkc(fx.sec))
    ll = xyz_to_lonlat(Ellipsoid(), SVector{3,Float64}(Float64.(fx.target_xyz)...))
    lon, lat = rad2deg(ll[1]), rad2deg(ll[2])

    # About a degree of longitude across, which is a realistic scene extent, under the identity transform.
    g = MapGrid(geotransform = (lon - 0.5, 0.025, 0.0, lat + 0.25, 0.0, -0.025), size = (40, 20))
    w = CartesianIndices((1:40, 1:20))
    f = (x, y, z) -> pixel_offset(pair, x, y, z)
    hs = (0.0, 1000.0, 2000.0)

    r1 = fit_offset(f, g, w; heights = hs, order = 1)
    r2 = fit_offset(f, g, w; heights = hs, order = 2)
    r3 = fit_offset(f, g, w; heights = hs, order = 3)
    @info "offset fit residual by order, samples" order1 = r1.residual[1] order2 = r2.residual[1] order3 = r3.residual[1]

    # First order is what an affine model would have been, and it is not enough: 0.03 samples of
    # systematic error, structured across the swath rather than random. Second order is 200 times better
    # for six more coefficients, which is the finding that justifies selecting the order rather than
    # naming it.
    @test 0.01 < r1.residual[1] < 0.1
    @test r2.residual[1] < 1e-3
    @test r2.residual[1] < r1.residual[1] / 50

    # Selection at a target between the two picks the second, without being told which.
    sel = fit_offset(f, g, w; heights = hs, target = 0.01)
    @test sel.order == 2
    @test all(sel.residual .<= 0.01)

    # A loose target is met by first order, so the selection is genuinely driven by the target and not
    # pinned to one answer.
    @test fit_offset(f, g, w; heights = hs, target = 0.5).order == 1

    # The fit reproduces the field away from its nodes, at a point deliberately off-lattice.
    gx = lon - 0.5 + 0.025 * 17.37
    gy = lat + 0.25 - 0.025 * 8.11
    ds, dl = sel(gx, gy, 750.0)
    ex, el = pixel_offset(pair, gx, gy, 750.0)
    @test ds ≈ ex atol = 0.01
    @test dl ≈ el atol = 0.01

    # The magnitudes remain the documented ones, as a guard against a scaling error in the fit.
    @test 15.0 < ds < 20.0
    @test -3.5 < dl < -2.0
end
