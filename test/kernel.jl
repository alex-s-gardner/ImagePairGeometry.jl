# The kernel checked against values derived by hand, with no reference and no PROJ.
#
# Under an identity or affine transform every quantity has a closed form, so these assert what the
# arithmetic *should* be rather than what another implementation happens to produce. That makes
# them a check on the transcription independent of the reference comparison, which cannot
# distinguish "both right" from "both wrong in the same way".

using ImagePairGeometry
using ImagePairGeometry: PointGeometry, pointgeometry, pixel_index, inbounds, cross_check,
                         surface_normal, NO_NORMAL, DEG2RAD,
                         per_year, close_slope_parallel, pixel_offset, offset_to_velocity,
                         scale_factors, search_pixels, chip_pixels, chip_size_pixels,
                         SearchRangeScaling, searchrange_scale, scaled_searchrange,
                         inverse, isidentity, dot3, norm3, unitvec3
using StaticArrays: SVector
using Test

v3(a, b, c) = SVector{3,Float64}(a, b, c)
const YR = 365.0 * 24.0 * 3600.0

@testset "per_year divides in the reference's order" begin
    @test per_year(YR) === 1.0
    # Sequential division is not division by the product, and the difference is real.
    differs = count(1:5000) do i
        x = i * 1234.567
        reinterpret(UInt64, per_year(x)) != reinterpret(UInt64, x / YR)
    end
    @test differs > 0
    @test all(1:2000) do i
        x = i * 1234.567
        isapprox(per_year(x), x / YR; rtol = 1e-15)
    end
end

@testset "identity transform: exact closed form" begin
    c = ProjectedCoordinate(origin = (300000.0, 7800000.0), spacing = (30.0, -30.0),
                            size = (100, 100))
    tf = transform_pair(IdentityTransform())
    @test isidentity(tf)

    g = pointgeometry(tf, 300150.0, 7799850.0, 500.0, c, NO_NORMAL)
    @test g.image_xy === (300150.0, 7799850.0)
    @test g.xunit === v3(1, 0, 0)
    @test g.yunit === v3(0, -1, 0)      # north-up raster: +y in image space is south
    @test g.xlen === 30.0
    @test g.ylen === 30.0

    # Scale factors are exactly one, no rounding involved.
    @test scale_factors(g, c) === (1.0, 1.0)

    # Pixel index: 150 m at 30 m spacing is 5 pixels in, in both axes.
    @test pixel_index(g, c) === (5.0, 5.0)
    @test inbounds(5.0, 5.0, c)
end

@testset "identity path equals the general path" begin
    # The specialization must be a shortcut in work only. An explicit identity affine transform
    # takes the general code path and must agree bit for bit.
    c = ProjectedCoordinate(origin = (300000.0, 7800000.0), spacing = (30.0, -30.0),
                            size = (500, 500))
    fast = transform_pair(IdentityTransform())
    slow = transform_pair(AffineTransform(a = 1.0, b = 0.0, c = 0.0, d = 0.0, e = 1.0, f = 0.0))
    @test !isidentity(slow)

    n = surface_normal(0.05, -0.03)
    for i in 0:40, j in 0:40
        gx = 300000.0 + i * 120.0
        gy = 7800000.0 - j * 120.0
        a = pointgeometry(fast, gx, gy, 500.0, c, n)
        b = pointgeometry(slow, gx, gy, 500.0, c, n)
        @test a.xunit == b.xunit
        @test a.yunit == b.yunit
        @test reinterpret(UInt64, a.xlen) == reinterpret(UInt64, b.xlen)
        @test reinterpret(UInt64, a.ylen) == reinterpret(UInt64, b.ylen)
        @test pixel_index(a, c) == pixel_index(b, c)
        @test scale_factors(a, c) == scale_factors(b, c)
    end
end

@testset "affine transform: derived by hand" begin
    # A pure scaling by 2: one image pixel of 30 m spans 15 grid units, so the axis lengths halve
    # and the scale factors are 0.5.
    c = ProjectedCoordinate(origin = (0.0, 0.0), spacing = (30.0, -30.0), size = (100, 100))
    t = AffineTransform(a = 2.0, b = 0.0, c = 0.0, d = 0.0, e = 2.0, f = 0.0)
    tf = transform_pair(t)

    g = pointgeometry(tf, 100.0, 200.0, 0.0, c, NO_NORMAL)
    @test g.image_xy === (200.0, 400.0)
    @test g.xunit === v3(1, 0, 0)
    @test g.yunit === v3(0, -1, 0)
    @test g.xlen === 15.0
    @test g.ylen === 15.0
    @test scale_factors(g, c) === (0.5, 0.5)

    # A 90-degree rotation: the image x axis points along grid +y.
    rot = AffineTransform(a = 0.0, b = -1.0, c = 0.0, d = 1.0, e = 0.0, f = 0.0)
    gr = pointgeometry(transform_pair(rot), 10.0, 20.0, 0.0, c, NO_NORMAL)
    @test gr.xunit ≈ v3(0, -1, 0)
    @test gr.yunit ≈ v3(-1, 0, 0)
    @test gr.xlen ≈ 30.0        # rotation preserves length
    @test gr.ylen ≈ 30.0
    @test all(isapprox(1.0), scale_factors(gr, c))
end

@testset "AffineTransform inverse round-trips" begin
    for t in (AffineTransform(a = 2.0, b = 0.0, c = 5.0, d = 0.0, e = 3.0, f = -7.0),
              AffineTransform(a = 0.0, b = -1.0, c = 0.0, d = 1.0, e = 0.0, f = 0.0),
              AffineTransform(a = 1.5, b = 0.25, c = -3.0, d = -0.5, e = 2.0, f = 11.0))
        inv = inverse(t)
        for (x, y, z) in ((1.0, 2.0, 3.0), (-100.0, 50.0, 0.0), (1e5, -2e5, 700.0))
            fx, fy, fz = t(x, y, z)
            rx, ry, rz = inv(fx, fy, fz)
            @test rx ≈ x && ry ≈ y && rz ≈ z
        end
    end
    @test_throws "singular" AffineTransform(a = 1.0, b = 2.0, c = 0.0, d = 2.0, e = 4.0, f = 0.0)
end

@testset "surface_normal" begin
    # Flat surface: the normal points straight up. Compared by value, not bit pattern: the
    # negation in `surface_normal` turns the zero components to `-0.0`, which is the same number.
    @test surface_normal(0.0, 0.0) == v3(0, 0, 1)

    # A slope tilts the normal against the gradient, and z stays positive.
    n = surface_normal(0.1, 0.2)
    @test n[1] < 0 && n[2] < 0 && n[3] > 0
    @test norm3(n) ≈ 1.0
    # The normal is perpendicular to both surface tangents.
    @test dot3(n, v3(1, 0, 0.1)) ≈ 0 atol = 1e-15
    @test dot3(n, v3(0, 1, 0.2)) ≈ 0 atol = 1e-15
end

@testset "slope-parallel closure puts velocity in the surface" begin
    for (dhdx, dhdy) in ((0.0, 0.0), (0.1, 0.0), (0.0, -0.2), (0.05, 0.07), (-0.3, 0.4))
        n = surface_normal(dhdx, dhdy)
        for (vx, vy) in ((100.0, 0.0), (0.0, -50.0), (30.0, 40.0))
            v = close_slope_parallel(vx, vy, n)
            @test v[1] === vx && v[2] === vy
            @test dot3(v, n) ≈ 0 atol = 1e-9        # flow lies in the surface
            # Downhill in x means the vertical component follows the slope.
            @test v[3] ≈ vx * dhdx + vy * dhdy atol = 1e-9
        end
    end

    # No slope raster: the normal is zero and the closure is non-finite, as the reference computes.
    v = close_slope_parallel(1.0, 1.0, NO_NORMAL)
    @test !isfinite(v[3])
end

@testset "pixel_offset" begin
    c = ProjectedCoordinate(origin = (0.0, 0.0), spacing = (30.0, -30.0), size = (100, 100))
    g = pointgeometry(transform_pair(IdentityTransform()), 0.0, 0.0, 0.0, c, NO_NORMAL)

    # 300 m/yr along +x for one year over 30 m pixels is 10 pixels.
    @test pixel_offset(v3(300, 0, 0), g, YR) == (10.0, 0.0)
    # Along +y the image axis points south, so a northward velocity is a negative pixel offset.
    @test pixel_offset(v3(0, 300, 0), g, YR) === (0.0, -10.0)
    # Half a year, half the displacement.
    @test pixel_offset(v3(300, 0, 0), g, YR / 2) == (5.0, 0.0)
    # Rounds to whole pixels, halves away from zero.
    @test pixel_offset(v3(45, 0, 0), g, YR) == (2.0, 0.0)     # 1.5 -> 2, not 1
end

@testset "offset_to_velocity inverts pixel_offset" begin
    # The operator undoes the displacement calculation, but only after the y sign convention is
    # applied. `pixel_offset` gives a displacement in *image* axes, where +y points down a
    # north-up raster; the operator expects the correlator's convention, where it points up. The
    # reference negates between the two — at `testautoRIFT.py:407` for the prior it feeds in, and
    # at `:790` for the displacement it gets back — so the negation belongs to the correlator, not
    # here. Verified against the reference's own arithmetic: for `vy = +250` it likewise returns
    # -240.66 without the flip.
    c = ProjectedCoordinate(origin = (0.0, 0.0), spacing = (30.0, -30.0), size = (2000, 2000))
    n = surface_normal(0.02, -0.05)
    g = pointgeometry(transform_pair(IdentityTransform()), 0.0, 0.0, 0.0, c, n)
    dt = 91 * 86400.0

    @test cross_check(g) > 1.0       # operator is valid here
    vx_dx, vx_dy, vy_dx, vy_dy = offset_to_velocity(g, c, dt)
    sfx, sfy = scale_factors(g, c)
    tol = 30.0 / dt * YR             # the rounding to whole pixels, expressed in m/yr

    for (vx, vy) in ((300.0, 0.0), (0.0, 250.0), (-120.0, 80.0), (1000.0, -1000.0),
                     (0.0, 0.0), (-500.0, -500.0))
        v = close_slope_parallel(vx, vy, n)
        dx, dy = pixel_offset(v, g, dt)
        dy_corr = -dy               # image-axis to correlator convention
        rx = vx_dx * (dx * sfx) + vx_dy * (dy_corr * sfy)
        ry = vy_dx * (dx * sfx) + vy_dy * (dy_corr * sfy)
        @test rx ≈ vx atol = tol
        @test ry ≈ vy atol = tol
    end
end

@testset "offset_to_velocity scales as 1/dt" begin
    # A property the expression must have, not a shortcut the code takes: doubling the interval
    # halves the velocity a given displacement implies.
    c = ProjectedCoordinate(origin = (0.0, 0.0), spacing = (30.0, -30.0), size = (100, 100))
    n = surface_normal(0.01, 0.02)
    g = pointgeometry(transform_pair(IdentityTransform()), 0.0, 0.0, 0.0, c, n)
    a = offset_to_velocity(g, c, YR)
    b = offset_to_velocity(g, c, 2 * YR)
    for k in 1:4
        @test b[k] ≈ a[k] / 2 rtol = 1e-12
    end
end

@testset "cross_check is 90 degrees less the slope angle" begin
    # For an axis-aligned image the axis unit vectors span the horizontal plane, so their cross
    # product is vertical and the angle to the surface normal is exactly the terrain slope. That
    # makes the reference's `> 1.0` gate a rejection of slopes steeper than 89° — near-vertical
    # terrain, effectively unreachable in a real DEM.
    #
    # It also settles the `acos` question: `acos` differs by one ULP between openlibm and the
    # system libm, but that can only flip an output for a point sitting within a ULP of the gate,
    # i.e. at 89° exactly.
    c = ProjectedCoordinate(origin = (0.0, 0.0), spacing = (30.0, -30.0), size = (100, 100))
    tf = transform_pair(IdentityTransform())

    for (dhdx, dhdy) in ((0.0, 0.0), (0.05, 0.05), (0.1, 0.1), (0.2, -0.3), (0.5, 0.5),
                         (1.0, 0.0), (2.0, 0.0), (5.0, 0.0), (20.0, 0.0), (200.0, 0.0))
        g = pointgeometry(tf, 0.0, 0.0, 0.0, c, surface_normal(dhdx, dhdy))
        slope_deg = atand(sqrt(dhdx^2 + dhdy^2))
        @test cross_check(g) ≈ 90.0 - slope_deg atol = 1e-9
    end

    # Flat ground is exactly 90°, and terrain at any plausible steepness clears the gate wide.
    flat = pointgeometry(tf, 0.0, 0.0, 0.0, c, surface_normal(0.0, 0.0))
    @test cross_check(flat) === 90.0
    steep = pointgeometry(tf, 0.0, 0.0, 0.0, c, surface_normal(1.0, 1.0))    # 55°
    @test cross_check(steep) > 1.0

    # The gate bites only within a degree of vertical.
    vertical = pointgeometry(tf, 0.0, 0.0, 0.0, c, surface_normal(200.0, 0.0))
    @test cross_check(vertical) < 1.0
end

@testset "scale_factors are one under identity, per-point otherwise" begin
    c = ProjectedCoordinate(origin = (0.0, 0.0), spacing = (30.0, -30.0), size = (100, 100))
    g = pointgeometry(transform_pair(IdentityTransform()), 0.0, 0.0, 0.0, c, NO_NORMAL)
    @test scale_factors(g, c) === (1.0, 1.0)

    # A 0.9996-style shrink, as at a UTM central meridian.
    t = AffineTransform(a = 1 / 0.9996, b = 0.0, c = 0.0, d = 0.0, e = 1 / 0.9996, f = 0.0)
    gs = pointgeometry(transform_pair(t), 0.0, 0.0, 0.0, c, NO_NORMAL)
    @test all(isapprox(0.9996), scale_factors(gs, c))
end

@testset "search_pixels" begin
    c = ProjectedCoordinate(origin = (0.0, 0.0), spacing = (30.0, -30.0), size = (100, 100))
    n = surface_normal(0.0, 0.0)
    g = pointgeometry(transform_pair(IdentityTransform()), 0.0, 0.0, 0.0, c, n)

    # 300 m/yr search over a year at 30 m pixels: 10 pixels each way.
    sr1 = close_slope_parallel(300.0, 300.0, n)
    sr2 = close_slope_parallel(-300.0, 300.0, n)
    @test search_pixels(sr1, sr2, g, c, YR) === (10.0, 10.0)

    # Always at least one pixel, however small the range.
    tiny1 = close_slope_parallel(0.001, 0.001, n)
    tiny2 = close_slope_parallel(-0.001, 0.001, n)
    @test search_pixels(tiny1, tiny2, g, c, YR) === (1.0, 1.0)

    # Asymmetric ranges give asymmetric extents, and the result is a magnitude.
    a1 = close_slope_parallel(900.0, 90.0, n)
    a2 = close_slope_parallel(-900.0, 90.0, n)
    sx, sy = search_pixels(a1, a2, g, c, YR)
    @test sx === 30.0 && sy === 3.0
end

@testset "searchrange_scale" begin
    p = SearchRangeScaling()
    @test p.dt_unity == 182.0 && p.max_scale == 5.0 && p.upper == 20000.0

    # Falls from max_scale toward one, reaching it at dt_unity and staying there.
    @test searchrange_scale(p, 1 * 86400.0) ≈ 5.0 rtol = 1e-3
    @test searchrange_scale(p, 182 * 86400.0) === 1.0
    @test searchrange_scale(p, 365 * 86400.0) === 1.0
    @test searchrange_scale(p, 10 * 365 * 86400.0) === 1.0

    # Monotone non-increasing in dt.
    scales = [searchrange_scale(p, d * 86400.0) for d in 1:400]
    @test issorted(scales; rev = true)
    @test all(>=(1.0), scales)
end

@testset "search range is not separable in dt" begin
    # The property that rules out factoring dt out of the search range: the pixel extent is not
    # monotone in dt, because the inflation falls faster than the displacement grows.
    p = SearchRangeScaling()
    c = ProjectedCoordinate(origin = (0.0, 0.0), spacing = (30.0, -30.0), size = (5000, 5000))
    n = surface_normal(0.0, 0.0)
    g = pointgeometry(transform_pair(IdentityTransform()), 0.0, 0.0, 0.0, c, n)

    px = map((1, 5, 16, 32, 91, 182, 365)) do days
        dt = days * 86400.0
        s = searchrange_scale(p, dt)
        sr = scaled_searchrange(p, 500.0, s)
        sr1 = close_slope_parallel(sr, sr, n)
        sr2 = close_slope_parallel(-sr, sr, n)
        first(search_pixels(sr1, sr2, g, c, dt))
    end
    @test !issorted(px)         # 91 days searches wider than 182

    # The clamp changes the *direction* of the search vector, not just its length: an x range at
    # the ceiling and a y range below it come out with a different ratio than they went in.
    s = searchrange_scale(p, 86400.0)
    @test scaled_searchrange(p, 5000.0, s) === 20000.0      # clamped
    @test scaled_searchrange(p, 100.0, s) < 20000.0         # not clamped
    @test scaled_searchrange(p, 5000.0, s) / scaled_searchrange(p, 100.0, s) != 50.0
end

@testset "scaled_searchrange clamps" begin
    p = SearchRangeScaling()
    @test scaled_searchrange(p, 100.0, 1.0) === 100.0
    @test scaled_searchrange(p, 1e9, 1.0) === 20000.0       # upper
    @test scaled_searchrange(p, -32767.0, 1.0) === 0.0      # a negative sentinel maps to lower

    # The search-range rasters' own +32767 sentinel clamps to the ceiling and reads as a valid
    # 20000 m/yr range — the reference's nodata test for search range cannot catch it.
    @test scaled_searchrange(p, 32767.0, 5.0) === 20000.0
end

@testset "chip size" begin
    # Reference chip size in pixels, rounded up to a multiple of four.
    @test chip_size_pixels(240.0, 30.0) === 8
    @test chip_size_pixels(240.0, 15.0) === 16
    @test chip_size_pixels(240.0, 10.0) === 24
    @test chip_size_pixels(240.0, 60.0) === 4
    @test chip_size_pixels(240.0, 7.0) === 36       # 34.28 -> ceil to 36
    @test chip_size_pixels(240.0, 30.0) % 4 == 0

    # Converting a chip size in meters: scaled by the pixel equivalent of the reference size.
    @test chip_pixels(240.0, 240.0, 240.0, 8, 8) === (8.0, 8.0)
    @test chip_pixels(480.0, 240.0, 240.0, 8, 8) === (16.0, 8.0)
    # Truncation, not rounding, is the caller's job — this returns the unrounded value.
    @test chip_pixels(340.0, 340.0, 240.0, 16, 16)[1] ≈ 22.666666666666668
end

@testset "inbounds" begin
    c = ProjectedCoordinate(origin = (0.0, 0.0), spacing = (1.0, -1.0), size = (10, 20))
    @test inbounds(0.0, 0.0, c)
    @test inbounds(9.0, 19.0, c)
    @test !inbounds(10.0, 0.0, c)
    @test !inbounds(0.0, 20.0, c)
    @test !inbounds(-1.0, 0.0, c)
    @test !inbounds(0.0, -1.0, c)
    # NaN reports in bounds, as the reference reports it. See REFERENCE.md.
    @test inbounds(NaN, NaN, c)
end

@testset "type stable and non-allocating" begin
    c = ProjectedCoordinate(origin = (0.0, 0.0), spacing = (30.0, -30.0), size = (100, 100))
    n = surface_normal(0.01, 0.02)
    tf = transform_pair(IdentityTransform())
    af = transform_pair(AffineTransform(a = 1.2, b = 0.1, c = 3.0, d = -0.1, e = 1.1, f = -2.0))

    @test @inferred(pointgeometry(tf, 1.0, 2.0, 3.0, c, n)) isa PointGeometry
    @test @inferred(pointgeometry(af, 1.0, 2.0, 3.0, c, n)) isa PointGeometry
    g = pointgeometry(tf, 1.0, 2.0, 3.0, c, n)
    @test @inferred(pixel_index(g, c)) isa NTuple{2,Float64}
    @test @inferred(cross_check(g)) isa Float64
    @test @inferred(offset_to_velocity(g, c, YR)) isa NTuple{4,Float64}
    @test @inferred(scale_factors(g, c)) isa NTuple{2,Float64}
    @test @inferred(pixel_offset(v3(1, 1, 1), g, YR)) isa NTuple{2,Float64}

    @test @allocated(pointgeometry(tf, 1.0, 2.0, 3.0, c, n)) == 0
    @test @allocated(pointgeometry(af, 1.0, 2.0, 3.0, c, n)) == 0
    @test @allocated(offset_to_velocity(g, c, YR)) == 0
    @test @allocated(cross_check(g)) == 0
end
