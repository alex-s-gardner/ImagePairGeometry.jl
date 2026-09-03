# Gate: the handoff to the correlator preserves each clause of the contract.
#
# Every assertion here corresponds to a way the conversion could silently misbehave rather than
# fail: an index off by one, a half pixel applied twice, a nodata sentinel arriving as a negative
# search radius, an output the correlator needs being dropped because `PointSet` has no field for it.

using ImagePairGeometry
using ImagePairGeometry: nodata_from, chip_size_pixels
using AutoRIFT
using Test

const AR_EXT = Base.get_extension(ImagePairGeometry, :ImagePairGeometryAutoRIFTExt)
using .AR_EXT: velocity_conversion

"""A geometry over a grid the pair covers, with every input band present."""
function ar_case(; csminy = 360.0, ssm = 1.0)
    grid = MapGrid(geotransform = (295000.0, 120.0, 0.0, 7805000.0, 0.0, -120.0),
                   size = (200, 200), crs = 32624)
    fp = ImageFootprint(origin = (300000.0, 7800000.0), spacing = (30.0, -30.0),
                        size = (400, 400))
    pair = coregister(fp, fp; dt = 91 * 86400.0)
    win = grid_window(grid, footprint_bounds(IdentityTransform(), pair.coordinate))
    n = size(win)
    inputs = GeometryInputs(dem = fill(500.0, n), dhdx = fill(0.02, n), dhdy = fill(-0.01, n),
                            vx = fill(120.0, n), vy = fill(-80.0, n),
                            srx = fill(400.0, n), sry = fill(300.0, n),
                            csminx = fill(240.0, n), csminy = fill(csminy, n),
                            csmaxx = fill(480.0, n), csmaxy = fill(720.0, n),
                            ssm = fill(ssm, n))
    r = pairgeometry(grid, pair, inputs; window = win, nodata = nodata_from(-32767.0))
    return r, grid, pair, win
end

@testset "the y prior's sign depends on the coordinate system" begin
    # The radar path negates the azimuth prior to reach AutoRIFT's north-up convention
    # (`testautoRIFT.py:405-407`, under `optical_flag == 0`); the projected path does not. Guessing
    # would be wrong half the time and silently so, so `coordinate` is required.
    r, _, pair, _ = ar_case()
    @test_throws "pass `coordinate`" AutoRIFT.pointset(r; pixel_size = 30.0)
    @test_throws "must be a ProjectedCoordinate" AutoRIFT.pointset(r; pixel_size = 30.0,
                                                                   coordinate = 42)

    proj = AutoRIFT.pointset(r; pixel_size = 30.0, coordinate = pair.coordinate)
    # A radar coordinate is not available here without an orbit, so the sign itself is checked
    # through the helper, and the prior's use of it through the projected path above.
    @test AR_EXT._prior_sign(pair.coordinate) === 1.0
    @test velocity_conversion(r; coordinate = pair.coordinate).dy_sign === 1.0
    @test all(>=(0), filter(!iszero, proj.dy_prior)) ||
          all(<=(0), filter(!iszero, proj.dy_prior))
end

@testset "pixel positions are one-based" begin
    r, _, pair, _ = ar_case()
    pts = AutoRIFT.pointset(r; coordinate = pair.coordinate, pixel_size = 30.0)
    @test pts isa AutoRIFT.PointSet{2}
    @test size(pts) == size(r)

    sentinel = Int32(-32767)
    valid = findall(!=(sentinel), r.location_x)
    # Geogrid's index is zero-based; AutoRIFT's is one-based. Exactly one, and no half pixel:
    # AutoRIFT adds that at correlation time for every pyramid level. Asserted over the whole grid
    # at once — the property is uniform, so one assertion per point would report the same fact
    # thousands of times.
    @test all(k -> pts.x[k] == r.location_x[k] + 1, valid)
    @test all(k -> pts.y[k] == r.location_y[k] + 1, valid)
    @test all(k -> isinteger(pts.x[k]), valid)
end

@testset "invalid points are skipped, not searched" begin
    r, _, pair, _ = ar_case()
    pts = AutoRIFT.pointset(r; coordinate = pair.coordinate, pixel_size = 30.0)
    sentinel = Int32(-32767)

    invalid = findall(==(sentinel), r.location_x)
    @test !isempty(invalid)     # the window overhangs the image, so some points are outside
    # Zero radius is how AutoRIFT marks a point to skip. Passing the sentinel through would make the
    # radius negative, which `gridpoints`' margin logic would size itself from.
    @test all(iszero, pts.radius_x[invalid])
    @test all(iszero, pts.radius_y[invalid])
    @test !any(k -> AutoRIFT.issearchable(pts, k), invalid)

    valid = findall(!=(sentinel), r.location_x)
    @test AutoRIFT.nsearchable(pts) == length(valid)
    @test all(k -> AutoRIFT.issearchable(pts, k), valid)
    @test all(>(0), pts.radius_x[valid])
end

@testset "search radius and prior carry through" begin
    r, _, pair, _ = ar_case()
    pts = AutoRIFT.pointset(r; coordinate = pair.coordinate, pixel_size = 30.0)
    sentinel = Int32(-32767)
    valid = findall(!=(sentinel), r.location_x)
    @test all(k -> pts.radius_x[k] == r.search_x[k], valid)
    @test all(k -> pts.radius_y[k] == r.search_y[k], valid)
    @test all(k -> pts.dx_prior[k] == r.offset_x[k], valid)
    @test all(k -> pts.dy_prior[k] == r.offset_y[k], valid)
end

@testset "chip size" begin
    r, _, pair, _ = ar_case()

    # From a pixel size, derived as the reference does.
    pts = AutoRIFT.pointset(r; coordinate = pair.coordinate, pixel_size = 30.0)
    @test all(==(chip_size_pixels(240.0, 30.0)), pts.chip_size_x)
    @test all(==(8), pts.chip_size_x)

    # Or given directly.
    pts32 = AutoRIFT.pointset(r; coordinate = pair.coordinate, chip_size = 32)
    @test all(==(32), pts32.chip_size_x)
    @test all(==(32), pts32.chip_size_y)

    # Bounds are per point, and a bound of zero means unbounded in AutoRIFT.
    sentinel = Int32(-32767)
    valid = findall(!=(sentinel), r.location_x)
    invalid = findall(==(sentinel), r.location_x)
    @test all(k -> pts.chip_size_min_x[k] == r.chip_min_x[k], valid)
    @test all(k -> pts.chip_size_max_x[k] == r.chip_max_x[k], valid)
    @test all(iszero, pts.chip_size_min_x[invalid])
    @test all(iszero, pts.chip_size_max_x[invalid])

    # Neither given is an error rather than a silent default: the base extent is not recoverable
    # from a PairGeometry, which stores the bounds but not the base.
    @test_throws "needs the base chip extent" AutoRIFT.pointset(r)
end

@testset "velocity_conversion carries what PointSet cannot" begin
    r, _, pair, _ = ar_case()
    c = velocity_conversion(r; coordinate = pair.coordinate)

    @test c.off2vx.dx === r.off2vx_dx
    @test c.off2vx.dy === r.off2vx_dy
    @test c.off2vy.dx === r.off2vy_dx
    @test c.off2vy.dy === r.off2vy_dy
    @test c.scale.x === r.scale_x
    @test c.scale.y === r.scale_y
    @test c.nodata == -32767.0

    # The chip aspect ratio the reference derives as `ScaleChipSizeY`.
    @test c.chip_scale_y == 1.5        # csminy 360 over csminx 240
    eq = ar_case(csminy = 240.0)
    @test velocity_conversion(eq[1]; coordinate = eq[3].coordinate).chip_scale_y == 1.0

    # The mask is boolean, and false wherever the point is invalid.
    @test eltype(c.stable_surface) === Bool
    sentinel = Int32(-32767)
    @test !any(c.stable_surface[findall(==(sentinel), r.location_x)])
    @test all(c.stable_surface[findall(!=(sentinel), r.location_x)])

    # A mask of zeros means nothing is stable, not that the band is missing.
    ns = ar_case(ssm = 0.0)
    @test !any(velocity_conversion(ns[1]; coordinate = ns[3].coordinate).stable_surface)
end

@testset "velocity reconstruction round-trips" begin
    # The whole point of the handoff: a displacement the correlator would report, converted back to
    # the map velocity that produced it. `dy` is negated because `PairGeometry.offset_y` is in image
    # axes where +y points south, while the operator expects AutoRIFT's convention.
    r, _, pair, _ = ar_case()
    c = velocity_conversion(r; coordinate = pair.coordinate)
    sentinel = Int32(-32767)
    k = first(findall(!=(sentinel), r.location_x))

    dx = Float64(r.offset_x[k])
    dy = -Float64(r.offset_y[k])
    vx = c.off2vx.dx[k] * (dx * c.scale.x[k]) + c.off2vx.dy[k] * (dy * c.scale.y[k])
    vy = c.off2vy.dx[k] * (dx * c.scale.x[k]) + c.off2vy.dy[k] * (dy * c.scale.y[k])

    # The offset was rounded to whole pixels, so the tolerance is one pixel expressed in m/yr.
    tol = 30.0 / (91 * 86400.0) * (365.0 * 24.0 * 3600.0)
    @test vx ≈ 120.0 atol = tol
    @test vy ≈ -80.0 atol = tol
end

@testset "a geometry with no search band is refused" begin
    # Without `srx`/`sry` every radius would be zero and nothing would be correlated. That is a
    # missing input, not a grid of skips, so it fails with a message saying which.
    grid = MapGrid(geotransform = (295000.0, 120.0, 0.0, 7805000.0, 0.0, -120.0),
                   size = (200, 200), crs = 32624)
    fp = ImageFootprint(origin = (300000.0, 7800000.0), spacing = (30.0, -30.0), size = (400, 400))
    pair = coregister(fp, fp; dt = 91 * 86400.0)
    win = grid_window(grid, footprint_bounds(IdentityTransform(), pair.coordinate))
    n = size(win)
    bare = pairgeometry(grid, pair, GeometryInputs(dem = fill(500.0, n)); window = win,
                        nodata = nodata_from(-32767.0))
    @test_throws "no search-range band" AutoRIFT.pointset(bare; pixel_size = 30.0)
end

@testset "blocked geometry converts identically" begin
    # The conversion reads only the result, so a blocked run must give the same PointSet.
    r, grid, pair, win = ar_case()
    n = size(win)
    inputs = GeometryInputs(dem = fill(500.0, n), dhdx = fill(0.02, n), dhdy = fill(-0.01, n),
                            vx = fill(120.0, n), vy = fill(-80.0, n),
                            srx = fill(400.0, n), sry = fill(300.0, n),
                            csminx = fill(240.0, n), csminy = fill(360.0, n),
                            csmaxx = fill(480.0, n), csmaxy = fill(720.0, n), ssm = fill(1.0, n))
    blocked = pairgeometry_blocked(grid, pair, InMemoryInputs(inputs, win);
                                   transform = IdentityTransform(), window = win,
                                   blocksize = (16, 16), nodata = nodata_from(-32767.0))
    a = AutoRIFT.pointset(r; coordinate = pair.coordinate, pixel_size = 30.0)
    b = AutoRIFT.pointset(blocked; coordinate = pair.coordinate, pixel_size = 30.0)
    for f in (:x, :y, :radius_x, :radius_y, :dx_prior, :dy_prior, :chip_size_x, :chip_size_y,
              :chip_size_min_x, :chip_size_max_x)
        @test getfield(a, f) == getfield(b, f)
    end
end
