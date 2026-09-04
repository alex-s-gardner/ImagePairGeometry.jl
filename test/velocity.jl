# Gate: `velocity_conversion` reports what a measured displacement needs to become a velocity.
#
# Two things are asserted. The arithmetic round-trips: a displacement the correlator would report,
# converted back through the operator, returns the velocity that produced it. And the y sign follows
# the coordinate system, which is the clause that fails silently — a wrong sign describes motion in
# the opposite direction and stays entirely plausible.

using ImagePairGeometry
using ImagePairGeometry: nodata_from
using Test

"""A geometry over a grid the pair covers, with every input band present."""
function vc_case(; csminy = 360.0, ssm = 1.0)
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
    return r, pair
end

# One geometry at the default inputs, shared by every testset that does not vary them: building it
# runs the whole kernel over the window.
const VC_R, VC_PAIR = vc_case()

@testset "the bands a displacement needs come through unchanged" begin
    r = VC_R
    c = velocity_conversion(r)

    # Shared, not copied: the arrays are large and the caller only reads them.
    @test c.off2vx.dx === r.off2vx_dx
    @test c.off2vx.dy === r.off2vx_dy
    @test c.off2vy.dx === r.off2vy_dx
    @test c.off2vy.dy === r.off2vy_dy
    @test c.scale.x === r.scale_x
    @test c.scale.y === r.scale_y
    @test c.nodata == -32767.0

    # The chip aspect ratio the reference derives as `ScaleChipSizeY` (`testautoRIFT.py:376`).
    @test c.chip_scale_y == 1.5        # csminy 360 over csminx 240
    @test velocity_conversion(first(vc_case(csminy = 240.0))).chip_scale_y == 1.0

    # The mask is boolean, and false wherever the point is invalid.
    @test eltype(c.stable_surface) === Bool
    sentinel = Int32(-32767)
    @test !any(c.stable_surface[findall(==(sentinel), r.location_x)])
    @test all(c.stable_surface[findall(!=(sentinel), r.location_x)])

    # A mask of zeros means nothing is stable, not that the band is missing.
    @test !any(velocity_conversion(first(vc_case(ssm = 0.0))).stable_surface)
end

@testset "velocity reconstruction round-trips" begin
    # `dy` is negated because `PairGeometry.offset_y` is in image axes, where +y points south, while
    # the operator expects +y pointing north.
    r = VC_R
    c = velocity_conversion(r)
    k = first(findall(!=(Int32(-32767)), r.location_x))

    dx = Float64(r.offset_x[k])
    dy = -Float64(r.offset_y[k])
    vx = c.off2vx.dx[k] * (dx * c.scale.x[k]) + c.off2vx.dy[k] * (dy * c.scale.y[k])
    vy = c.off2vy.dx[k] * (dx * c.scale.x[k]) + c.off2vy.dy[k] * (dy * c.scale.y[k])

    # The offset was rounded to whole pixels, so the tolerance is one pixel expressed in m/yr.
    tol = 30.0 / (91 * 86400.0) * (365.0 * 24.0 * 3600.0)
    @test vx ≈ 120.0 atol = tol
    @test vy ≈ -80.0 atol = tol
end

@testset "the y sign comes from the coordinate system" begin
    # A projected image needs no negation: its +y already points down its own second axis. The radar
    # case, which does, needs a coordinate with a real orbit and is asserted in `radar_geometry.jl`.
    r = VC_R

    @test y_displacement_sign(VC_PAIR.coordinate) === 1.0

    # Read off the result's own coordinate, so a caller cannot get it wrong by omission.
    @test velocity_conversion(r).dy_sign === 1.0

    # Something that is not a coordinate is refused rather than silently signed.
    @test_throws "expected a ProjectedCoordinate" velocity_conversion(r; coordinate = 42)
    @test_throws "expected a ProjectedCoordinate" y_displacement_sign(42)
end
