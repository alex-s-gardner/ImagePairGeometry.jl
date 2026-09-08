# The offset between the two images of a pair: the interface, and what it refuses.
#
# `pixel_offset` is where the package answers "the secondary does not sit on the reference's grid, by how
# much?" — a question the geometry kernel does not ask, since it computes one pixel index and hands it to
# a correlator for both images.
#
# What is tested here is mostly the *preconditions*, and deliberately so. An offset is a pixel shift in
# the reference's own axes, which describes two images only when both are indexed the same way, so a pair
# the operation cannot answer has to be refused rather than answered wrongly. Each refusal is asserted on
# its message rather than on `ArgumentError`, since the type alone would pass for the wrong reason.
#
# The projected method returning zero is the substance of that path, not a placeholder: two
# orthorectified images are already on one grid. See `src/misregistration.jl` for why nothing is derived
# there.

using ImagePairGeometry
using ImagePairGeometry: ProjectedCoordinate, RadarCoordinate, CoregisteredPair, ImageFootprint,
                         coregister, Orbit, LookRight, incidence_angle, pixel_offset,
                         height_sensitivity, Ellipsoid, xyz_to_lonlat
using StaticArrays: SVector
using JSON3
using Test

# A circular orbit, as `test/radar_geometry.jl` builds one.
#
# Spans well beyond the acquisition it carries, because a zero-Doppler solve converges to wherever the
# geometry puts it and the interpolator refuses a time outside its state vectors — correctly, since
# extrapolating an orbit is how a plausible wrong answer gets made. A real product's orbit brackets its
# acquisition by design, and `SLCDatasets` refuses one that does not.
function _orbit(; t0 = -2000.0, n = 401, dt = 10.0, radius = 7.0e6, incl = deg2rad(98.0))
    omega = sqrt(3.986004418e14 / radius^3)
    ts = collect(t0 .+ dt .* (0:(n - 1)))
    pos = SVector{3,Float64}[]
    vel = SVector{3,Float64}[]
    for t in ts
        a = omega * t
        push!(pos, SVector(radius * cos(a), radius * sin(a) * cos(incl), radius * sin(a) * sin(incl)))
        push!(vel, SVector(-radius * omega * sin(a), radius * omega * cos(a) * cos(incl),
                           radius * omega * cos(a) * sin(incl)))
    end
    return Orbit(; time = ts, position = pos, velocity = vel)
end

function _radar(; sensing_start = 60.0, orb = _orbit())
    kw = (; orbit = orb, starting_range = 8.0e5, dr = 2.3295621147, sensing_start,
          prf = 486.4863103, nsamples = 500, nlines = 400, look_side = LookRight,
          wavelength = 0.05546576)
    return RadarCoordinate(; kw..., incidence_angle = incidence_angle(; kw...))
end

_proj(; origin = (0.0, 0.0), spacing = (10.0, -10.0), size = (20, 20)) =
    ProjectedCoordinate(; origin, spacing, size)

# The two acquisitions of the committed fixture, as coordinates. The image size is the subswath the
# fixture was extracted from; it does not enter the offset, which is a difference of two solves, but a
# `RadarCoordinate` cannot exist without it.
function _fixture_coord(o)
    kw = (; orbit = Orbit(; time = collect(Float64.(o.time)),
                          position = [SVector{3,Float64}(p...) for p in o.pos],
                          velocity = [SVector{3,Float64}(v...) for v in o.vel]),
          starting_range = Float64(o.r0), dr = Float64(o.dr), sensing_start = Float64(o.t0),
          prf = Float64(o.prf), nsamples = 24845, nlines = 12244, look_side = LookRight,
          wavelength = Float64(o.wavelength))
    return RadarCoordinate(; kw..., incidence_angle = incidence_angle(; kw...))
end

# A coordinate system this package has never seen, with a closed-form offset, to drive the
# coordinate-agnostic spine. Declared in the test rather than in `src/` precisely because the point is
# that the field, lattice and fit require nothing of a coordinate beyond a `pixel_offset` method.
struct _ToyCoordinate <: ImagePairGeometry.AbstractImageCoordinate end

ImagePairGeometry.nsamples(::_ToyCoordinate) = 1000
ImagePairGeometry.nlines(::_ToyCoordinate) = 1000
ImagePairGeometry.y_displacement_sign(::_ToyCoordinate) = 1.0

# Affine in position and constant in elevation, so bilinear interpolation reproduces it exactly.
ImagePairGeometry._check_secondary(::_ToyCoordinate, ::_ToyCoordinate) = nothing
ImagePairGeometry._pixel_offset(::_ToyCoordinate, ::_ToyCoordinate,
                               x::Float64, y::Float64, z::Float64, _) =
    (1.0 + 1.0e-3 * x + 2.0e-4 * y, -0.5 + 5.0e-5 * x - 1.0e-4 * y)

@testset "a pair carries no secondary coordinate unless given one" begin
    # Every pair built before this interface existed is in this state, and it is not an error: the
    # geometry outputs do not read the field, so only `pixel_offset` objects.
    a = ImageFootprint(origin = (100.0, 900.0), spacing = (10.0, -10.0), size = (50, 50))
    b = ImageFootprint(origin = (150.0, 950.0), spacing = (10.0, -10.0), size = (50, 50))
    p = coregister(a, b; dt = 86400.0)
    @test p.secondary_offset_coordinate === nothing

    # `coregister` computes the overlap, which *is* the frame both images share, so there is no
    # per-image geometry left over to record.
    @test CoregisteredPair(_proj(); dt = 1.0).secondary_offset_coordinate === nothing
    @test CoregisteredPair(_radar(); dt = 1.0).secondary_offset_coordinate === nothing

    # Reaching the offset without one says what to supply rather than failing on dispatch.
    @test_throws "carries no secondary coordinate" pixel_offset(p, 1.0, 2.0, 3.0)
end

@testset "the two coordinates must be the same kind" begin
    # A pixel shift presumes a common frame. Slant-range/azimuth and row/column are not one, so there is
    # no offset to compute in either direction — asserted both ways round, since a rule that held only
    # for one ordering would be a dispatch accident.
    @test_throws "must be the same kind" CoregisteredPair(_proj(); dt = 1.0, secondary = _radar())
    @test_throws "must be the same kind" CoregisteredPair(_radar(); dt = 1.0, secondary = _proj())
    # The message names both kinds, so a caller knows which argument was wrong.
    err = try
        CoregisteredPair(_proj(); dt = 1.0, secondary = _radar())
    catch e
        sprint(showerror, e)
    end
    @test occursin("ProjectedCoordinate", err) && occursin("RadarCoordinate", err)
end

@testset "a projected pair must share a pixel spacing" begin
    # The same precondition `coregister` places on two footprints: an offset in pixels presumes one
    # pixel size. Checked when the pair is built, not when the offset is first read.
    @test_throws "share a pixel spacing" CoregisteredPair(
        _proj(); dt = 1.0, secondary = _proj(spacing = (20.0, -20.0)))
    # The message carries both spacings.
    err = try
        CoregisteredPair(_proj(); dt = 1.0, secondary = _proj(spacing = (20.0, -20.0)))
    catch e
        sprint(showerror, e)
    end
    @test occursin("(10.0, -10.0)", err) && occursin("(20.0, -20.0)", err)

    # Differing origin and size are fine: two scenes on one grid overlap partially, which is the normal
    # case and is what the offsets into each image describe.
    p = CoregisteredPair(_proj(); dt = 1.0,
                         secondary = _proj(origin = (500.0, -500.0), size = (33, 44)))
    @test p.secondary_offset_coordinate isa ProjectedCoordinate
end

@testset "the projected offset is zero" begin
    # Both images are terrain-corrected onto the same map grid, so a grid point is at the same place in
    # each. Zero is the answer, not a stub: the displacement an orthorectification would introduce has
    # already been removed by the producer, and what remains is not a function of anything a
    # `ProjectedCoordinate` holds.
    p = CoregisteredPair(_proj(); dt = 86400.0, secondary = _proj(origin = (5.0, 5.0)))
    @test pixel_offset(p, 0.0, 0.0, 0.0) === (0.0, 0.0)
    @test pixel_offset(p, 1234.5, -987.6, 2500.0) === (0.0, 0.0)
    # Exactly zero at every elevation, since no height term exists on this path — a nonzero result at
    # some elevation would mean a parallax term had been introduced without the view geometry to justify
    # it. See `src/misregistration.jl`.
    @test all(pixel_offset(p, 100.0, 200.0, z) === (0.0, 0.0) for z in (-400.0, 0.0, 1e4))
end

@testset "an acquisition paired with itself has no offset" begin
    # The strongest cheap check on the radar path, and the one that fails on any clock error: one
    # acquisition against itself must give exactly zero, because both solves are the same solve. A
    # `sensing_start` read for the wrong image, an epoch added once instead of twice, or the two
    # coordinates swapped all break this and little else — the offsets they produce are small and
    # plausible everywhere else.
    c = _radar()
    p = CoregisteredPair(c; dt = 86400.0, secondary = c)
    for (x, y, z) in ((0.0, 0.0, 0.0), (12.0, -34.0, 500.0), (-100.0, 60.0, 3000.0))
        @test pixel_offset(p, x, y, z) === (0.0, 0.0)
    end

    # Two *equal but distinct* coordinates, so the zero comes from the geometry agreeing rather than
    # from the two arguments being the same object.
    @test pixel_offset(CoregisteredPair(_radar(); dt = 1.0, secondary = _radar()),
                       5.0, 5.0, 100.0) === (0.0, 0.0)
end

@testset "the radar offset against isce3" begin
    # A real Sentinel-1 pair, 24 days apart, against `isce3.geometry.geo2rdr` run at a tighter
    # threshold than its default so the recorded value is the converged fixed point. See
    # `reference/gen_rdr2rdr.py`; the fixture holds the state vectors, so this needs no granule.
    #
    # geogrid has no equivalent operation to compare against — it assumes the secondary is already
    # resampled onto the reference's grid, which is the gap this whole path exists to close — so isce3
    # is the reference here rather than the compiled kernel.
    fx = JSON3.read(read(joinpath(@__DIR__, "reference", "rdr2rdr.json"), String))
    @test fx.versions.isce3 == "0.25.12"

    ref, sec = _fixture_coord(fx.ref), _fixture_coord(fx.sec)
    pair = CoregisteredPair(ref; dt = 24 * 86400.0, secondary = sec)

    # The fixture's target is ECEF; `pixel_offset` takes a grid point, so it is passed as lon/lat
    # degrees under the identity transform — the same convention `test/radar_coordinate.jl` uses to
    # isolate the radar solve from PROJ.
    xyz = SVector{3,Float64}(Float64.(fx.target_xyz)...)
    lonlat = xyz_to_lonlat(Ellipsoid(), xyz)
    ds, dl = pixel_offset(pair, rad2deg(lonlat[1]), rad2deg(lonlat[2]), lonlat[3])

    # 1.15e-9 of a range sample is 2.7 nm of ground position, which is the 1.9e-9 m REFERENCE.md
    # establishes for the ellipsoid conversions rather than anything this composition adds.
    @test ds ≈ fx.isce3.dsamp atol = 1e-8
    @test dl ≈ fx.isce3.dline atol = 1e-8
    # Reported so a platform difference shows up as a number rather than as a pass or a fail.
    @info "pixel_offset vs isce3" dsamp_error = abs(ds - fx.isce3.dsamp) dline_error = abs(dl - fx.isce3.dline)

    # The magnitudes this pair is documented with: about 17.8 samples of range and −2.8 lines of
    # azimuth. Asserted loosely, as a guard against a sign flip or a swapped axis rather than as a
    # precision claim — the tight comparison is above.
    @test 17.0 < ds < 18.5
    @test -3.0 < dl < -2.5
end

@testset "the field is lazy and window-shaped" begin
    g = MapGrid(geotransform = (0.0, 120.0, 0.0, 12000.0, 0.0, -120.0), size = (100, 100))
    c = _proj(spacing = (30.0, -30.0), size = (400, 400))
    p = CoregisteredPair(c; dt = 86400.0, secondary = c)
    w = CartesianIndices((5:60, 8:70))

    f = OffsetField(p, g, w; dem = 0.0)
    # Shaped like the window, not like the grid, so it lines up with a `PairGeometry` over the same
    # window and can be read a block at a time.
    @test size(f) == size(w) == (56, 63)
    @test f isa AbstractMatrix{NTuple{2,Float64}}
    @test eltype(f) === NTuple{2,Float64}

    # A `view` is a block of the field, indexed within it — the same convention `GeometryInputs` uses.
    sub = view(f, 3:9, 4:11)
    @test size(sub) == (7, 8)
    @test sub[2, 2] === f[4, 5]

    # An elevation array must cover the window exactly, as the kernel's inputs must.
    @test_throws DimensionMismatch OffsetField(p, g, w; dem = zeros(3, 3))
    @test OffsetField(p, g, w; dem = zeros(size(w))) isa OffsetField

    # A pair with no secondary is refused when the field is built, not when an element is read: the
    # mistake is in the call that built it.
    @test_throws "carrying a secondary" OffsetField(
        CoregisteredPair(c; dt = 1.0), g, w; dem = 0.0)
    @test_throws "carrying a secondary" LatticeOffsetField(
        CoregisteredPair(c; dt = 1.0), g, w; dem = 0.0)
end

@testset "the lattice field agrees with the exact one" begin
    # On the projected path both are identically zero, which checks the plumbing — the lattice is built,
    # queried at every point of the window including its corners, and returns the same answer. The
    # accuracy question is a radar one and is measured below.
    g = MapGrid(geotransform = (0.0, 120.0, 0.0, 12000.0, 0.0, -120.0), size = (100, 100))
    c = _proj(spacing = (30.0, -30.0), size = (400, 400))
    p = CoregisteredPair(c; dt = 86400.0, secondary = c)
    w = CartesianIndices((2:64, 3:50))

    exact = OffsetField(p, g, w; dem = 0.0)
    for L in (1, 8, 32)
        lat = LatticeOffsetField(p, g, w; dem = 0.0, lattice = L)
        @test size(lat) == size(exact)
        # Every element, corners included: a lattice sized without slack refuses a corner query rather
        # than clamping, and that is exactly the boundary this asserts.
        @test all(lat[i] === exact[i] for i in eachindex(exact))
    end

    @test_throws "at least 1 grid spacing" LatticeOffsetField(p, g, w; dem = 0.0, lattice = 0)
end

@testset "the lattice field on radar geometry" begin
    # The accuracy claim `LatticeOffsetField`'s docstring tabulates, on the real pair: a coarse lattice
    # reproduces the exact field far below what a correlator resolves, because the offset is set by two
    # orbits' geometry and varies over kilometres rather than pixels.
    fx = JSON3.read(read(joinpath(@__DIR__, "reference", "rdr2rdr.json"), String))
    pair = CoregisteredPair(_fixture_coord(fx.ref); dt = 24 * 86400.0,
                            secondary = _fixture_coord(fx.sec))
    lonlat = xyz_to_lonlat(Ellipsoid(), SVector{3,Float64}(Float64.(fx.target_xyz)...))
    lon, lat = rad2deg(lonlat[1]), rad2deg(lonlat[2])

    # A lon/lat grid at about 200 m spacing around scene center, under the identity transform.
    g = MapGrid(geotransform = (lon - 0.064, 0.002, 0.0, lat + 0.064, 0.0, -0.002), size = (64, 64))
    w = CartesianIndices((1:64, 1:64))
    E = collect(OffsetField(pair, g, w; dem = 500.0))

    # The offset varies by about a sample across this window, so the errors below are against a real
    # signal rather than against a constant.
    @test maximum(x -> x[1], E) - minimum(x -> x[1], E) > 0.1

    worst = Float64[]
    for L in (4, 16, 64)
        A = collect(LatticeOffsetField(pair, g, w; dem = 500.0, lattice = L))
        push!(worst, maximum(abs(A[i][1] - E[i][1]) for i in eachindex(E)))
        # A thousandth of a sample is nanometres of ground position; nothing downstream can see it.
        @test worst[end] < 5e-3
        @test maximum(abs(A[i][2] - E[i][2]) for i in eachindex(E)) < 1e-4
    end
    # Coarser is worse at the coarse end, where the node spacing dominates. It is *not* monotone across
    # the whole range under the default `zrange`, because at fine spacings the error is the elevation
    # interpolation across 4200 m rather than the lattice — the floor `LatticeOffsetField`'s docstring
    # tabulates. So the claim asserted is the one that holds: the coarsest is worse than the finest.
    @test worst[end] > worst[1]
    @info "lattice offset error, samples" l4 = worst[1] l16 = worst[2] l64 = worst[3]

    # Narrowing `zrange` to the relief present removes the elevation-interpolation floor the docstring
    # records, which is the error that dominates at fine spacings under the default range.
    tight = collect(LatticeOffsetField(pair, g, w; dem = 500.0, lattice = 4,
                                       zrange = (400.0, 600.0)))
    @test maximum(abs(tight[i][1] - E[i][1]) for i in eachindex(E)) < worst[1]
end

@testset "the field is coordinate-agnostic" begin
    # The spine — the field types, and the lattice and fit above them — must hold no radar concept, so
    # that a coordinate system added later inherits all of it. This asserts that by driving it with a
    # coordinate type declared *here*, in the test, which the package has never seen: a closed-form
    # offset over a toy coordinate, in the way `AffineTransform` lets the kernel's arithmetic be tested
    # without PROJ.
    #
    # If this needs an orbit, a solve, or a product to run, the abstraction is in the wrong place.
    p = CoregisteredPair(_ToyCoordinate(), _ToyCoordinate(), (0, 0), (0, 0), 86400.0)
    g = MapGrid(geotransform = (0.0, 100.0, 0.0, 5000.0, 0.0, -100.0), size = (50, 50))
    w = CartesianIndices((4:40, 6:44))

    exact = OffsetField(p, g, w; dem = 0.0)
    # The toy offset is affine in position, which bilinear interpolation reproduces exactly — so the
    # lattice must agree to rounding at any spacing, and a discrepancy is a plumbing error rather than
    # an accuracy limit.
    for L in (2, 16)
        A = LatticeOffsetField(p, g, w; dem = 0.0, lattice = L)
        @test all(isapprox(A[i][k], exact[i][k]; atol = 1e-9) for i in eachindex(exact), k in 1:2)
    end
    # And it is a real signal, not zeros — otherwise the agreement above would be vacuous.
    @test abs(exact[1, 1][1] - exact[end, end][1]) > 1.0
end

@testset "height sensitivity is small for a near-repeat pass" begin
    # The measure of whether the offset is a pixel shift or mostly parallax. For two orbits a repeat
    # cycle apart the look directions nearly agree, so the offset barely moves with elevation — which
    # is what makes a DEM optional on this pair and is the property that degrades as the baseline
    # grows.
    fx = JSON3.read(read(joinpath(@__DIR__, "reference", "rdr2rdr.json"), String))
    pair = CoregisteredPair(_fixture_coord(fx.ref); dt = 24 * 86400.0,
                            secondary = _fixture_coord(fx.sec))
    lonlat = xyz_to_lonlat(Ellipsoid(), SVector{3,Float64}(Float64.(fx.target_xyz)...))
    lon, lat = rad2deg(lonlat[1]), rad2deg(lonlat[2])

    dsdh, dldh = height_sensitivity(pair, lon, lat)
    # 3.4e-5 samples per meter: 0.07 samples across 2 km of relief, so terrain cannot move a pixel.
    @test 0 < dsdh < 1e-4
    @test abs(dldh) < 1e-5
    @info "height sensitivity" samples_per_m = dsdh lines_per_m = dldh

    # It is a difference of two offsets across `heights`, so it must agree with taking that difference
    # directly. Over the same interval it is exact; over a different one it agrees only to the curvature
    # of the height dependence, which is what the second assertion allows for.
    z0, z1 = 0.0, 1000.0
    s0, _ = pixel_offset(pair, lon, lat, z0)
    s1, _ = pixel_offset(pair, lon, lat, z1)
    @test height_sensitivity(pair, lon, lat; heights = (z0, z1))[1] ≈ (s1 - s0) / (z1 - z0) rtol = 1e-12
    # The default interval is `DEFAULT_ZRANGE`, a wider span over which the slope is nearly but not
    # exactly the same — so the two differ by well under a percent rather than to machine precision.
    @test dsdh ≈ (s1 - s0) / (z1 - z0) rtol = 1e-2

    # Two identical heights have no slope to report.
    @test_throws "two distinct heights" height_sensitivity(pair, lon, lat; heights = (0.0, 0.0))
end
