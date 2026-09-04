# The radar image coordinate system, and the two scene-level quantities derived from it.
#
# Where `ProjectedCoordinate` is an origin, a spacing and a size — an affine relation to map
# coordinates — a radar acquisition is a range axis, a time axis, and an orbit. There is no
# geotransform, so the two scene-level quantities the projected path reads off one have to be solved
# for instead: the footprint comes from `rdr2geo` at the swath corners, and the pixel sizes from the
# incidence angle and the platform speed.
#
# Both are scene metadata, not per-point geometry: one incidence angle serves every grid point
# (`GeogridRadar.py:297`), and the footprint is a bounding box.

"""
    RadarCoordinate(; starting_range, dr, sensing_start, prf, nsamples, nlines,
                      look_side, wavelength, orbit, incidence_angle, orbit_epoch_offset = 0.0)

An image in radar slant-range/azimuth coordinates.

# Fields
- `starting_range`: slant range to the first range sample, in meters.
- `dr`: range sample spacing in meters — the reference's `rangePixelSize`.
- `sensing_start`: azimuth time of the first line, **in seconds since midnight** of the acquisition
  day. That is the scale `GeogridRadar.py:328-330` computes and the one the azimuth index is measured
  against; see `orbit_epoch_offset` for the other clock.
- `prf`: pulse repetition frequency in Hz, so `1/prf` is the line spacing in time.
- `nsamples`, `nlines`: image width and height in pixels.
- `look_side`: [`LookLeft`](@ref) or [`LookRight`](@ref).
- `wavelength`: radar wavelength in meters. Used only through the Doppler term, which is zero for
  the zero-Doppler geometry the reference uses, so it does not affect the result — carried because
  `rdr2geo` takes it and a nonzero Doppler would.
- `orbit`: the [`Orbit`](@ref) covering the acquisition.
- `incidence_angle`: scene-center incidence angle in radians, from [`incidence_angle`](@ref). One
  scalar for the whole scene, as the reference computes it.
- `orbit_epoch_offset`: seconds to add to a `sensing_start`-scale time to reach the orbit's scale.

Unlike [`ProjectedCoordinate`](@ref) this describes the *reference* image alone, not an overlap:
`testGeogrid.py:427-470` takes every radar parameter from image 1 and the secondary only for the
repeat interval. There is no radar analogue of [`coregister`](@ref).

# The two time scales

`sensing_start` and the orbit are on different clocks, and `orbit_epoch_offset` is the constant
between them. The reference carries the same split: it measures the azimuth index against
seconds-since-midnight (`geogridRadar.cpp:972`) while interpolating the orbit against
seconds-since-reference-epoch (`:432`). Keeping both means neither the index nor the interpolation
needs a conversion at the point of use. See [`geo2rdr`](@ref) and `REFERENCE.md`.
"""
struct RadarCoordinate{O} <: AbstractImageCoordinate
    starting_range::Float64
    dr::Float64
    sensing_start::Float64
    prf::Float64
    nsamples::Int
    nlines::Int
    look_side::LookSide
    wavelength::Float64
    orbit::O
    incidence_angle::Float64
    orbit_epoch_offset::Float64

    function RadarCoordinate{O}(starting_range, dr, sensing_start, prf, nsamples, nlines,
                                look_side, wavelength, orbit, incidence_angle,
                                orbit_epoch_offset) where {O}
        starting_range > 0 || throw(ArgumentError(
            "RadarCoordinate starting_range must be positive, got $starting_range m"))
        dr > 0 || throw(ArgumentError(
            "RadarCoordinate dr must be positive, got $dr m"))
        prf > 0 || throw(ArgumentError(
            "RadarCoordinate prf must be positive, got $prf Hz"))
        nsamples > 0 && nlines > 0 || throw(ArgumentError(
            "RadarCoordinate size must be positive, got $(nsamples)x$(nlines)"))
        wavelength > 0 || throw(ArgumentError(
            "RadarCoordinate wavelength must be positive, got $wavelength m"))
        # A zero incidence angle would divide by zero in `xsize`; a right angle means looking
        # straight down, which no side-looking radar does.
        0 < incidence_angle < pi / 2 || throw(ArgumentError(
            "RadarCoordinate incidence_angle must be in (0, π/2) radians, got " *
            "$incidence_angle ($(rad2deg(incidence_angle))°)"))
        return new{O}(starting_range, dr, sensing_start, prf, nsamples, nlines, look_side,
                      wavelength, orbit, incidence_angle, orbit_epoch_offset)
    end
end

function RadarCoordinate(; starting_range, dr, sensing_start, prf, nsamples, nlines,
                         look_side, wavelength, orbit, incidence_angle,
                         orbit_epoch_offset = 0.0)
    return RadarCoordinate{typeof(orbit)}(
        Float64(starting_range), Float64(dr), Float64(sensing_start), Float64(prf),
        Int(nsamples), Int(nlines), look_side, Float64(wavelength), orbit,
        Float64(incidence_angle), Float64(orbit_epoch_offset))
end

"""
    nsamples(c::RadarCoordinate)
    nlines(c::RadarCoordinate)

Image width and height in pixels — range samples and azimuth lines.
"""
nsamples(c::RadarCoordinate) = c.nsamples
nlines(c::RadarCoordinate) = c.nlines

"""
    y_displacement_sign(c::RadarCoordinate)

`-1.0`: azimuth increases along the track, while a north-up raster's `+y` points down, so the two
conventions oppose.
"""
y_displacement_sign(::RadarCoordinate) = -1.0

"""
    xsize(c::RadarCoordinate)
    ysize(c::RadarCoordinate)

Ground pixel size along each axis, in meters: `dr / sin(incidence)` across track and
`|v| / prf` along it.

Matches `geogridRadar.cpp:684-686`, which reports these as `grd_res`/`azm_res` where the projected
path reports `X_res`/`Y_res`. So the chip-size conversion consumes them identically on both paths —
these are the radar equivalents of a geotransform's pixel size, recovered from geometry because
there is no geotransform to read them off.

The azimuth size uses the platform speed at scene center, so it is a single value for the scene
rather than a per-point quantity. Specifically it uses `satvmid` (`geogridRadar.cpp:686`), which is
interpolated at `tmidd` (`:435`) — the *orbit*-clock midpoint from [`orbit_midtime`](@ref), not the
`sensing_start`-clock one from [`midtime`](@ref). The two differ by a pulse interval, so the choice
changes `azm_res` in the last bits.
"""
xsize(c::RadarCoordinate) = c.dr / sin(c.incidence_angle)

function ysize(c::RadarCoordinate)
    _, vel = interpolate(c.orbit, orbit_midtime(c))
    return norm3(vel) / c.prf
end

"""
    midtime(c::RadarCoordinate) -> Float64

Azimuth time of the scene's middle line, on the `sensing_start` clock.

`sensing_start + 0.5 * nlines / prf`, matching `geogridRadar.cpp:328`, which is the initial guess
every per-point solve starts from. Note this is *not* the expression
`GeogridRadar.py:346-347` uses to build the orbit-clock midpoint — see [`orbit_midtime`](@ref).
"""
midtime(c::RadarCoordinate) = c.sensing_start + 0.5 * c.nlines / c.prf

"""
    orbit_midtime(c::RadarCoordinate) -> Float64

Scene midpoint on the *orbit's* clock, as `GeogridRadar.py:346-347` computes it:
`sensing_start + (floor(nlines / 2) - 1) / prf`, shifted by `orbit_epoch_offset`.

Deliberately a different expression from [`midtime`](@ref), and the difference reaches the output.
Both are offsets from `sensing_start` on their own clocks, so the epoch cancels from
`midtime - orbit_midtime`, leaving `0.5 * nlines / prf` against `(floor(nlines / 2) - 1) / prf` — one
pulse interval apart for an even line count.

`geo2rdr` preserves that difference, so the converged azimuth time corresponds to a satellite position
one pulse from where the orbit was interpolated, and the azimuth index is one line off the
geometrically consistent value. Reproduced rather than reconciled: the reference's products contain
that offset. See `REFERENCE.md`.
"""
orbit_midtime(c::RadarCoordinate) =
    c.sensing_start + (floor(c.nlines / 2) - 1) / c.prf + c.orbit_epoch_offset

"""
    sensing_stop(c::RadarCoordinate) -> Float64

Azimuth time of the last line: `sensing_start + (nlines - 1) / prf` (`GeogridRadar.py:193`).
"""
sensing_stop(c::RadarCoordinate) = c.sensing_start + (c.nlines - 1) / c.prf

"""
    FOOTPRINT_RANGE_SAMPLES

Number of range positions the footprint is evaluated at: 21, from
`np.linspace(0, numberOfSamples - 1, num=21)` (`GeogridRadar.py:152`).
"""
const FOOTPRINT_RANGE_SAMPLES = 21

"""
    FOOTPRINT_AZIMUTH_SAMPLES

Number of azimuth fractions the footprint's intermediate lines are taken at: 21, of which the first
and last are dropped by `[1:-1]` (`GeogridRadar.py:153`), leaving 19.
"""
const FOOTPRINT_AZIMUTH_SAMPLES = 21

"""
    footprint_bounds(transform, c::RadarCoordinate; zrange = DEFAULT_ZRANGE) -> Extent

Bounding box, in grid coordinates, of the swath described by `c`.

`transform` maps geodetic degrees to grid coordinates, called as `transform(lon, lat, h)`. Unlike
the projected path — where the image's four corners transform directly — a radar footprint has to be
*solved* for: each sample point is a `rdr2geo` call at one azimuth time and slant range.

Reproduces `GeogridRadar.determineBbox` (`GeogridRadar.py:140-251`), including its sampling exactly:
21 range positions on the first line, the same 21 on the last, and the two range edges of the 19
intermediate azimuth fractions — 80 positions, each at both elevations in `zrange`, so 160 `rdr2geo`
solves. A coarser sampling would move the bounding box and shift the grid window, so the count is
transcribed rather than chosen.

`zrange` widens nothing for a grid on WGS84, for the reason [`DEFAULT_ZRANGE`](@ref) records, but is
evaluated anyway because the reference evaluates it.
"""
function footprint_bounds(transform, c::RadarCoordinate; zrange = DEFAULT_ZRANGE)
    el = Ellipsoid()

    # `np.linspace(0, nsamples - 1, num=21)` — endpoints included, so the last is the far edge.
    ranges = ntuple(FOOTPRINT_RANGE_SAMPLES) do i
        frac = (i - 1) / (FOOTPRINT_RANGE_SAMPLES - 1)
        return c.starting_range + frac * (c.nsamples - 1) * c.dr
    end

    # The azimuth times, in the reference's order: the first and last range lines
    # (`GeogridRadar.py:169-215`), then the 19 intermediate fractions its `linspace(0, 1, 21)[1:-1]`
    # leaves (`:219`). Built as a tuple so the sampling loop below is one flat pass.
    aztimes = (c.sensing_start, sensing_stop(c),
               ntuple(FOOTPRINT_AZIMUTH_SAMPLES - 2) do i
                   frac = i / (FOOTPRINT_AZIMUTH_SAMPLES - 1)
                   return c.sensing_start + frac * (c.nlines - 1) / c.prf
               end...)

    # Unpacked once, and the sampling written as one loop rather than through a helper. Neither is
    # cosmetic: `Orbit` and `RadarCoordinate` both hold `Vector`s, so neither is `isbits`, and passing
    # either into a function the compiler declines to inline copies it to the heap once per call. With
    # the helper this cost 21 kB and 160 µs; inline it is 0.7 kB and 74 µs.
    orbit = c.orbit
    wvl = c.wavelength
    side = c.look_side
    epoch = c.orbit_epoch_offset

    xmin = ymin = Inf
    xmax = ymax = -Inf

    for (ai, t) in pairs(aztimes)
        # The first two times are full range lines; the rest are sampled at the two range edges only,
        # since the intermediate lines contribute nothing between them.
        rs = ai <= 2 ? ranges : (ranges[1], ranges[end])
        for rng in rs, z in zrange
            llh = rdr2geo(orbit, el, t + epoch, rng; height = z, wavelength = wvl, side)
            gx, gy, _ = transform(rad2deg(llh[1]), rad2deg(llh[2]), llh[3])
            xmin = min(xmin, gx); xmax = max(xmax, gx)
            ymin = min(ymin, gy); ymax = max(ymax, gy)
        end
    end

    isfinite(xmin) && isfinite(ymin) || throw(ArgumentError(
        "footprint_bounds got a non-finite result for a radar footprint: " *
        "X = ($xmin, $xmax), Y = ($ymin, $ymax). The swath most likely falls outside the " *
        "transform's area of validity."))

    return Extent(X = (xmin, xmax), Y = (ymin, ymax))
end

# A `TransformPair`'s inverse is the image-to-grid direction, matching the projected path's method.
footprint_bounds(tf::TransformPair, c::RadarCoordinate; zrange = DEFAULT_ZRANGE) =
    footprint_bounds(tf.inverse, c; zrange)

"""
    incidence_angle(orbit, el, aztime, range; wavelength, side,
                    zrange = DEFAULT_ZRANGE) -> Float64

Incidence angle in radians: the angle at the target between the line of sight to the satellite and
the local ellipsoid normal.

Reproduces `GeogridRadar.getIncidenceAngle` (`GeogridRadar.py:253-297`): solve for the ground point
at each elevation in `zrange`, take the angle there, and return the **mean** of the two. Averaging
two elevations rather than evaluating at the terrain is the reference's choice, and since one scalar
then serves every grid point through [`xsize`](@ref), it is what the ground range spacing rests on.

The normal is the geodetic one, `(cos φ cos λ, cos φ sin λ, sin φ)` — not the geocentric direction
the TCN basis uses.
"""
function incidence_angle(orbit::Orbit, el::Ellipsoid, aztime::Real, range::Real;
                         wavelength::Real, side::LookSide, zrange = DEFAULT_ZRANGE)
    pos, _ = interpolate(orbit, aztime)

    total = 0.0
    for z in zrange
        llh = rdr2geo(orbit, el, aztime, range; height = z, wavelength, side)
        target = lonlat_to_xyz(el, llh)
        los = unitvec3(pos - target)
        lon, lat = llh[1], llh[2]
        nvec = SVector{3,Float64}(cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat))
        total += acos(dot3(los, nvec))
    end
    return total / length(zrange)
end

"""
    incidence_angle(; orbit, starting_range, dr, sensing_start, prf, nsamples, nlines,
                      look_side, wavelength, orbit_epoch_offset = 0.0,
                      zrange = DEFAULT_ZRANGE) -> Float64

Scene-center incidence angle, for the parameters that will build a [`RadarCoordinate`](@ref).

Separate from the type because the type *stores* the result: the reference calls
`getIncidenceAngle()` before `geogridRadar()` (`testGeogrid.py:487-488`), so the angle is an input to
the geometry rather than something derived on demand.

Scene center is `floor(nsamples / 2) - 1` in range and `floor(nlines / 2) - 1` in azimuth
(`GeogridRadar.py:265-266`) — one sample short of the true midpoint at both, reproduced as written.
"""
function incidence_angle(; orbit, starting_range, dr, sensing_start, prf, nsamples, nlines,
                         look_side, wavelength, orbit_epoch_offset = 0.0,
                         zrange = DEFAULT_ZRANGE)
    midrange = starting_range + (floor(nsamples / 2) - 1) * dr
    midsensing = sensing_start + (floor(nlines / 2) - 1) / prf
    return incidence_angle(orbit, Ellipsoid(), midsensing + orbit_epoch_offset, midrange;
                           wavelength, side = look_side, zrange)
end
