# The reference ellipsoid, and the orthonormal frame the range-Doppler solve works in.
#
# The two ECEF conversions delegate to FastGeoProjections, which implements the same Vermeille 2002
# closed form for the inverse. Delegating to PROJ instead is not an option at the altitudes this
# package works at: PROJ's EPSG:4978-to-EPSG:4979 pipeline returns a height 4.0e-3 m out at 700 km,
# where Vermeille's form recovers an exactly-computed position to 1.8e-9 m.
#
# isce3 sets `e2 = 0.0066943799901` (`geogridRadar.cpp:324-325`), truncated at eight significant
# digits from WGS84's twelve. The full value is used here instead, and that choice dominates every
# other difference: it moves a position 1.5e-7 m, seventy-nine times the 1.9e-9 m by which this
# arithmetic and isce3's differ. Comparisons against the isce3 fixture are therefore to a tolerance
# rather than to the bit, on isce3's own constant as well as on this one -- see `REFERENCE.md`.
#
# Angles are radians and ordered `(lon, lat, height)` throughout, as isce3 orders them. That is not
# the order the surrounding package uses for map coordinates, and the two meet at the transform
# boundary in `geo2rdr.jl`, which is why the conversion helpers here are explicit about which they
# take. FastGeoProjections takes degrees, so the conversion happens at this boundary and nowhere
# else.

"""
    WGS84_A

Semi-major axis in meters: `6378137.0`.
"""
const WGS84_A = 6378137.0

"""
    WGS84_E2

Eccentricity squared: `6.69437999014e-3`, the WGS84 value at full precision.

isce3 truncates this to `0.0066943799901` at eight significant digits. The difference is 6e-12
relative and reaches a position as 1.5e-7 m, well inside the tolerance every downstream comparison
uses.
"""
const WGS84_E2 = 6.69437999014e-3

"""
    Ellipsoid(a, e2)
    Ellipsoid()

A biaxial ellipsoid, stored as semi-major axis and eccentricity squared with every other quantity
derived.

The no-argument form is WGS84: [`WGS84_A`](@ref) and [`WGS84_E2`](@ref).

Mirrors `isce3::core::Ellipsoid` (`Ellipsoid.h`).
"""
struct Ellipsoid
    a::Float64
    e2::Float64
end

Ellipsoid() = Ellipsoid(WGS84_A, WGS84_E2)

"""
    semiminor(el::Ellipsoid) -> Float64

Semi-minor axis, `a * sqrt(1 - e2)` (`Ellipsoid.h:41`).
"""
@inline semiminor(el::Ellipsoid) = el.a * sqrt(1.0 - el.e2)

"""
    r_east(el::Ellipsoid, lat) -> Float64

Prime vertical radius of curvature at geodetic latitude `lat` in radians (`Ellipsoid.h:150-153`).
"""
@inline r_east(el::Ellipsoid, lat::Float64) = r_east_sin(el, sin(lat))

"""
    r_east_sin(el::Ellipsoid, slat) -> Float64

[`r_east`](@ref) from an already-computed `sin(lat)`, for a caller that needs the sine anyway.
"""
@inline r_east_sin(el::Ellipsoid, slat::Float64) = el.a / sqrt(1.0 - (el.e2 * slat^2))

"""
    lonlat_to_xyz(el::Ellipsoid, llh) -> SVector{3,Float64}

ECEF position in meters from `llh = (lon, lat, height)` with the angles in radians.

Evaluated by `FastGeoProjections.LonLatToGeocentric`, which takes degrees; the conversion is here so
that the radians convention the radar solves use stops at this boundary.
"""
@inline function lonlat_to_xyz(el::Ellipsoid, llh::SVector{3,Float64})
    x, y, z = _fgp_forward(el)(rad2deg(llh[1]), rad2deg(llh[2]), llh[3])
    return SVector{3,Float64}(x, y, z)
end

"""
    xyz_to_lonlat(el::Ellipsoid, xyz) -> SVector{3,Float64}

`(lon, lat, height)` with the angles in radians, from an ECEF position in meters.

Vermeille's 2002 closed form, following `Ellipsoid.h:210-238`. Agrees with PROJ's
EPSG:4978 to EPSG:4979 pipeline to 9e-7 m over latitudes, longitudes and heights.

The cube root is `cbrt`, not the `x^(1/3)` isce3 spells (`std::pow(x, 1./3.)`): the two agree to
1.4e-9 in the result and `cbrt` is the faster of them, since `pow` has no way to know the exponent
is a cube root.

Angles come back within one ULP of a platform's own `atan2` rather than bitwise equal to it. Julia's
`atan` is openlibm's, and openlibm is kept deliberately: both it and any system libm are faithful
and neither is correctly rounded, measured against a 256-bit evaluation, so matching a particular
machine would trade openlibm's cross-platform reproducibility for agreement with whichever libm
happens to be running. One ULP of angle here is 5e-10 m of position, against a range sample of about
2.3 m. See `REFERENCE.md`.
"""
@inline function xyz_to_lonlat(el::Ellipsoid, xyz::SVector{3,Float64})
    lon, lat, h = _fgp_inverse(el)(xyz[1], xyz[2], xyz[3])
    return SVector{3,Float64}(deg2rad(lon), deg2rad(lat), h)
end

# The two FastGeoProjections operators for an ellipsoid, built per call.
#
# Constructing one is a handful of arithmetic operations on the ellipsoid's own fields -- no table
# lookup and no PROJ context -- so this is not a cache, and measuring it against a `const` pair for
# WGS84 showed no difference. Building per call is what keeps a non-WGS84 `Ellipsoid` working, which
# `test/radar_numerics.jl` exercises on isce3's truncated `e2`.
@inline _fgp_ellipsoid(el::Ellipsoid) =
    FGP.Ellipsoid(el.a, el.a * sqrt(1.0 - el.e2), 1.0 - sqrt(1.0 - el.e2),
                  sqrt(el.e2), el.e2, nothing, FGP.EPSG(0))
@inline _fgp_forward(el::Ellipsoid) = FGP.LonLatToGeocentric(ellips = _fgp_ellipsoid(el))
@inline _fgp_inverse(el::Ellipsoid) = FGP.GeocentricToLonLat(ellips = _fgp_ellipsoid(el))

"""
    TCNBasis

An orthonormal frame at a satellite position: along-track, cross-track and nadir-pointing.

# Fields
- `that`: along-track, `c × n` normalized.
- `chat`: cross-track, `n × v` normalized.
- `nhat`: geocentric nadir, `-x` normalized — pointing *down*, toward the Earth's center.

The range-Doppler solve expresses the satellite-to-target vector in this frame, so the sign of
`nhat` and the handedness of the pair are load-bearing: `beta`'s sign carries the look side against
`chat`, and flipping either would place the target on the wrong side of the track.
"""
struct TCNBasis
    that::SVector{3,Float64}
    chat::SVector{3,Float64}
    nhat::SVector{3,Float64}
end

"""
    geodetic_tcn(pos, vel) -> TCNBasis

The geocentric TCN frame at satellite position `pos` with velocity `vel`, both ECEF.

Despite the name — inherited from `isce3::core::Basis`'s constructor — this frame is geo*centric*:
`nhat` is antiparallel to the position vector, not to the ellipsoid normal. The two differ by up to
0.19° at mid-latitudes. The range-Doppler solve compensates by working against a local sphere of
radius `radius`, computed per point in [`geo2rdr`](@ref)'s caller.

`geogridRadar.cpp:1027-1034` and `Basis.h:50-58` build this identically, the former inline with
`unitvec_C2` and the latter with Eigen's `normalized()`. Both divide by the norm rather than scaling
by its reciprocal, so [`unitvec3`](@ref) serves both.
"""
@inline function geodetic_tcn(pos::SVector{3,Float64}, vel::SVector{3,Float64})
    nhat = -unitvec3(pos)
    chat = unitvec3(cross3(nhat, vel))
    that = unitvec3(cross3(chat, nhat))
    return TCNBasis(that, chat, nhat)
end

"""
    nadir_sphere(el::Ellipsoid, pos) -> NTuple{4,Float64}

`(radius, height, eta, sat_dist)` for the sphere osculating the ellipsoid beneath the satellite at
ECEF position `pos`.

`radius` is the local radius directly below the satellite and `height` the satellite's height above
it; `eta` is the scale factor relating the two to the geocentric distance `sat_dist`. The
range-Doppler solve iterates on a target height measured from this sphere rather than from the
ellipsoid, which is why it needs all of them — `sat_dist` is returned rather than left to the caller
to recompute, since it is `norm3(pos)` and already in hand here.

`geogridRadar.cpp:1016-1024` and `Rdr2Geo.icc:69-79` agree term for term.
"""
@inline function nadir_sphere(el::Ellipsoid, pos::SVector{3,Float64})
    sat_dist = norm3(pos)
    major = el.a
    minor = semiminor(el)
    # Not `norm3`: the components are scaled by different axes, so this is the ellipsoidal
    # gauge of the position rather than a Euclidean length.
    temp = SVector{3,Float64}(pos[1] / major, pos[2] / major, pos[3] / minor)
    eta = 1.0 / norm3(temp)
    return (eta * sat_dist, (1.0 - eta) * sat_dist, eta, sat_dist)
end
