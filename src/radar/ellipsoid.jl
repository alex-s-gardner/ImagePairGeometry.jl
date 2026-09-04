# The reference ellipsoid, and the orthonormal frame the range-Doppler solve works in.
#
# ECEF to geodetic has no closed form; this is Vermeille's 2002 solution, which differs from
# GeographicLib's series in the last bits everywhere. It agrees with PROJ to 6e-9 m forward and
# 9e-7 m inverse across latitudes, longitudes and heights.
#
# isce3 sets `e2 = 0.0066943799901` (`geogridRadar.cpp:324-325`), truncated at eight significant
# digits from WGS84's twelve. The full value is used here instead: the difference reaches positions
# as 1.5e-7 m, which is 6e-8 of a range sample, so it moves no rounded index and every reference
# output band still matches. Comparisons against the isce3 fixture are therefore to a tolerance
# rather than to the bit.
#
# Angles are radians and ordered `(lon, lat, height)` throughout, as isce3 orders them. That is not
# the order the surrounding package uses for map coordinates, and the two meet at the transform
# boundary in `geo2rdr.jl`, which is why the conversion helpers here are explicit about which they
# take.

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

Transcribed from `Ellipsoid.h:195-208`.
"""
@inline function lonlat_to_xyz(el::Ellipsoid, llh::SVector{3,Float64})
    lon, lat, h = llh[1], llh[2], llh[3]
    # `sincos` returns the pair in one call. Each value is bit-identical to the separate `sin`/`cos`
    # it replaces — and `sin(lat)` is needed twice, once for the prime vertical radius and once for
    # the z component, where the reference also evaluates it twice.
    slat, clat = sincos(lat)
    slon, clon = sincos(lon)
    re = r_east_sin(el, slat)
    return SVector{3,Float64}(
        (re + h) * clat * clon,
        (re + h) * clat * slon,
        ((re * (1.0 - el.e2)) + h) * slat,
    )
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
    x, y, z = xyz[1], xyz[2], xyz[3]
    e2 = el.e2
    e4 = e2 * e2
    a2 = el.a * el.a

    # Squared lateral distance, needed here and again for `d` below. The reference spells
    # `std::pow(xyz[0], 2) + std::pow(xyz[1], 2)` out at both sites; one evaluation gives the same
    # value, since each was already deterministic.
    lat2 = x^2 + y^2

    # Lateral and polar distances, normalized by the axes.
    p = lat2 / a2
    q = ((1.0 - e2) * z^2) / a2
    r = (p + q - e4) / 6.0
    s = (e4 * p * q) / (4.0 * r^3)
    t = cbrt(1.0 + s + sqrt(s * (2.0 + s)))
    u = r * (1.0 + t + (1.0 / t))
    rv = sqrt(u^2 + (e4 * q))
    w = (e2 * (u + rv - q)) / (2.0 * rv)
    k = sqrt(u + rv + w^2) - w
    d = (k * sqrt(lat2)) / (k + e2)

    return SVector{3,Float64}(
        atan(y, x),
        atan(z, d),
        ((k + e2 - 1.0) * sqrt(d^2 + z^2)) / k,
    )
end

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
