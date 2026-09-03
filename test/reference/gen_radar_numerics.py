#!/usr/bin/env python
"""Generate fixtures for the radar numerics: ellipsoid, orbit interpolation, and rdr2geo.

Calls isce3 directly — `isce3.core.Ellipsoid`, `isce3.core.Orbit`, `isce3.geometry.rdr2geo` — the
same objects `geogridRadar.cpp` links against. These three have callable reference implementations,
unlike `geo2rdr` and the range-Doppler solve, which exist only inside the compiled kernel and are
covered by the whole-kernel fixture instead.

Every float is recorded as a hex literal alongside its decimal form. `float.hex()` round-trips
exactly and JSON's decimal repr does not reliably, and these fixtures are the basis for bitwise
assertions, so the hex form is what the tests read.

Run in the reference environment:

    micromamba run -n geogrid-ref python test/reference/gen_radar_numerics.py
"""
import json
import os

import numpy as np
import isce3
from isce3.core import Ellipsoid, Orbit, StateVector, DateTime, LookSide
from isce3.geometry import rdr2geo, DEMInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))

# The reference's ellipsoid, with its truncated e2 (`geogridRadar.cpp:324-325`).
A = 6378137.0
E2 = 0.0066943799901

# An epoch with a nonzero time of day, so a test that silently drops the fractional part fails.
EPOCH = DateTime('2021-06-15T00:00:00.000000')

WAVELENGTH = 0.05546576  # Sentinel-1 C-band
PRF = 486.4863103
DR = 2.329562114715323
STARTING_RANGE = 800000.0


def h(x):
    """A float as both a decimal and an exactly round-tripping hex literal."""
    return {'dec': float(x), 'hex': float(x).hex()}


def v3(a):
    return [h(x) for x in (a[0], a[1], a[2])]


def synth_orbit(n=25, spacing=10.0, t0=0.0):
    """A near-polar circular orbit: analytic, so the Julia side builds the identical state vectors.

    Circular rather than a real ephemeris because the state vectors must be reproducible on the
    Julia side to the bit from a short description. Hermite interpolation does not care whether the
    trajectory is physical.
    """
    R = 7.0e6
    mu = 3.986004418e14
    w = np.sqrt(mu / R**3)
    inc = np.deg2rad(98.0)
    times, positions, velocities = [], [], []
    for i in range(n):
        ti = t0 + i * spacing
        ang = w * ti
        p = np.array([R * np.cos(ang),
                      R * np.sin(ang) * np.cos(inc),
                      R * np.sin(ang) * np.sin(inc)])
        v = np.array([-R * w * np.sin(ang),
                      R * w * np.cos(ang) * np.cos(inc),
                      R * w * np.cos(ang) * np.sin(inc)])
        times.append(ti)
        positions.append(p)
        velocities.append(v)
    return {'R': R, 'mu': mu, 'w': w, 'inc': inc, 'n': n, 'spacing': spacing, 't0': t0,
            'times': times, 'positions': positions, 'velocities': velocities}


def build_orbit(spec):
    svs = [StateVector(EPOCH + isce3.core.TimeDelta(t), p, v)
           for t, p, v in zip(spec['times'], spec['positions'], spec['velocities'])]
    return Orbit(svs, EPOCH)


out = {}
el = Ellipsoid(a=A, e2=E2)

# ----------------------------------------------------------------------------- ellipsoid

# Points spanning both hemispheres, both signs of longitude, the poles and the equator, and heights
# from below the geoid to above any terrain. The poles are where Vermeille's form is most delicate:
# the lateral distance goes to zero and the latitude comes from `atan2(z, ~0)`.
ELL_CASES = [
    ('equator_prime',      0.0,    0.0,      0.0),
    ('mid_north',        -45.0,   40.0,    500.0),
    ('mid_south',        135.0,  -40.0,    500.0),
    ('greenland',        -45.0,   72.0,   2000.0),
    ('antarctica',       160.0,  -78.0,   3000.0),
    ('north_pole',         0.0,   90.0,      0.0),
    ('south_pole',         0.0,  -90.0,      0.0),
    ('near_north_pole',   17.0,   89.9,    100.0),
    ('dateline',         180.0,   10.0,      0.0),
    ('neg_height',        20.0,  -20.0,   -200.0),
    ('high_terrain',     -70.0,  -33.0,   6800.0),
    ('sat_altitude',      30.0,   60.0, 700000.0),
]

ell = []
for name, lon_d, lat_d, hgt in ELL_CASES:
    lon, lat = np.deg2rad(lon_d), np.deg2rad(lat_d)
    llh = np.array([lon, lat, hgt])
    xyz = np.array(el.lon_lat_to_xyz(llh).tolist())
    llh_back = np.array(el.xyz_to_lon_lat(xyz).tolist())
    ell.append({
        'name': name,
        'lon_deg': h(lon_d), 'lat_deg': h(lat_d),
        'llh': v3(llh),
        'xyz': v3(xyz),
        'llh_roundtrip': v3(llh_back),
    })
out['ellipsoid'] = {
    'a': h(A), 'e2': h(E2), 'b': h(el.b),
    'cases': ell,
}

# ----------------------------------------------------------------------------- TCN basis
#
# isce3 exposes the geocentric TCN construction only through C++ `Basis(p, v)`, so the fixture
# records the inputs and the arithmetic is asserted against the Julia implementation's own round
# trip plus orthonormality. `geodeticTCN` is a *different*, ellipsoid-normal basis and is not what
# either the kernel or rdr2geo uses.
spec = synth_orbit()
tcn = []
for ti in (0.0, 37.5, 120.0, 240.0):
    orb = build_orbit(spec)
    p, v = orb.interpolate(ti)
    tcn.append({'t': h(ti), 'position': v3(np.array(p).ravel()),
                'velocity': v3(np.array(v).ravel())})
out['tcn'] = {'cases': tcn}

# ----------------------------------------------------------------------------- orbit
#
# Times chosen to exercise every branch of the index selection: exactly on a node, between nodes,
# inside the first and last two spacings where the `[0, size-4]` clamp engages, and exactly at both
# domain edges.
orb = build_orbit(spec)
t_last = spec['t0'] + (spec['n'] - 1) * spec['spacing']
ORBIT_TIMES = [
    ('start_edge',      spec['t0']),
    ('first_interval',  spec['t0'] + 3.7),
    ('clamp_low',       spec['t0'] + spec['spacing'] * 0.5),
    ('node_1',          spec['t0'] + spec['spacing']),
    ('node_2',          spec['t0'] + spec['spacing'] * 2),
    ('mid_between',     spec['t0'] + spec['spacing'] * 11.5),
    ('mid_node',        spec['t0'] + spec['spacing'] * 12),
    ('mid_offset',      spec['t0'] + spec['spacing'] * 12 + 1e-6),
    ('clamp_high',      t_last - spec['spacing'] * 0.5),
    ('last_node_minus', t_last - spec['spacing']),
    ('stop_edge',       t_last),
]

orbit_cases = []
for name, ti in ORBIT_TIMES:
    p, v = orb.interpolate(ti)
    orbit_cases.append({
        'name': name,
        't': h(ti),
        'position': v3(np.array(p).ravel()),
        'velocity': v3(np.array(v).ravel()),
    })

out['orbit'] = {
    'spec': {
        'R': h(spec['R']), 'mu': h(spec['mu']), 'w': h(spec['w']), 'inc': h(spec['inc']),
        'n': spec['n'], 'spacing': h(spec['spacing']), 't0': h(spec['t0']),
    },
    'epoch': str(EPOCH),
    'state_vectors': [
        {'t': h(t), 'position': v3(p), 'velocity': v3(v)}
        for t, p, v in zip(spec['times'], spec['positions'], spec['velocities'])
    ],
    'cases': orbit_cases,
}

# ----------------------------------------------------------------------------- rdr2geo
#
# Both look sides, a spread of ranges across a plausible swath, and the heights the reference's own
# callers use — `zrange = [-200, 4000]` for the footprint (`GeogridRadar.py:140`) and the incidence
# angle (`:253`).
rdr_cases = []
for side_name, side in (('right', LookSide.Right), ('left', LookSide.Left)):
    for rng_off in (0.0, 12000.0, 60000.0):
        for zz in (-200.0, 0.0, 500.0, 4000.0):
            for ti in (60.0, 120.0):
                rng = STARTING_RANGE + rng_off
                llh = rdr2geo(
                    ti,
                    rng,
                    orb,
                    side,
                    0.0,            # doppler
                    WAVELENGTH,
                    DEMInterpolator(zz),
                    el,
                )
                llh = np.array(llh).ravel()
                rdr_cases.append({
                    'side': side_name,
                    't': h(ti),
                    'range': h(rng),
                    'height': h(zz),
                    'llh': v3(llh),
                })

out['rdr2geo'] = {
    'wavelength': h(WAVELENGTH),
    'doppler': h(0.0),
    'threshold': h(1e-8), 'maxiter': 25, 'extraiter': 15,
    'cases': rdr_cases,
}

# ----------------------------------------------------------------------------- radar params
out['radar'] = {
    'prf': h(PRF), 'dr': h(DR), 'starting_range': h(STARTING_RANGE),
    'wavelength': h(WAVELENGTH),
}

out['provenance'] = {
    'isce3_version': isce3.__version__,
    'numpy_version': np.__version__,
    'ellipsoid_a': h(A), 'ellipsoid_e2': h(E2),
}

with open(os.path.join(HERE, 'radar_numerics.json'), 'w') as fh:
    json.dump(out, fh, indent=1, sort_keys=True)

print(f'ellipsoid: {len(out["ellipsoid"]["cases"])} cases')
print(f'tcn:       {len(out["tcn"]["cases"])} cases')
print(f'orbit:     {len(out["orbit"]["cases"])} cases over {spec["n"]} state vectors')
print(f'rdr2geo:   {len(out["rdr2geo"]["cases"])} cases')
sz = os.path.getsize(os.path.join(HERE, 'radar_numerics.json'))
print(f'\nwrote radar_numerics.json ({sz / 1e3:.0f} kB), isce3 {isce3.__version__}')
