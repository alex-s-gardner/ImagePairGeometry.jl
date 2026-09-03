#!/usr/bin/env python
"""Generate the whole-kernel radar fixture by running the compiled reference.

Calls `GeogridRadar.geogridRadar()` — the C++ extension — on synthetic inputs and records its nine
output rasters plus the scalars, the same shape as `gen_geogrid.py` does for the optical path.

No real SLC is needed. `geogridRadar` takes every radar parameter as a scalar and reads the orbit
from a file it parses itself, with plain string matching on `<UTC>` and `<X unit>` tags
(`geogridRadar.cpp:333-427`). So a synthetic circular orbit written in that format exercises the
kernel end to end, and the fixture stays small enough to commit with no network and no large data.

Run in the reference environment:

    micromamba run -n geogrid-ref python test/reference/gen_radar_geogrid.py
"""
import datetime
import json
import os
import tempfile

import numpy as np
import isce3
from isce3.core import DateTime, LookSide, Orbit, StateVector
from osgeo import gdal, osr

from geogrid import GeogridRadar

gdal.UseExceptions()
HERE = os.path.dirname(os.path.abspath(__file__))

NODATA_OUT = -32767

# The reference's ellipsoid constants (`geogridRadar.cpp:324-325`).
A = 6378137.0
E2 = 0.0066943799901

# A Sentinel-1-like acquisition.
PRF = 486.4863103
DR = 2.329562114715323
STARTING_RANGE = 800000.0
WAVELENGTH = 0.05546576

# Orbit epoch, and the acquisition start within that day.
EPOCH_DAY = datetime.datetime(2021, 6, 15)
ORBIT_N = 61
ORBIT_SPACING = 10.0


def h(x):
    return {'dec': float(x), 'hex': float(x).hex()}


def synth_orbit(n=ORBIT_N, spacing=ORBIT_SPACING):
    """A near-polar circular orbit, analytic so the Julia side rebuilds it to the bit."""
    R = 7.0e6
    mu = 3.986004418e14
    w = np.sqrt(mu / R**3)
    inc = np.deg2rad(98.0)
    times, positions, velocities = [], [], []
    for i in range(n):
        ti = i * spacing
        a = w * ti
        positions.append(np.array([R * np.cos(a),
                                   R * np.sin(a) * np.cos(inc),
                                   R * np.sin(a) * np.sin(inc)]))
        velocities.append(np.array([-R * w * np.sin(a),
                                    R * w * np.cos(a) * np.cos(inc),
                                    R * w * np.cos(a) * np.sin(inc)]))
        times.append(ti)
    return {'R': R, 'mu': mu, 'w': w, 'inc': inc, 'n': n, 'spacing': spacing,
            'times': times, 'positions': positions, 'velocities': velocities}


def write_orbit_eof(path, spec):
    """The orbit in the tag format `geogridRadar.cpp:333-427` scrapes.

    It reads only `<UTC>UTC=...`, `<X unit=...>`, `<Y ...>`, `<Z ...>`, `<VX ...>`, `<VY ...>`,
    `<VZ ...>`, one state vector per group, and keeps those inside `[itime, ftime]`. Nothing else in
    the file is parsed, so this is the minimal document that satisfies it.
    """
    with open(path, 'w') as fh:
        fh.write('<?xml version="1.0"?>\n<Earth_Explorer_File>\n<Data_Block>\n<List_of_OSVs>\n')
        for t, p, v in zip(spec['times'], spec['positions'], spec['velocities']):
            stamp = (EPOCH_DAY + datetime.timedelta(seconds=t)).strftime('%Y-%m-%dT%H:%M:%S.%f')
            fh.write('<OSV>\n')
            fh.write(f'<UTC>UTC={stamp}</UTC>\n')
            fh.write(f'<X unit="m">{p[0]!r}</X>\n')
            fh.write(f'<Y unit="m">{p[1]!r}</Y>\n')
            fh.write(f'<Z unit="m">{p[2]!r}</Z>\n')
            fh.write(f'<VX unit="m/s">{v[0]!r}</VX>\n')
            fh.write(f'<VY unit="m/s">{v[1]!r}</VY>\n')
            fh.write(f'<VZ unit="m/s">{v[2]!r}</VZ>\n')
            fh.write('</OSV>\n')
        fh.write('</List_of_OSVs>\n</Data_Block>\n</Earth_Explorer_File>\n')


def build_orbit(spec):
    epoch = DateTime(EPOCH_DAY.strftime('%Y-%m-%dT%H:%M:%S.%f'))
    svs = [StateVector(epoch + isce3.core.TimeDelta(t), p, v)
           for t, p, v in zip(spec['times'], spec['positions'], spec['velocities'])]
    return Orbit(svs, epoch)


def write_raster(path, arr, gt, epsg, nodata=None, dtype=gdal.GDT_Float64):
    ny, nx = arr.shape
    ds = gdal.GetDriverByName('GTiff').Create(path, nx, ny, 1, dtype)
    ds.SetGeoTransform(list(gt))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    if nodata is not None:
        band.SetNoDataValue(float(nodata))
    band.WriteArray(arr)
    ds.FlushCache()
    del ds


def read_bands(path, tag, arrays):
    """Every band of a raster, or None if the reference wrote no file for it."""
    if not os.path.exists(path):
        return None
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    out = {'size': [ds.RasterXSize, ds.RasterYSize],
           'geotransform': list(ds.GetGeoTransform()), 'bands': []}
    for b in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(b)
        a = band.ReadAsArray()
        out['bands'].append({'dtype': gdal.GetDataTypeName(band.DataType),
                             'nodata': band.GetNoDataValue(), 'shape': list(a.shape)})
        arrays[f'{tag}/{os.path.basename(path)}/band{b}'] = np.ascontiguousarray(a, dtype=np.float64)
    del ds
    return out


def synth(shape, kind, seed, holes=False):
    """Deterministic input fields, varied but plausible."""
    rng = np.random.default_rng(seed)
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
    if kind == 'dem':
        arr = 500.0 + 40.0 * np.sin(xx / 7.0) + 30.0 * np.cos(yy / 5.0) + rng.normal(0, 2, shape)
    elif kind == 'dhdx':
        arr = 0.03 * np.cos(xx / 7.0) + rng.normal(0, 0.002, shape)
    elif kind == 'dhdy':
        arr = -0.02 * np.sin(yy / 5.0) + rng.normal(0, 0.002, shape)
    elif kind == 'vx':
        arr = 120.0 * np.sin(xx / 11.0) + 40.0 + rng.normal(0, 5, shape)
    elif kind == 'vy':
        arr = -90.0 * np.cos(yy / 9.0) + rng.normal(0, 5, shape)
    elif kind == 'srx':
        arr = 200.0 + 150.0 * np.abs(np.sin(xx / 6.0))
    elif kind == 'sry':
        arr = 150.0 + 100.0 * np.abs(np.cos(yy / 8.0))
    elif kind in ('csminx', 'csminy'):
        arr = np.full(shape, 240.0)
    elif kind in ('csmaxx', 'csmaxy'):
        arr = np.full(shape, 480.0)
    elif kind == 'ssm':
        arr = (((xx.astype(int) + yy.astype(int)) % 3) == 0).astype(np.float64)
    else:
        raise ValueError(kind)
    if holes and kind != 'dhdx' and kind != 'dhdy':
        arr = arr.copy()
        arr[(np.arange(shape[0])[:, None] + np.arange(shape[1])[None, :]) % 7 == 0] = -32767.0
    return arr


# name, grid epsg, grid spacing m, grid size, look side, dt days, holes, full inputs
CASES = [
    # A projected grid over the swath: the production shape.
    ('utm32n',        32632, 500.0, 48, 'right',  6,   False, True),
    # A coarser grid, so a different set of points lands near rounding boundaries.
    ('utm32n_coarse', 32632, 900.0, 32, 'right',  6,   False, True),
    # Left-looking: the swath lands on the other side of the track.
    ('utm32n_left',   32632, 500.0, 48, 'left',   6,   False, True),
    # A long interval, so the search-range inflation is at its floor rather than its ceiling.
    ('long_dt',       32632, 500.0, 32, 'right', 365,  False, True),
    # Nodata scattered through every input raster.
    ('with_nodata',   32632, 500.0, 48, 'right',  6,   True,  True),
    # DEM only: most bands are unsupported and the reference writes no file for them.
    ('dem_only',      32632, 500.0, 32, 'right',  6,   False, False),
    # DEM and slope, no velocity or search range: the operator and scale factors are written but the
    # offset and search bands are not.
    ('dem_slope',     32632, 500.0, 32, 'right',  6,   False, 'slope'),
    # A grid deliberately larger than the swath, so points fall outside it and the `rgind`/`azind`
    # bounds test at `geogridRadar.cpp:1112` writes sentinels for geometric rather than nodata reasons.
    ('oversize',      32632, 500.0, 200, 'right', 6,   False, True),
]

out = {'cases': []}
arrays = {}
tmp = tempfile.mkdtemp()
spec = synth_orbit()
orbit_path = os.path.join(tmp, 'orbit.EOF')
write_orbit_eof(orbit_path, spec)

for (name, epsg, spacing, npix, side, dt_days, holes, full) in CASES:
    workdir = os.path.join(tmp, name)
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)

    nsamples, nlines = 10000, 8000
    # Seconds since midnight, which is the scale `setAzimuthParameters` receives
    # (`GeogridRadar.py:328-330`).
    sensing_start = 300.0
    sensing_dt = EPOCH_DAY + datetime.timedelta(seconds=sensing_start)

    obj = GeogridRadar()
    obj.startingRange = STARTING_RANGE
    obj.rangePixelSize = DR
    obj.sensingStart = sensing_dt
    obj.sensingStop = sensing_dt + datetime.timedelta(seconds=(nlines - 1) / PRF)
    obj.orbitname = orbit_path
    obj.prf = PRF
    obj.aztime = sensing_start
    obj.wavelength = WAVELENGTH
    # The same attribute reaches the Python `rdr2geo` (which wants a `LookSide`) and
    # `setLookSide_Py` (which wants the int the C++ struct holds), so it must be the enum: it casts to
    # -1 for right and +1 for left, matching `geogridRadar.cpp:1048`.
    obj.lookSide = LookSide.Right if side == 'right' else LookSide.Left
    obj.repeatTime = dt_days * 86400.0
    obj.numberOfLines = nlines
    obj.numberOfSamples = nsamples
    obj.nodata_out = NODATA_OUT
    obj.chipSizeX0 = 240.0
    obj.orbit = build_orbit(spec)
    obj.epsg = epsg

    # The footprint, so the DEM can be centred on it.
    obj.getIncidenceAngle()
    obj.determineBbox()
    cx = 0.5 * (obj._xlim[0] + obj._xlim[1])
    cy = 0.5 * (obj._ylim[0] + obj._ylim[1])
    x0 = np.floor((cx - npix * spacing / 2) / spacing) * spacing
    y1 = np.ceil((cy + npix * spacing / 2) / spacing) * spacing
    dem_gt = (x0, spacing, 0.0, y1, 0.0, -spacing)
    shape = (npix, npix)
    obj.gridSpacingX = spacing

    if full == 'slope':
        kinds = ['dem', 'dhdx', 'dhdy']
    elif full:
        kinds = ['dem', 'dhdx', 'dhdy', 'vx', 'vy', 'srx', 'sry',
                 'csminx', 'csminy', 'csmaxx', 'csmaxy', 'ssm']
    else:
        kinds = ['dem']
    fields = {}
    for si, kind in enumerate(kinds):
        arr = synth(shape, kind, seed=2000 + si, holes=holes)
        fields[kind] = arr
        arrays[f'{name}/input/{kind}'] = np.ascontiguousarray(arr, dtype=np.float64)
        write_raster(os.path.join(workdir, kind + '.tif'), arr, dem_gt, epsg,
                     nodata=-32767.0 if kind == 'dem' else None)

    obj.demname = os.path.join(workdir, 'dem.tif')
    for kind, attr in (('dhdx', 'dhdxname'), ('dhdy', 'dhdyname'), ('vx', 'vxname'),
                       ('vy', 'vyname'), ('srx', 'srxname'), ('sry', 'sryname'),
                       ('csminx', 'csminxname'), ('csminy', 'csminyname'),
                       ('csmaxx', 'csmaxxname'), ('csmaxy', 'csmaxyname'), ('ssm', 'ssmname')):
        setattr(obj, attr, os.path.join(workdir, kind + '.tif') if kind in fields else None)

    obj.winlocname = 'window_location.tif'
    obj.winoffname = 'window_offset.tif'
    obj.winsrname = 'window_search_range.tif'
    obj.wincsminname = 'window_chip_size_min.tif'
    obj.wincsmaxname = 'window_chip_size_max.tif'
    obj.winssmname = 'window_stable_surface_mask.tif'
    obj.winro2vxname = 'window_rdr_off2vel_x_vec.tif'
    obj.winro2vyname = 'window_rdr_off2vel_y_vec.tif'
    obj.winsfname = 'window_scale_factor.tif'

    obj.geogridRadar()

    # `geogridRadar` ends with `GDALDestroyDriverManager()` (`geogridRadar.cpp:1355`), which tears
    # down GDAL for the whole process — so reading its own output, or running a second case, needs the
    # drivers registered again.
    gdal.AllRegister()

    rec = {
        'name': name,
        'radar': {'starting_range': h(STARTING_RANGE), 'dr': h(DR), 'prf': h(PRF),
                  'wavelength': h(WAVELENGTH), 'nsamples': nsamples, 'nlines': nlines,
                  'sensing_start': h(sensing_start), 'look_side': side,
                  'incidence_angle': h(obj.incidenceAngle)},
        'grid': {'geotransform': list(dem_gt), 'size': list(shape), 'epsg': epsg},
        'dt': dt_days * 86400.0,
        'chip_size_0': 240.0,
        'has_full_inputs': full is True,
        'input_names': sorted(fields),
        'scalars': {'pOff': int(obj.pOff), 'lOff': int(obj.lOff),
                    'pCount': int(obj.pCount), 'lCount': int(obj.lCount),
                    'X_res': h(obj.X_res), 'Y_res': h(obj.Y_res)},
        'bbox': {'xlim': [h(obj._xlim[0]), h(obj._xlim[1])],
                 'ylim': [h(obj._ylim[0]), h(obj._ylim[1])]},
        'outputs': {fn: read_bands(os.path.join(workdir, fn), name, arrays)
                    for fn in ('window_location.tif', 'window_offset.tif',
                               'window_search_range.tif', 'window_chip_size_min.tif',
                               'window_chip_size_max.tif', 'window_stable_surface_mask.tif',
                               'window_rdr_off2vel_x_vec.tif', 'window_rdr_off2vel_y_vec.tif',
                               'window_scale_factor.tif')},
    }
    out['cases'].append(rec)
    present = [f for f, v in rec['outputs'].items() if v is not None]
    print(f'{name}: window {obj.pCount}x{obj.lCount} at ({obj.pOff},{obj.lOff}), '
          f'{len(present)}/9 outputs')

os.chdir(HERE)
import geogrid
out['orbit'] = {
    'epoch': EPOCH_DAY.strftime('%Y-%m-%dT%H:%M:%S.%f'),
    'R': h(spec['R']), 'w': h(spec['w']), 'inc': h(spec['inc']),
    'n': spec['n'], 'spacing': h(spec['spacing']),
}
out['provenance'] = {
    'autorift_version': geogrid.__version__,
    'isce3_version': isce3.__version__,
    'gdal_version': gdal.__version__,
    'proj_version': f'{osr.GetPROJVersionMajor()}.{osr.GetPROJVersionMinor()}.'
                    f'{osr.GetPROJVersionMicro()}',
    'nodata_out': NODATA_OUT,
    'ellipsoid_a': h(A), 'ellipsoid_e2': h(E2),
}
with open(os.path.join(HERE, 'radar_geogrid.json'), 'w') as fh:
    json.dump(out, fh, indent=1, sort_keys=True)
np.savez_compressed(os.path.join(HERE, 'radar_geogrid_arrays.npz'), **arrays)
jsz = os.path.getsize(os.path.join(HERE, 'radar_geogrid.json'))
asz = os.path.getsize(os.path.join(HERE, 'radar_geogrid_arrays.npz'))
print(f'\nwrote radar_geogrid.json ({jsz / 1e3:.0f} kB) and '
      f'radar_geogrid_arrays.npz ({asz / 1e6:.2f} MB), {len(out["cases"])} cases')
