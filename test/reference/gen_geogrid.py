#!/usr/bin/env python
"""Generate whole-kernel fixtures by running the reference geogrid.

Calls the installed `GeogridOptical.runGeogrid()` — the compiled C++ kernel — on small synthetic
inputs and records the nine output rasters plus the scalars.

The optical path reads no image *pixel* values: it uses the image only for its geotransform and
CRS. So the inputs are a small DEM window and matching parameter rasters, and the fixture stays
small enough to commit and needs no network.

Run in the reference environment:

    micromamba run -n geogrid-ref python test/reference/gen_geogrid.py
"""
import json
import os
import tempfile

import numpy as np
from osgeo import gdal, osr
from geogrid import GeogridOptical

gdal.UseExceptions()
HERE = os.path.dirname(os.path.abspath(__file__))

NODATA_OUT = -32767


def write_raster(path, arr, gt, epsg, nodata=None, dtype=gdal.GDT_Float32):
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
    """Every band of a raster as a list of nested lists, or None if the file is absent."""
    if not os.path.exists(path):
        return None
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    out = {
        'size': [ds.RasterXSize, ds.RasterYSize],
        'geotransform': list(ds.GetGeoTransform()),
        'bands': [],
    }
    for b in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(b)
        a = band.ReadAsArray()
        out['bands'].append({
            'dtype': gdal.GetDataTypeName(band.DataType),
            'nodata': band.GetNoDataValue(),
                'shape': list(a.shape),
        })
        arrays[f'{tag}/{os.path.basename(path)}/band{b}'] = np.ascontiguousarray(
            a, dtype=np.float64)
    del ds
    return out


def synth(shape, kind, seed):
    """Deterministic input fields that are varied but plausible."""
    rng = np.random.default_rng(seed)
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
    if kind == 'dem':
        return 500.0 + 40.0 * np.sin(xx / 7.0) + 30.0 * np.cos(yy / 5.0) + rng.normal(0, 2, shape)
    if kind == 'dhdx':
        return 0.03 * np.cos(xx / 7.0) + rng.normal(0, 0.002, shape)
    if kind == 'dhdy':
        return -0.02 * np.sin(yy / 5.0) + rng.normal(0, 0.002, shape)
    if kind == 'vx':
        return 120.0 * np.sin(xx / 11.0) + 40.0 + rng.normal(0, 5, shape)
    if kind == 'vy':
        return -90.0 * np.cos(yy / 9.0) + rng.normal(0, 5, shape)
    if kind == 'srx':
        return 200.0 + 150.0 * np.abs(np.sin(xx / 6.0))
    if kind == 'sry':
        return 150.0 + 100.0 * np.abs(np.cos(yy / 8.0))
    if kind == 'csminx':
        return np.full(shape, 240.0)
    if kind == 'csminy':
        return np.full(shape, 240.0)
    if kind == 'csmaxx':
        return np.full(shape, 480.0)
    if kind == 'csmaxy':
        return np.full(shape, 480.0)
    if kind == 'ssm':
        return (((xx.astype(int) + yy.astype(int)) % 3) == 0).astype(np.float64)
    raise ValueError(kind)


# name, image epsg, dem epsg, image spacing, dem spacing, dem size, dt days, holes, full
CASES = [
    # Same CRS: the identity transform path.
    ('same_crs',        32624, 32624, 30.0, 120.0, 60, 91,  False, True),
    # Cross CRS: a real reprojection, three transforms per point.
    ('cross_crs',       32624,  3413, 30.0, 120.0, 60, 91,  False, True),
    ('cross_crs_3031',  32719,  3031, 30.0, 120.0, 48, 182, False, True),
    # Odd spacing ratio: every index lands on an exact .5, so the rounding mode shows.
    ('odd_ratio',       32624, 32624, 30.0,  90.0, 60, 91,  False, True),
    ('ratio_15m',       32624, 32624, 15.0, 120.0, 60, 32,  False, True),
    # Short interval: the search-range inflation is at its maximum.
    ('short_dt',        32624, 32624, 30.0, 120.0, 48, 5,   False, True),
    ('long_dt',         32624, 32624, 30.0, 120.0, 48, 365, False, True),
    # Nodata scattered through every input raster.
    ('with_nodata',     32624, 32624, 30.0, 120.0, 60, 91,  True,  True),
    ('cross_crs_nodata', 32624, 3413, 30.0, 120.0, 60, 91,  True,  True),
    # DEM only: most bands are unsupported and no file is written for them.
    ('dem_only',        32624, 32624, 30.0, 120.0, 48, 91,  False, False),
]

out = {'cases': []}
arrays = {}
tmp = tempfile.mkdtemp()

for (name, img_epsg, dem_epsg, img_sp, dem_sp, dem_n, dt_days, holes, full) in CASES:
    workdir = os.path.join(tmp, name)
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)

    # An image footprint, then a DEM grid derived to cover it in the DEM's CRS.
    if img_epsg == 32624:
        img_origin = (300000.0, 7800000.0)
    else:
        img_origin = (400000.0, 2600000.0)
    img_size = (1200, 1200)

    probe = GeogridOptical()
    probe.startingX, probe.startingY = img_origin
    probe.XSize, probe.YSize = img_sp, -img_sp
    probe.numberOfSamples, probe.numberOfLines = img_size
    probe.epsgDat, probe.epsgDem = img_epsg, dem_epsg
    probe.determineBbox()

    # Centre a dem_n x dem_n grid on the footprint so the window lands inside it.
    cx = 0.5 * (probe._xlim[0] + probe._xlim[1])
    cy = 0.5 * (probe._ylim[0] + probe._ylim[1])
    x0 = np.floor((cx - dem_n * dem_sp / 2) / dem_sp) * dem_sp
    y1 = np.ceil((cy + dem_n * dem_sp / 2) / dem_sp) * dem_sp
    dem_gt = (x0, dem_sp, 0.0, y1, 0.0, -dem_sp)
    shape = (dem_n, dem_n)

    fields = {}
    kinds = ['dem'] if not full else ['dem', 'dhdx', 'dhdy', 'vx', 'vy', 'srx', 'sry',
                                      'csminx', 'csminy', 'csmaxx', 'csmaxy', 'ssm']
    for si, kind in enumerate(kinds):
        arr = synth(shape, kind, seed=1000 + si)
        if holes and kind in ('vx', 'vy', 'srx', 'sry', 'csminx', 'csminy', 'csmaxx', 'csmaxy',
                              'ssm', 'dem'):
            # The DEM's sentinel, which the reference applies to every raster.
            arr = arr.copy()
            arr[(np.arange(shape[0])[:, None] + np.arange(shape[1])[None, :]) % 7 == 0] = -32767.0
        fields[kind] = arr
        arrays[f'{name}/input/{kind}'] = np.ascontiguousarray(arr, dtype=np.float64)
        path = os.path.join(workdir, kind + '.tif')
        write_raster(path, arr, dem_gt, dem_epsg,
                     nodata=-32767.0 if kind == 'dem' else None,
                     dtype=gdal.GDT_Float64)

    # A header-only image standing in for the reference scene.
    img_path = os.path.join(workdir, 'image.tif')
    write_raster(img_path, np.zeros((2, 2), dtype=np.float32),
                 (img_origin[0], img_sp, 0.0, img_origin[1], 0.0, -img_sp), img_epsg,
                 dtype=gdal.GDT_Byte)

    obj = GeogridOptical()
    obj.startingX, obj.startingY = img_origin
    obj.XSize, obj.YSize = img_sp, -img_sp
    obj.numberOfSamples, obj.numberOfLines = img_size
    obj.repeatTime = dt_days * 86400.0
    obj.nodata_out = NODATA_OUT
    obj.chipSizeX0 = 240.0
    obj.gridSpacingX = dem_sp
    obj.dat1name = img_path
    obj.demname = os.path.join(workdir, 'dem.tif')
    for kind, attr in (('dhdx', 'dhdxname'), ('dhdy', 'dhdyname'), ('vx', 'vxname'),
                       ('vy', 'vyname'), ('srx', 'srxname'), ('sry', 'sryname'),
                       ('csminx', 'csminxname'), ('csminy', 'csminyname'),
                       ('csmaxx', 'csmaxxname'), ('csmaxy', 'csmaxyname'), ('ssm', 'ssmname')):
        setattr(obj, attr, os.path.join(workdir, kind + '.tif') if kind in fields else '')
    obj.winlocname = 'window_location.tif'
    obj.winoffname = 'window_offset.tif'
    obj.winsrname = 'window_search_range.tif'
    obj.wincsminname = 'window_chip_size_min.tif'
    obj.wincsmaxname = 'window_chip_size_max.tif'
    obj.winssmname = 'window_stable_surface_mask.tif'
    obj.winro2vxname = 'window_rdr_off2vel_x_vec.tif'
    obj.winro2vyname = 'window_rdr_off2vel_y_vec.tif'
    obj.winsfname = 'window_scale_factor.tif'

    obj.runGeogrid()

    rec = {
        'name': name,
        'image': {'origin': list(img_origin), 'spacing': [img_sp, -img_sp],
                  'size': list(img_size), 'epsg': img_epsg},
        'dem': {'geotransform': list(dem_gt), 'size': list(shape), 'epsg': dem_epsg},
        'dt': dt_days * 86400.0,
        'chip_size_0': 240.0,
        'has_full_inputs': full,
        'input_names': sorted(fields),
        'scalars': {
            'pOff': int(obj.pOff), 'lOff': int(obj.lOff),
            'pCount': int(obj.pCount), 'lCount': int(obj.lCount),
            'X_res': float(obj.X_res), 'Y_res': float(obj.Y_res),
            'X_res_hex': float(obj.X_res).hex(), 'Y_res_hex': float(obj.Y_res).hex(),
            'cen_lat': float(obj.cen_lat), 'cen_lon': float(obj.cen_lon),
        },
        'outputs': {fn: read_bands(os.path.join(workdir, fn), name, arrays) for fn, in
                    [(f,) for f in ('window_location.tif', 'window_offset.tif',
                                    'window_search_range.tif', 'window_chip_size_min.tif',
                                    'window_chip_size_max.tif',
                                    'window_stable_surface_mask.tif',
                                    'window_rdr_off2vel_x_vec.tif',
                                    'window_rdr_off2vel_y_vec.tif',
                                    'window_scale_factor.tif')]},
    }
    out['cases'].append(rec)
    present = [f for f, v in rec['outputs'].items() if v is not None]
    print(f'{name}: window {obj.pCount}x{obj.lCount} at ({obj.pOff},{obj.lOff}), '
          f'{len(present)}/9 outputs written')

os.chdir(HERE)
import geogrid
out['provenance'] = {
    'autorift_version': geogrid.__version__,
    'gdal_version': gdal.__version__,
    'proj_version': f'{osr.GetPROJVersionMajor()}.{osr.GetPROJVersionMinor()}.{osr.GetPROJVersionMicro()}',
    'nodata_out': NODATA_OUT,
}
with open(os.path.join(HERE, 'geogrid.json'), 'w') as fh:
    json.dump(out, fh, indent=1, sort_keys=True)
# Arrays as raw little-endian Float64 in a compressed archive: exact to the bit, and far smaller
# than the same values as JSON text.
np.savez_compressed(os.path.join(HERE, 'geogrid_arrays.npz'), **arrays)
jsz = os.path.getsize(os.path.join(HERE, 'geogrid.json'))
asz = os.path.getsize(os.path.join(HERE, 'geogrid_arrays.npz'))
print(f'\nwrote geogrid.json ({jsz / 1e3:.0f} kB) and geogrid_arrays.npz ({asz / 1e6:.2f} MB), '
      f'{len(out["cases"])} cases, {len(arrays)} arrays')
