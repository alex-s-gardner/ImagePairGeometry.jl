#!/usr/bin/env python
"""Generate coregister fixtures from the reference implementation.

Calls the real `GeogridOptical.coregister` from the installed `geogrid` package on synthetic
header-only GeoTIFFs. The optical path never reads image pixel values, so a 1x1 raster carrying
the right geotransform and CRS is a faithful stand-in for a full scene and keeps the fixture
small enough to commit.

Run in the reference environment:

    micromamba run -n geogrid-ref python test/reference/gen_coregister.py

Writes `coregister.json` beside this file.
"""
import json
import os
import tempfile

from osgeo import gdal, osr
from geogrid import GeogridOptical

gdal.UseExceptions()

HERE = os.path.dirname(os.path.abspath(__file__))


def make_tif(path, origin, spacing, size, epsg):
    """A GeoTIFF with the given geotransform and CRS. One pixel of data; only headers matter."""
    ds = gdal.GetDriverByName('GTiff').Create(path, size[0], size[1], 1, gdal.GDT_Byte)
    ds.SetGeoTransform([origin[0], spacing[0], 0.0, origin[1], 0.0, spacing[1]])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    ds.SetProjection(srs.ExportToWkt())
    ds.FlushCache()
    del ds


# (name, ref_origin, sec_origin, spacing, ref_size, sec_size, epsg)
# Chosen to exercise: exact alignment, offsets in each axis and both, differing sizes, a
# one-pixel overlap, negative and positive coordinates, several spacings including the
# 15 m and 10 m cases where the DEM/image ratio is odd, and non-square pixels.
CASES = [
    ('identical',        (300000.0, 7800000.0), (300000.0, 7800000.0), (30.0, -30.0), (200, 200), (200, 200), 32624),
    ('offset_x',         (300000.0, 7800000.0), (300600.0, 7800000.0), (30.0, -30.0), (200, 200), (200, 200), 32624),
    ('offset_y',         (300000.0, 7800000.0), (300000.0, 7799400.0), (30.0, -30.0), (200, 200), (200, 200), 32624),
    ('offset_both',      (300000.0, 7800000.0), (300600.0, 7799400.0), (30.0, -30.0), (200, 200), (200, 200), 32624),
    ('negative_offset',  (300600.0, 7799400.0), (300000.0, 7800000.0), (30.0, -30.0), (200, 200), (200, 200), 32624),
    ('differing_size',   (300000.0, 7800000.0), (300300.0, 7799700.0), (30.0, -30.0), (300, 250), (200, 400), 32624),
    ('one_pixel',        (300000.0, 7800000.0), (305970.0, 7794030.0), (30.0, -30.0), (200, 200), (200, 200), 32624),
    ('spacing_15',       (300000.0, 7800000.0), (300015.0, 7799985.0), (15.0, -15.0), (300, 300), (300, 300), 32624),
    ('spacing_10',       (300000.0, 7800000.0), (300010.0, 7799990.0), (10.0, -10.0), (300, 300), (300, 300), 32624),
    ('spacing_60',       (300000.0, 7800000.0), (300060.0, 7799940.0), (60.0, -60.0), (100, 100), (100, 100), 32624),
    ('nonsquare_pixel',  (300000.0, 7800000.0), (300060.0, 7799970.0), (20.0, -30.0), (150, 150), (150, 150), 32624),
    ('polar_3413',       (-200000.0, -2200000.0), (-199880.0, -2200120.0), (120.0, -120.0), (200, 200), (200, 200), 3413),
    ('polar_3031',       (-1500000.0, 1500000.0), (-1499880.0, 1499880.0), (120.0, -120.0), (200, 200), (200, 200), 3031),
    ('negative_coords',  (-500000.0, -100000.0), (-499970.0, -100030.0), (30.0, -30.0), (200, 200), (200, 200), 32624),
    ('large_offset',     (300000.0, 7800000.0), (302970.0, 7797030.0), (30.0, -30.0), (200, 200), (200, 200), 32624),
]

# Cases the reference must reject. Recorded so our error paths are gated too, not just the
# successful ones.
FAIL_CASES = [
    ('no_overlap',    (300000.0, 7800000.0), (400000.0, 7700000.0), (30.0, -30.0), (200, 200), (200, 200), 32624),
    ('half_pixel',    (300000.0, 7800000.0), (300015.0, 7799985.0), (30.0, -30.0), (200, 200), (200, 200), 32624),
]

out = {'cases': [], 'fail_cases': []}
tmp = tempfile.mkdtemp()

for name, o1, o2, sp, s1, s2, epsg in CASES:
    f1 = os.path.join(tmp, name + '_1.tif')
    f2 = os.path.join(tmp, name + '_2.tif')
    make_tif(f1, o1, sp, s1, epsg)
    make_tif(f2, o2, sp, s2, epsg)
    obj = GeogridOptical()
    x1a, y1a, xsize1, ysize1, x2a, y2a, xsize2, ysize2, trans = obj.coregister(f1, f2)
    out['cases'].append({
        'name': name,
        'reference': {'origin': list(o1), 'spacing': list(sp), 'size': list(s1)},
        'secondary': {'origin': list(o2), 'spacing': list(sp), 'size': list(s2)},
        'epsg': epsg,
        'expect': {
            'x1a': int(x1a), 'y1a': int(y1a), 'xsize1': int(xsize1), 'ysize1': int(ysize1),
            'x2a': int(x2a), 'y2a': int(y2a), 'xsize2': int(xsize2), 'ysize2': int(ysize2),
            # Hex so the comparison is against the exact double, not a reparsed decimal.
            'trans': [float(t) for t in trans],
            'trans_hex': [float(t).hex() for t in trans],
        },
    })
    print(f'{name}: off1=({x1a},{y1a}) off2=({x2a},{y2a}) size={xsize1}x{ysize1} '
          f'origin=({trans[0]},{trans[3]})')

for name, o1, o2, sp, s1, s2, epsg in FAIL_CASES:
    f1 = os.path.join(tmp, name + '_1.tif')
    f2 = os.path.join(tmp, name + '_2.tif')
    make_tif(f1, o1, sp, s1, epsg)
    make_tif(f2, o2, sp, s2, epsg)
    obj = GeogridOptical()
    try:
        obj.coregister(f1, f2)
        raised = None
        print(f'{name}: NO ERROR (reference accepted it)')
    except Exception as e:
        raised = str(e)
        print(f'{name}: raised {raised[:70]}')
    out['fail_cases'].append({
        'name': name,
        'reference': {'origin': list(o1), 'spacing': list(sp), 'size': list(s1)},
        'secondary': {'origin': list(o2), 'spacing': list(sp), 'size': list(s2)},
        'epsg': epsg,
        'raised': raised,
    })

import geogrid
from osgeo import gdal as _g
out['provenance'] = {
    'autorift_version': geogrid.__version__,
    'gdal_version': _g.__version__,
    'proj_version': f'{osr.GetPROJVersionMajor()}.{osr.GetPROJVersionMinor()}.{osr.GetPROJVersionMicro()}',
}

with open(os.path.join(HERE, 'coregister.json'), 'w') as fh:
    json.dump(out, fh, indent=1, sort_keys=True)
print('\nwrote coregister.json:', len(out['cases']), 'cases,', len(out['fail_cases']), 'fail cases')
print('provenance:', out['provenance'])
