#!/usr/bin/env python
"""Generate footprint-bounds and grid-window fixtures from the reference implementation.

Two things are captured, because they are computed in different places:

`determineBbox` is Python (`GeogridOptical.py:115-155`) and is called directly here. It sets
`_xlim`/`_ylim`, the bounding box of the image footprint in DEM coordinates.

The window (`pOff`/`lOff`/`pCount`/`lCount`) is computed in C++ from that box
(`geogridOptical.cpp:239-244`). Rather than run the whole kernel, the same arithmetic is
replayed here in Python with the same operation order, and the *whole-kernel* comparison in a
later step gates it end to end.

Run in the reference environment:

    micromamba run -n geogrid-ref python test/reference/gen_window.py
"""
import json
import math
import os
import tempfile

from osgeo import gdal, osr
from geogrid import GeogridOptical

gdal.UseExceptions()
HERE = os.path.dirname(os.path.abspath(__file__))


def make_tif(path, origin, spacing, size, epsg):
    ds = gdal.GetDriverByName('GTiff').Create(path, size[0], size[1], 1, gdal.GDT_Byte)
    ds.SetGeoTransform([origin[0], spacing[0], 0.0, origin[1], 0.0, spacing[1]])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    ds.SetProjection(srs.ExportToWkt())
    ds.FlushCache()
    del ds


def cpp_window(geo_trans, dem_size, xlim, ylim):
    """Replay geogridOptical.cpp:239-244 with its operation order and int truncation."""
    xmin, xmax = xlim
    ymin, ymax = ylim
    demXSize, demYSize = dem_size
    lOff = int(max(math.floor((ymax - geo_trans[3]) / geo_trans[5]), 0.0))
    lCount = int(min(math.ceil((ymin - geo_trans[3]) / geo_trans[5]), demYSize - 1.0)) - lOff
    pOff = int(max(math.floor((xmin - geo_trans[0]) / geo_trans[1]), 0.0))
    pCount = int(min(math.ceil((xmax - geo_trans[0]) / geo_trans[1]), demXSize - 1.0)) - pOff
    return pOff, lOff, pCount, lCount


# Image coordinate system (the coregistered overlap), DEM grid, and the CRS of each.
# Exercises: same-CRS (window arithmetic only), and the production cross-CRS cases where a polar
# stereographic DEM is paired with UTM imagery.
# (name, img_origin, img_spacing, img_size, img_epsg, dem_gt, dem_size, dem_epsg)
CASES = [
    ('same_crs_utm',
     (300000.0, 7800000.0), (30.0, -30.0), (2000, 2000), 32624,
     (295000.0, 7805000.0, 120.0, -120.0), (200, 200), 32624),
    ('same_crs_offset_grid',
     (300037.0, 7799963.0), (30.0, -30.0), (1500, 1500), 32624,
     (295000.0, 7805000.0, 120.0, -120.0), (200, 200), 32624),
    ('same_crs_spacing15',
     (300000.0, 7800000.0), (15.0, -15.0), (3000, 3000), 32624,
     (295000.0, 7805000.0, 120.0, -120.0), (200, 200), 32624),
    ('same_crs_grid90',
     (300000.0, 7800000.0), (30.0, -30.0), (2000, 2000), 32624,
     (295000.0, 7805000.0, 90.0, -90.0), (300, 300), 32624),
    # Image larger than the DEM: window clamps to the grid, and the demXSize-1 clamp shows.
    ('img_exceeds_dem',
     (200000.0, 7900000.0), (30.0, -30.0), (8000, 8000), 32624,
     (295000.0, 7805000.0, 120.0, -120.0), (100, 100), 32624),
    # Image entirely inside one DEM cell.
    ('tiny_image',
     (300000.0, 7800000.0), (30.0, -30.0), (3, 3), 32624,
     (295000.0, 7805000.0, 120.0, -120.0), (200, 200), 32624),
]

# Production shape: a polar stereographic DEM paired with UTM imagery, so the footprint is a
# rotated quadrilateral in grid coordinates and every corner transform matters.
#
# `determineBbox` reads no DEM file — only `epsgDat` and `epsgDem` — so the grid covering the
# footprint is derived from the bounds themselves, snapped outward to a whole number of grid
# cells. Hand-picking a grid origin instead risks a window that misses the footprint entirely,
# which tests the error path rather than the arithmetic.
# (name, img_origin, img_spacing, img_size, img_epsg, grid_spacing, margin_cells, dem_epsg)
DERIVED_CASES = [
    ('cross_crs_3413_utm24',
     (300000.0, 7800000.0), (30.0, -30.0), (2000, 2000), 32624, 120.0, 40, 3413),
    ('cross_crs_3031_utm19s',
     (400000.0, 2600000.0), (30.0, -30.0), (2000, 2000), 32719, 120.0, 40, 3031),
    ('cross_crs_utm_to_3413_grid240',
     (500000.0, 8100000.0), (30.0, -30.0), (1000, 1000), 32621, 240.0, 25, 3413),
    # Grid finer than the image pixel: window is larger than the image in points.
    ('cross_crs_fine_grid',
     (300000.0, 7800000.0), (30.0, -30.0), (500, 500), 32624, 10.0, 30, 3413),
    # Southern-hemisphere UTM against a polar grid, and an odd grid spacing.
    ('cross_crs_odd_spacing',
     (400000.0, 2600000.0), (30.0, -30.0), (800, 800), 32719, 90.0, 33, 3031),
]


def derive_grid(xlim, ylim, spacing, margin_cells):
    """A north-up grid covering `xlim`/`ylim` with `margin_cells` of slack on every side."""
    x0 = math.floor(xlim[0] / spacing - margin_cells) * spacing
    y1 = math.ceil(ylim[1] / spacing + margin_cells) * spacing
    nx = int(math.ceil((xlim[1] - x0) / spacing)) + margin_cells
    ny = int(math.ceil((y1 - ylim[0]) / spacing)) + margin_cells
    return (x0, y1, spacing, -spacing), (nx, ny)

# Where the reference produces a negative count and hands it to GDAL as a raster size.
FAIL_CASES = [
    ('no_grid_overlap',
     (300000.0, 7800000.0), (30.0, -30.0), (200, 200), 32624,
     (900000.0, 6000000.0, 120.0, -120.0), (100, 100), 32624),
]

out = {'cases': [], 'fail_cases': []}
tmp = tempfile.mkdtemp()


def run(name, img_origin, img_spacing, img_size, img_epsg, dem_gt, dem_size, dem_epsg):
    dem_path = os.path.join(tmp, name + '_dem.tif')
    make_tif(dem_path, (dem_gt[0], dem_gt[1]), (dem_gt[2], dem_gt[3]), dem_size, dem_epsg)

    obj = GeogridOptical()
    obj.startingX, obj.startingY = img_origin
    obj.XSize, obj.YSize = img_spacing
    obj.numberOfSamples, obj.numberOfLines = img_size
    obj.epsgDat = img_epsg
    obj.epsgDem = dem_epsg
    obj.determineBbox()

    geo_trans = [dem_gt[0], dem_gt[2], 0.0, dem_gt[1], 0.0, dem_gt[3]]
    pOff, lOff, pCount, lCount = cpp_window(geo_trans, dem_size, obj._xlim, obj._ylim)
    return obj, pOff, lOff, pCount, lCount


for case in CASES:
    name = case[0]
    obj, pOff, lOff, pCount, lCount = run(*case)
    _, img_origin, img_spacing, img_size, img_epsg, dem_gt, dem_size, dem_epsg = case
    out['cases'].append({
        'name': name,
        'image': {'origin': list(img_origin), 'spacing': list(img_spacing),
                  'size': list(img_size), 'epsg': img_epsg},
        'dem': {'geotransform': [dem_gt[0], dem_gt[2], 0.0, dem_gt[1], 0.0, dem_gt[3]],
                'size': list(dem_size), 'epsg': dem_epsg},
        'expect': {
            'xlim': [float(v) for v in obj._xlim],
            'ylim': [float(v) for v in obj._ylim],
            'xlim_hex': [float(v).hex() for v in obj._xlim],
            'ylim_hex': [float(v).hex() for v in obj._ylim],
            'pOff': pOff, 'lOff': lOff, 'pCount': pCount, 'lCount': lCount,
        },
    })
    print(f'{name}: xlim={obj._xlim} ylim={obj._ylim}')
    print(f'    window: pOff={pOff} lOff={lOff} pCount={pCount} lCount={lCount}')

for name, img_origin, img_spacing, img_size, img_epsg, spacing, margin, dem_epsg in DERIVED_CASES:
    # First pass with a placeholder grid to get the bounds, then a grid derived to cover them.
    probe = GeogridOptical()
    probe.startingX, probe.startingY = img_origin
    probe.XSize, probe.YSize = img_spacing
    probe.numberOfSamples, probe.numberOfLines = img_size
    probe.epsgDat, probe.epsgDem = img_epsg, dem_epsg
    probe.determineBbox()

    dem_gt, dem_size = derive_grid(probe._xlim, probe._ylim, spacing, margin)
    obj, pOff, lOff, pCount, lCount = run(
        name, img_origin, img_spacing, img_size, img_epsg, dem_gt, dem_size, dem_epsg)
    assert pCount > 0 and lCount > 0, f'{name}: derived grid does not cover the footprint'
    out['cases'].append({
        'name': name,
        'image': {'origin': list(img_origin), 'spacing': list(img_spacing),
                  'size': list(img_size), 'epsg': img_epsg},
        'dem': {'geotransform': [dem_gt[0], dem_gt[2], 0.0, dem_gt[1], 0.0, dem_gt[3]],
                'size': list(dem_size), 'epsg': dem_epsg},
        'expect': {
            'xlim': [float(v) for v in obj._xlim],
            'ylim': [float(v) for v in obj._ylim],
            'xlim_hex': [float(v).hex() for v in obj._xlim],
            'ylim_hex': [float(v).hex() for v in obj._ylim],
            'pOff': pOff, 'lOff': lOff, 'pCount': pCount, 'lCount': lCount,
        },
    })
    print(f'{name}: xlim={obj._xlim} ylim={obj._ylim}')
    print(f'    grid: origin=({dem_gt[0]},{dem_gt[1]}) spacing={spacing} size={dem_size}')
    print(f'    window: pOff={pOff} lOff={lOff} pCount={pCount} lCount={lCount}')

for case in FAIL_CASES:
    name = case[0]
    obj, pOff, lOff, pCount, lCount = run(*case)
    _, img_origin, img_spacing, img_size, img_epsg, dem_gt, dem_size, dem_epsg = case
    out['fail_cases'].append({
        'name': name,
        'image': {'origin': list(img_origin), 'spacing': list(img_spacing),
                  'size': list(img_size), 'epsg': img_epsg},
        'dem': {'geotransform': [dem_gt[0], dem_gt[2], 0.0, dem_gt[1], 0.0, dem_gt[3]],
                'size': list(dem_size), 'epsg': dem_epsg},
        'expect': {'pOff': pOff, 'lOff': lOff, 'pCount': pCount, 'lCount': lCount},
    })
    print(f'{name}: window pCount={pCount} lCount={lCount}  (reference produces this and '
          f'passes it to GDAL)')

import geogrid
out['provenance'] = {
    'autorift_version': geogrid.__version__,
    'gdal_version': gdal.__version__,
    'proj_version': f'{osr.GetPROJVersionMajor()}.{osr.GetPROJVersionMinor()}.{osr.GetPROJVersionMicro()}',
    'zrange': [-200, 4000],
}

with open(os.path.join(HERE, 'window.json'), 'w') as fh:
    json.dump(out, fh, indent=1, sort_keys=True)
print('\nwrote window.json:', len(out['cases']), 'cases,', len(out['fail_cases']), 'fail cases')
