#!/usr/bin/env python
"""Run the compiled reference over a real NISAR acquisition and a real parameter grid.

Where `gen_radar_geogrid.py` drives `geogridRadar` on a synthetic orbit and synthetic input rasters,
this drives it on a real acquisition's metadata and the ITS_LIVE 120 m parameter rasters, so the nine
outputs it writes are a real-data reference for the Julia radar path.

No SLC is needed. `GeogridRadar` takes every radar parameter as a scalar and the orbit as a
constructed `isce3.core.Orbit`, so the acquisition metadata harvested by
`SLCDatasets.jl/test/reference/gen_nisar_metadata.py` is enough — a few kilobytes of JSON in place of two
12 GB granules.

The run sequence mirrors `testGeogrid.py:427-488`: set the radar parameters from image 1, the repeat
interval from image 2, the eleven input rasters and the nine output names, then `getIncidenceAngle()`
followed by `geogridRadar()`.

Run in the reference environment:

    micromamba run -n geogrid-ref python test/reference/gen_radar_realdata.py \
        --reference <ref_metadata.json> --secondary <sec_metadata.json> \
        --params ~/data/autorift/tests/params --outdir <dir>
"""
import argparse
from datetime import datetime, timedelta
import json
import os
import time

import numpy as np
from osgeo import gdal

import isce3
from isce3.core import DateTime, Orbit, StateVector, TimeDelta
from geogrid import GeogridRadar

gdal.UseExceptions()

# `testGeogrid.py:358` in production.
CHIP_SIZE_X0 = 240
NODATA_OUT = -32767

# The nine outputs, named as the reference's consumers expect (`testGeogrid.py:472-480`).
OUTPUTS = {
    'winlocname': 'window_location.tif',
    'winoffname': 'window_offset.tif',
    'winsrname': 'window_search_range.tif',
    'wincsminname': 'window_chip_size_min.tif',
    'wincsmaxname': 'window_chip_size_max.tif',
    'winssmname': 'window_stable_surface_mask.tif',
    'winro2vxname': 'window_rdr_off2vel_x_vec.tif',
    'winro2vyname': 'window_rdr_off2vel_y_vec.tif',
    'winsfname': 'window_scale_factor.tif',
}

# The ITS_LIVE parameter rasters, by the attribute each is assigned to.
PARAMS = {
    'demname': 'h',
    'dhdxname': 'dhdx',
    'dhdyname': 'dhdy',
    'vxname': 'vx',
    'vyname': 'vy',
    'srxname': 'vxSearchRange',
    'sryname': 'vySearchRange',
    'csminxname': 'xMinChipSize',
    'csminyname': 'yMinChipSize',
    'csmaxxname': 'xMaxChipSize',
    'csmaxyname': 'yMaxChipSize',
    'ssmname': 'StableSurface',
}


def gx(v):
    """The fixture's hex literal, which round-trips the last bit where its decimal need not."""
    return float.fromhex(v['hex'])


def build_orbit(meta):
    """The acquisition's state vectors as an `isce3.core.Orbit`.

    The epoch comes from the CF units string the product carries, and the times are seconds against
    it, so the orbit is on exactly the clock the azimuth times are.
    """
    orb = meta['orbit']
    epoch = DateTime(orb['epoch'].split('since')[1].strip().replace(' ', 'T'))
    svs = []
    for i in range(orb['n']):
        t = gx(orb['time'][i])
        p = np.array([gx(c) for c in orb['position'][i]])
        v = np.array([gx(c) for c in orb['velocity'][i]])
        svs.append(StateVector(epoch + TimeDelta(t), p, v))
    return Orbit(svs, epoch), epoch


def write_orbit_eof(path, meta):
    """The state vectors in the tag format `geogridRadar.cpp:333-427` scrapes.

    `setOrbit_Py` passes `orbitname` — a path — to the C++, which parses the orbit itself rather than
    taking the `isce3.core.Orbit` the Python object also holds. That object is used only for the
    incidence angle, so the state vectors have to reach the kernel through a file.

    Only `<UTC>UTC=...`, `<X unit=...>` and its five siblings are read, one state vector per `<OSV>`,
    so this is the minimal document that satisfies the parser. The values are written with `repr` to
    keep every bit of the double.
    """
    orb = meta['orbit']
    epoch = datetime.strptime(
        orb['epoch'].split('since')[1].strip().replace(' ', 'T'), '%Y-%m-%dT%H:%M:%S')
    with open(path, 'w') as fh:
        fh.write('<?xml version="1.0"?>\n<Earth_Explorer_File>\n<Data_Block>\n<List_of_OSVs>\n')
        for i in range(orb['n']):
            t = gx(orb['time'][i])
            p = [gx(c) for c in orb['position'][i]]
            v = [gx(c) for c in orb['velocity'][i]]
            stamp = (epoch + timedelta(seconds=t)).strftime('%Y-%m-%dT%H:%M:%S.%f')
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
    return path


def seconds_since_midnight(epoch, seconds):
    """An azimuth time on the seconds-since-midnight scale the reference measures the index against.

    `geogridRadar` takes `sensingStart` as a `datetime` and `aztime` as seconds against the orbit's
    epoch, so both scales are needed and the offset between them is the epoch's own time of day.
    """
    midnight = DateTime(epoch.isoformat().split('T')[0] + 'T00:00:00')
    return float((epoch - midnight).total_seconds()) + seconds


def run(ref_meta, sec_meta, params_dir, outdir, epsg=3413):
    os.makedirs(outdir, exist_ok=True)
    prev = os.getcwd()
    os.chdir(outdir)
    try:
        g = ref_meta['geometry']
        orbit, epoch = build_orbit(ref_meta)

        sensing_start = gx(g['sensing_start'])
        sensing_stop = gx(g['sensing_stop'])
        prf = gx(g['prf'])

        obj = GeogridRadar()

        # Radar parameters come from image 1 alone (`testGeogrid.py:427-446`).
        obj.startingRange = gx(g['starting_range'])
        obj.rangePixelSize = gx(g['range_pixel_spacing'])
        obj.prf = prf
        obj.wavelength = gx(g['wavelength'])
        obj.numberOfLines = g['nlines']
        obj.numberOfSamples = g['nsamples']
        obj.lookSide = (isce3.core.LookSide.Left
                        if ref_meta['identification']['look_direction'].lower() == 'left'
                        else isce3.core.LookSide.Right)

        # `sensingStart` and `sensingStop` must be Python `datetime`s: `GeogridRadar.setState` takes
        # the azimuth time as `sensingStart` minus its own midnight (`GeogridRadar.py:328-330`), which
        # is the seconds-since-midnight scale, and calls `.replace(hour=0, ...)` to get there.
        obj.sensingStart = datetime.strptime(
            (epoch + TimeDelta(sensing_start)).isoformat_usec()[:26], '%Y-%m-%dT%H:%M:%S.%f')
        obj.sensingStop = datetime.strptime(
            (epoch + TimeDelta(sensing_stop)).isoformat_usec()[:26], '%Y-%m-%dT%H:%M:%S.%f')
        obj.aztime = sensing_start
        obj.orbit = orbit
        # The C++ parses the orbit from this path; the `Orbit` object above serves the
        # incidence angle only.
        obj.orbitname = write_orbit_eof(os.path.join(outdir, 'orbit.EOF'), ref_meta)

        # The secondary contributes only the repeat interval.
        sec_orbit_epoch = DateTime(
            sec_meta['orbit']['epoch'].split('since')[1].strip().replace(' ', 'T'))
        sec_start = gx(sec_meta['geometry']['sensing_start'])
        obj.repeatTime = float(
            ((sec_orbit_epoch + TimeDelta(sec_start)) - (epoch + TimeDelta(sensing_start)))
            .total_seconds())

        obj.nodata_out = NODATA_OUT
        obj.chipSizeX0 = CHIP_SIZE_X0

        dem = os.path.join(params_dir, f'NPS_0120m_{PARAMS["demname"]}.tif')
        info = gdal.Info(dem, format='json')
        obj.gridSpacingX = info['geoTransform'][1]
        obj.epsg = epsg

        for attr, stem in PARAMS.items():
            setattr(obj, attr, os.path.join(params_dir, f'NPS_0120m_{stem}.tif'))
        for attr, name in OUTPUTS.items():
            setattr(obj, attr, name)

        t0 = time.time()
        obj.getIncidenceAngle()
        t_incidence = time.time() - t0

        t0 = time.time()
        obj.geogridRadar()
        t_kernel = time.time() - t0

        record = {
            'reference': ref_meta.get('source'),
            'secondary': sec_meta.get('source'),
            'grid': {'epsg': epsg, 'spacing': obj.gridSpacingX, 'dem': os.path.basename(dem)},
            'radar': {
                'starting_range': {'dec': obj.startingRange,
                                   'hex': float(obj.startingRange).hex()},
                'range_pixel_spacing': {'dec': obj.rangePixelSize,
                                        'hex': float(obj.rangePixelSize).hex()},
                'prf': {'dec': obj.prf, 'hex': float(obj.prf).hex()},
                'wavelength': {'dec': obj.wavelength, 'hex': float(obj.wavelength).hex()},
                'aztime': {'dec': obj.aztime, 'hex': float(obj.aztime).hex()},
                'sensing_start_sod': {
                    'dec': seconds_since_midnight(epoch, sensing_start),
                    'hex': float(seconds_since_midnight(epoch, sensing_start)).hex()},
                'nlines': obj.numberOfLines,
                'nsamples': obj.numberOfSamples,
                'look_side': str(obj.lookSide),
                'repeat_time_s': obj.repeatTime,
                'incidence_angle': {'dec': obj.incidenceAngle,
                                    'hex': float(obj.incidenceAngle).hex()},
            },
            'window': {'pOff': obj.pOff, 'lOff': obj.lOff,
                       'pCount': obj.pCount, 'lCount': obj.lCount},
            'resolution': {'X_res': obj.X_res, 'Y_res': obj.Y_res},
            'npoints': obj.pCount * obj.lCount,
            'times_s': {'incidence': t_incidence, 'kernel': t_kernel},
            'versions': {'gdal': gdal.__version__, 'isce3': isce3.__version__},
        }
        json.dump(record, open('run.json', 'w'), indent=1)
        # The Julia comparison builds its coordinate from these rather than from the record above, so
        # it needs the two metadata files beside the outputs.
        json.dump(ref_meta, open('reference_metadata.json', 'w'), indent=1)
        json.dump(sec_meta, open('secondary_metadata.json', 'w'), indent=1)
        print(json.dumps(record['window'], indent=1))
        print(f'incidence angle : {obj.incidenceAngle} rad')
        print(f'window          : {obj.pCount} x {obj.lCount}')
        print(f'kernel          : {t_kernel:.3f} s')
        print(f'wrote           : {outdir}')
        return record
    finally:
        os.chdir(prev)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', required=True, help='harvested metadata JSON for image 1')
    ap.add_argument('--secondary', required=True, help='harvested metadata JSON for image 2')
    ap.add_argument('--params', required=True, help='directory of NPS_0120m_*.tif rasters')
    ap.add_argument('--outdir', required=True, help='where to write the nine GeoTIFFs and run.json')
    ap.add_argument('--epsg', type=int, default=3413)
    a = ap.parse_args()
    run(json.load(open(a.reference)), json.load(open(a.secondary)),
        os.path.expanduser(a.params), os.path.expanduser(a.outdir), a.epsg)


if __name__ == '__main__':
    main()
