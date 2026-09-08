"""Generate the sinc interpolation fixture: the kernel table, and isce3's interpolated values.

`pixel_offset` says where a secondary sample sits; a resampler has to *read* it, and reading between
samples of a complex image is sinc interpolation. This fixture pins both halves of that against isce3.

The kernel table is regenerated from `_sinc_coef` plus the per-position normalization
(`cxx/isce3/core/Sinc2dInterpolator.cpp`) rather than read out of the library, because isce3 exposes no
binding for the coefficients — only for interpolation through `resample_to_coords`. So the table is
checked as an algorithm transcription and the interpolated values are checked against the compiled
routine, which together cover both.

Recorded as hex float literals, since `float.hex()` round-trips exactly where a decimal repr does not,
and these are asserted bitwise: the coefficients are `cos` and `sin` of exactly representable arguments
followed by a normalization, so bitwise is achievable and is the right standard.

Run in the reference environment:

    micromamba run -n geogrid-ref python test/reference/gen_sinc.py
"""

import json
import os

import isce3
import numpy as np
from isce3.core import DateTime, LUT2d
from isce3.image.v2.resample_slc import resample_to_coords
from isce3.product import RadarGridParameters

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "sinc.json")

SINC_LEN, SINC_SUB = 8, 8192


def kernel_table(kl=SINC_LEN, dec=SINC_SUB, beta=1.0, pedestal=0.0, weight=1):
    """isce3's Sinc2dInterpolator kernel: _sinc_coef, then normalized per sub-pixel position."""
    n = dec * kl
    f = np.empty(n)
    h = (1.0 - pedestal) / 2.0
    soff = (n - 1) / 2.0
    for i in range(n):
        wgt = (1.0 - h) + h * np.cos((np.pi * (i - soff)) / soff)
        s = (np.floor(i - soff) * beta) / dec
        fct = np.sin(np.pi * s) / (np.pi * s) if s != 0.0 else 1.0
        f[i] = fct * wgt if weight == 1 else fct
    K = np.empty((dec, kl))
    for i in range(dec):
        ssum = sum(f[i + dec * j] for j in range(kl))
        for j in range(kl):
            K[i, j] = f[i + dec * j] / ssum
    return K


def main():
    K = kernel_table()

    # A few rows spanning the table: the ends, where the Hamming window is at its extremes, and the
    # middle. Every tap of each, as hex, so the assertion is bitwise.
    rows = [0, 1, 2, 4095, 4096, 8190, 8191]
    table = {str(r): [float.hex(v) for v in K[r]] for r in rows}
    sums = {str(r): float.hex(float(K[r].sum())) for r in rows}

    # Interpolation against the compiled routine. A random complex block, queried at positions chosen to
    # exercise the truncating table index: exact integers, exact halves, and values a hair either side of
    # an integer where the truncation and a rounding would disagree.
    np.random.seed(7)
    n = 32
    blk = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex64)
    queries = [
        (12.0, 12.0), (12.25, 13.5), (15.5, 15.5), (20.125, 18.875),
        (12.0001, 12.9999), (17.99999, 14.00001), (13.5 - 1e-9, 19.5 + 1e-9),
    ]
    rg = np.array([[q[0] for q in queries]], dtype=np.float64)
    az = np.array([[q[1] for q in queries]], dtype=np.float64)
    grid = RadarGridParameters(
        sensing_start=0.0, wavelength=0.055, prf=1000.0, starting_range=8e5,
        range_pixel_spacing=2.3, lookside=isce3.core.LookSide.Right,
        length=n, width=n, ref_epoch=DateTime("2020-01-01T00:00:00"),
    )
    out = resample_to_coords(blk, rg, az, grid, LUT2d())

    d = {
        "kernel": {"length": SINC_LEN, "decimation": SINC_SUB, "rows": table, "rowsums": sums},
        "block_real": [[float.hex(float(v)) for v in row] for row in blk.real],
        "block_imag": [[float.hex(float(v)) for v in row] for row in blk.imag],
        "queries": [list(q) for q in queries],
        "isce3": [[float.hex(float(v.real)), float.hex(float(v.imag))] for v in out[0]],
        "versions": {"isce3": isce3.__version__},
    }
    with open(FIXTURE, "w") as fh:
        json.dump(d, fh, indent=1)
    print("wrote %s: %d kernel rows, %d queries" % (FIXTURE, len(rows), len(queries)))


if __name__ == "__main__":
    main()
