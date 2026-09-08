"""Generate the ResampledSLC fixture: a block resampled by isce3 against a known offset field.

`ResampledSLC` presents the secondary acquisition's samples on the reference's grid. isce3's
`resample_to_coords` does the same thing given index grids, so it is the oracle — and it is the routine
the NISAR workflow actually calls, rather than the older `ResampSlc`.

The offsets here are deliberately a mix: an integer shift (which must reproduce the samples exactly), a
half-sample shift (where the kernel is symmetric), an arbitrary fraction, and a NaN (which must yield the
fill value rather than an interpolation somewhere arbitrary). A field varying across the block rather than
one constant offset, so a transposed axis or a swapped sign shows up as a mismatch instead of cancelling.

Zero Doppler, which is the case every acquisition this package handles has and the one where both of
`resampleToCoords`'s phasors are unity.

Run in the reference environment:

    micromamba run -n geogrid-ref python test/reference/gen_resamp.py
"""

import json
import os

import isce3
import numpy as np
from isce3.core import DateTime, LUT2d
from isce3.image.v2.resample_slc import resample_to_coords
from isce3.product import RadarGridParameters

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "resamp.json")

N = 48
OUT = 12


def main():
    rng = np.random.default_rng(23)
    block = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))).astype(np.complex64)

    # An offset field over a 12x12 output window placed well inside the block, so every stencil fits
    # except where a NaN is planted deliberately.
    base_az, base_rg = 16, 16
    az_off = np.zeros((OUT, OUT))
    rg_off = np.zeros((OUT, OUT))
    for i in range(OUT):
        for j in range(OUT):
            # A smooth ramp plus a constant, which is what a real offset field looks like.
            rg_off[i, j] = 3.0 + 0.25 * j + 0.05 * i
            az_off[i, j] = -2.0 + 0.5 * i - 0.125 * j
    # Exact integers and exact halves, to pin those cases.
    rg_off[0, 0], az_off[0, 0] = 4.0, -3.0
    rg_off[0, 1], az_off[0, 1] = 4.5, -2.5
    # A hole: the geometry could not place this point.
    rg_off[5, 5], az_off[5, 5] = np.nan, np.nan

    # The absolute input indices, as `offsets_to_indices` forms them: output index plus offset. The output
    # window starts at (base_az, base_rg) on the reference grid and the input block is the whole array, so
    # the grid offset is the window's own start.
    az_idx = np.zeros((OUT, OUT))
    rg_idx = np.zeros((OUT, OUT))
    for i in range(OUT):
        for j in range(OUT):
            az_idx[i, j] = base_az + i + az_off[i, j]
            rg_idx[i, j] = base_rg + j + rg_off[i, j]

    grid = RadarGridParameters(
        sensing_start=0.0, wavelength=0.055, prf=1000.0, starting_range=8e5,
        range_pixel_spacing=2.3, lookside=isce3.core.LookSide.Right,
        length=N, width=N, ref_epoch=DateTime("2020-01-01T00:00:00"),
    )
    out = resample_to_coords(block, rg_idx, az_idx, grid, LUT2d())

    d = {
        "block_real": [[float.hex(float(v)) for v in row] for row in block.real],
        "block_imag": [[float.hex(float(v)) for v in row] for row in block.imag],
        "base": [base_az, base_rg],
        "rg_off": [[float.hex(float(v)) for v in row] for row in rg_off],
        "az_off": [[float.hex(float(v)) for v in row] for row in az_off],
        "isce3_real": [[float.hex(float(v.real)) for v in row] for row in out],
        "isce3_imag": [[float.hex(float(v.imag)) for v in row] for row in out],
        "versions": {"isce3": isce3.__version__},
    }
    with open(FIXTURE, "w") as fh:
        json.dump(d, fh, indent=1)
    finite = int(np.sum(np.isfinite(out.real)))
    print("wrote %s: %dx%d output, %d finite" % (FIXTURE, OUT, OUT, finite))


if __name__ == "__main__":
    main()
