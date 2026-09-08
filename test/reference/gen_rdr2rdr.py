"""Generate the pixel_offset fixture: the same ground point solved against two acquisitions' orbits.

`pixel_offset` composes two zero-Doppler solves — one against each acquisition — and differences the
results in each acquisition's own pixel units. isce3 exposes the halves as `isce3.geometry.geo2rdr`, so
the composition is checked against it directly rather than against the compiled geogrid kernel, which
has no equivalent operation: geogrid assumes the secondary has already been resampled onto the
reference's grid.

The inputs are a real Sentinel-1 pair — 24 days apart, two exact orbit cycles, the smallest baseline a
repeat pass offers — reduced to what the solve needs: the state vectors bracketing each acquisition, the
range and azimuth scales, and one ECEF target at the reference's scene center. That is a few kilobytes,
so the fixture is committed and the test needs no granule and no network.

isce3's `geo2rdr` runs here at a tighter threshold than its default (1e-10 m, 100 iterations) so the
recorded answer is the converged fixed point rather than the solver's own stopping point. The Julia side
runs `GEO2RDR_ITERATIONS` of a linearly-converging iteration and agrees to 1.15e-9 of a range sample,
which is 2.7 nm of ground position — the bound `REFERENCE.md` establishes for the ellipsoid conversions.

Run in the reference environment:

    micromamba run -n geogrid-ref python test/reference/gen_rdr2rdr.py

Reads the input half of `rdr2rdr.json` and rewrites the file with the isce3 answer, so regenerating it
needs neither the granules the inputs were extracted from nor the orbit files.
"""

import json
import os

import isce3
import numpy as np
from isce3.core import DateTime, Ellipsoid, LUT2d, LookSide, Orbit, StateVector
from isce3.geometry import geo2rdr

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "rdr2rdr.json")


def build_orbit(o):
    """The acquisition's state vectors as an isce3 Orbit on its own epoch."""
    epoch = DateTime(o["epoch"])
    svs = [
        StateVector(epoch + isce3.core.TimeDelta(float(t)), np.array(p), np.array(v))
        for t, p, v in zip(o["time"], o["pos"], o["vel"])
    ]
    return Orbit(svs, epoch)


def solve(o, xyz):
    """Zero-Doppler azimuth time and slant range at which `o` images `xyz`, and those in pixels."""
    ellipsoid = Ellipsoid()
    llh = ellipsoid.xyz_to_lon_lat(np.array(xyz))
    side = LookSide.Right if o["side"] == "right" else LookSide.Left
    aztime, rng = geo2rdr(
        llh, ellipsoid, build_orbit(o), LUT2d(), o["wavelength"], side,
        threshold=1e-10, maxiter=100, delta_range=1e-4,
    )
    return {
        "aztime": aztime,
        "range": rng,
        "samp": (rng - o["r0"]) / o["dr"],
        "line": (aztime - o["t0"]) * o["prf"],
    }


def main():
    with open(FIXTURE) as f:
        d = json.load(f)

    ref = solve(d["ref"], d["target_xyz"])
    sec = solve(d["sec"], d["target_xyz"])

    d["isce3"] = {
        "dsamp": sec["samp"] - ref["samp"],
        "dline": sec["line"] - ref["line"],
        "ref_aztime": ref["aztime"],
        "ref_range": ref["range"],
        "sec_aztime": sec["aztime"],
        "sec_range": sec["range"],
    }
    d["versions"] = {"isce3": isce3.__version__}

    with open(FIXTURE, "w") as f:
        json.dump(d, f, indent=1)
    print("dsamp = %.17g\ndline = %.17g" % (d["isce3"]["dsamp"], d["isce3"]["dline"]))


if __name__ == "__main__":
    main()
