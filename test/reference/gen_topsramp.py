"""Generate the TOPS azimuth carrier fixture.

The phase a TOPS acquisition's samples carry, which `ResampledSLC` removes before interpolating and
reapplies after. Unlike the sinc kernel and the resampler, this has no isce3 C++ equivalent to check
against: the deramp lives in the Python `s1reader` package, in `Sentinel1BurstSlc.az_carrier_components`
(`src/s1reader/s1_burst_slc.py`), which cites ESA's TOPS deramping note. So the fixture is generated from
that function's arithmetic rather than from a compiled kernel, and the Julia side is held to a tight
relative tolerance rather than to the bit.

What this pins, and what a round-trip test cannot: the polynomial argument convention (`range - r0`, with
`r0` the reference range rather than an index), the zero-based `n_lines // 2` center line, and the sign of
the reference time. A remove-then-reapply test stays green under any of those being wrong, because both
directions share the error.

The inputs are synthetic but physically plausible Sentinel-1 IW values — a 1500-line burst, a 2 ms azimuth
interval, an FM rate of a few thousand Hz/s, a 1.59 deg/s antenna sweep. Synthetic rather than extracted
from a granule because nothing here needs a real product: the arithmetic is a closed-form function of the
eight numbers below, and a committed granule would only obscure that.

`azimuthSteeringRate` is written in degrees per second in the annotation and used in radians per second, so
the conversion is part of what the consumer does and `ks` is recorded here already converted.

Run with `s1reader` importable to check the transcription against the package itself:

    micromamba run -n s1-reader python test/reference/gen_topsramp.py

With `s1reader` absent it falls back to the transcribed arithmetic below, which is asserted against the
package's own when both are available.
"""

import json
import math
import os

# One burst of Sentinel-1 IW, reduced to what the carrier needs.
LINES_PER_BURST = 1500
AZIMUTH_TIME_INTERVAL = 2.0e-3
STARTING_RANGE = 800_000.0
RANGE_PIXEL_SPACING = 2.33
WAVELENGTH = 0.0555
PLATFORM_SPEED = 7590.0
STEERING_RATE_DEG = 1.590368784

# Both polynomials are referenced to `R0` and quadratic in `range - R0`, as the annotation writes them.
R0 = 800_000.0
FM_RATE_COEFFS = (-2300.0, 0.5e-3, -1.0e-9)
DOPPLER_COEFFS = (-40.0, 0.25e-3, -2.0e-10)


def poly(coeffs, rng):
    t = rng - R0
    return coeffs[0] + coeffs[1] * t + coeffs[2] * t * t


def carrier(line, sample):
    """The phase in radians at a one-based (line, sample) of the burst.

    Mirrors `az_carrier_components` with `offset = 0`. Its `y` and `x` are zero-based, so a one-based index
    reaches the same place as `index - 1`.
    """
    y, x = line - 1, sample - 1
    steer = STEERING_RATE_DEG * math.pi / 180.0
    ks = 2 * PLATFORM_SPEED * steer / WAVELENGTH

    eta = (y - (LINES_PER_BURST // 2)) * AZIMUTH_TIME_INTERVAL
    rng = STARTING_RANGE + x * RANGE_PIXEL_SPACING

    f_etac = poly(DOPPLER_COEFFS, rng)
    ka = poly(FM_RATE_COEFFS, rng)
    eta_ref = (poly(DOPPLER_COEFFS, STARTING_RANGE) / poly(FM_RATE_COEFFS, STARTING_RANGE)) - (
        f_etac / ka
    )
    kt = ks / (1.0 - ks / ka)
    return math.pi * kt * ((eta - eta_ref) ** 2)


def carrier_from_s1reader(line, sample):
    """The same phase, from `s1reader`'s own dataclass, where the package is importable.

    Builds only the fields `az_carrier_components` reads, so this needs no granule.
    """
    import numpy as np
    from s1reader.s1_burst_slc import AzimuthCarrierComponents

    steer = STEERING_RATE_DEG * math.pi / 180.0
    ks = 2 * PLATFORM_SPEED * steer / WAVELENGTH
    y, x = line - 1, sample - 1

    eta = (y - (LINES_PER_BURST // 2)) * AZIMUTH_TIME_INTERVAL
    rng = STARTING_RANGE + x * RANGE_PIXEL_SPACING
    ka = poly(FM_RATE_COEFFS, rng)
    eta_ref = (poly(DOPPLER_COEFFS, STARTING_RANGE) / poly(FM_RATE_COEFFS, STARTING_RANGE)) - (
        poly(DOPPLER_COEFFS, rng) / ka
    )
    kt = ks / (1.0 - ks / ka)
    return float(AzimuthCarrierComponents(np.array(kt), eta, eta_ref).carrier)


# Corners, edges and the center line, plus a scatter across the swath. The center line is where `eta` is
# zero and the phase vanishes, which is the one point whose value is structural rather than numerical.
LINES = [1, 2, 375, 750, 751, 1125, 1499, 1500]
SAMPLES = [1, 2, 1000, 5000, 12000, 20000, 25000]


def main():
    steer = STEERING_RATE_DEG * math.pi / 180.0
    out = {
        "note": (
            "TOPS azimuth carrier phase from s1reader's az_carrier_components. Synthetic burst "
            "parameters; see gen_topsramp.py."
        ),
        "lines_per_burst": LINES_PER_BURST,
        "azimuth_time_interval": AZIMUTH_TIME_INTERVAL,
        "starting_range": STARTING_RANGE,
        "range_pixel_spacing": RANGE_PIXEL_SPACING,
        "wavelength": WAVELENGTH,
        "platform_speed": PLATFORM_SPEED,
        "steering_rate_deg": STEERING_RATE_DEG,
        "ks": 2 * PLATFORM_SPEED * steer / WAVELENGTH,
        "poly_reference_range": R0,
        "fm_rate_coeffs": list(FM_RATE_COEFFS),
        "doppler_coeffs": list(DOPPLER_COEFFS),
        "lines": LINES,
        "samples": SAMPLES,
    }

    try:
        reference = carrier_from_s1reader
        # Asserting the transcription rather than trusting it: where the package is importable, every point
        # must agree with the arithmetic above before either is recorded.
        for line in LINES:
            for sample in SAMPLES:
                a, b = carrier(line, sample), carrier_from_s1reader(line, sample)
                assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (line, sample, a, b)
        out["source"] = "s1reader.s1_burst_slc.AzimuthCarrierComponents"
    except ImportError:
        reference = carrier
        out["source"] = "transcribed from s1reader.s1_burst_slc.az_carrier_components"

    out["carrier"] = [[reference(line, sample) for sample in SAMPLES] for line in LINES]

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "topsramp.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"wrote {path} ({out['source']})")


if __name__ == "__main__":
    main()
