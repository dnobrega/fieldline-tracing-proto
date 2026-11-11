"""Measure scalar-diagnostic and winding sensitivity to saved-point density.

Run: python validation/diagnostic_sensitivity.py
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from types import SimpleNamespace

os.environ.setdefault("NUMBA_NUM_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numba
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from fieldline_tracing_proto.tracing import trace_field_lines
from fieldline_tracing_proto.sampling import sample_scalar_on_lines
from fieldline_tracing_proto.diagnostics import compute_line_diagnostics
from fieldline_tracing_proto.winding import compute_local_winding
from validation.run_validation import write_csv, format_table


CENTER = 7 / 32
SIGMA = 1 / 32


def gaussian(z):
    return np.exp(-0.5 * ((z - CENTER) / SIGMA) ** 2)


def scalar_reference():
    integral = quad(gaussian, -1.0, 1.0, epsabs=1e-14)[0]
    centroid = quad(lambda z: z * gaussian(z), -1.0, 1.0, epsabs=1e-14)[0] / integral
    percentiles = [
        brentq(
            lambda z: quad(gaussian, -1.0, z, epsabs=1e-14)[0] / integral - p,
            -1.0,
            1.0,
            xtol=1e-14,
        )
        for p in (0.16, 0.84)
    ]
    return dict(
        integral=integral,
        mean=integral / 2,
        max_abs=1.0,
        centroid=centroid,
        width=percentiles[1] - percentiles[0],
    )


def scalar_case(method, stride=1, max_step=1 / 64):
    zgrid = np.linspace(-1.0, 1.0, 1025)
    xy = np.array([-1.0, 1.0])
    x, y, z = np.meshgrid(xy, xy, zgrid, indexing="ij")
    controls = dict(
        ds=1 / 512,
        max_steps=512,
        save_stride=stride,
        dtype=np.float64,
        periodic_x=False,
        periodic_y=False,
    )
    if method == "rk45":
        controls.update(
            method="rk45",
            ds=max_step,
            max_steps=20000,
            max_step=max_step,
            max_length=1.0,
            rtol=1e-9,
            atol=1e-11,
        )
    result = trace_field_lines(
        np.zeros_like(x),
        np.zeros_like(x),
        np.ones_like(x),
        xy,
        xy,
        zgrid,
        np.zeros((1, 3)),
        **controls,
        periodic_convention="endpoint_included",
    )
    sample = sample_scalar_on_lines(
        result,
        gaussian(z),
        xy,
        xy,
        zgrid,
        dtype=np.float64,
        periodic_x=False,
        periodic_y=False,
    )
    assert np.all(sample.codes_all == 0)
    diagnostics = compute_line_diagnostics(result, sample, dtype=np.float64)
    return result, diagnostics


def scalar_study():
    reference = scalar_reference()
    rows = []
    cases = [
        ("rk2-stride", "rk2", s, 1 / 512) for s in (1, 2, 4, 8, 16, 32, 64, 128, 512)
    ]
    cases += [
        ("rk45-stepcap", "rk45", 1, step)
        for step in (1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 256)
    ]
    for study, method, stride, step in cases:
        result, diagnostics = scalar_case(method, stride, step)
        rows.append(
            dict(
                study=study,
                method=method,
                save_stride=stride,
                integration_step_cap=step,
                saved_points=result.total_points,
                max_saved_gap=float(np.max(np.diff(result.sfl))),
                integral_relative_error=abs(
                    float(diagnostics.integral[0]) / reference["integral"] - 1
                ),
                mean_relative_error=abs(
                    float(diagnostics.mean[0]) / reference["mean"] - 1
                ),
                maximum_relative_error=abs(float(diagnostics.max_abs[0]) - 1),
                centroid_absolute_error=abs(
                    float(diagnostics.z_centroid_abs[0]) - reference["centroid"]
                ),
                width_absolute_error=abs(
                    float(diagnostics.W68_s_abs[0]) - reference["width"]
                ),
                trajectory_max_error=float(
                    np.max(np.abs(result.zfl - (result.sfl - 1.0)))
                ),
            )
        )
    return rows


def synthetic_pair(points, turns=3.0):
    u = np.linspace(0.0, 1.0, points)
    theta = 2 * np.pi * turns * u
    return SimpleNamespace(
        xfl=np.r_[np.zeros_like(u), np.cos(theta)],
        yfl=np.r_[np.zeros_like(u), np.sin(theta)],
        sfl=np.r_[u, u],
        line_start=np.array([0, points]),
        line_count=np.array([points, points]),
    )


def pair_winding(result, nsample):
    return float(
        compute_local_winding(
            result,
            (1, 2),
            neighbor_offsets=((0, 1),),
            nsample=nsample,
            dtype=np.float64,
            parameterization="normalized",
            plane="xy",
        ).mean_wind[0]
    )


def winding_study():
    rows = []
    for points in (5, 9, 17, 33, 65, 129):
        for nsample in (5, 7, 9, 17, 33, 65, 129, 257):
            winding = pair_winding(synthetic_pair(points), nsample)
            rows.append(
                dict(
                    study="synthetic",
                    method="analytic",
                    save_stride=0,
                    saved_points_per_line=points,
                    nsample=nsample,
                    expected_winding=3.0,
                    measured_winding=winding,
                    absolute_error=abs(winding - 3.0),
                )
            )
    # End-to-end pair on equal-radius helices, same normalized arc-length phases.
    xy = np.linspace(-2.0, 2.0, 9)
    zg = np.linspace(-4.0, 4.0, 5)
    x, y, z = np.meshgrid(xy, xy, zg, indexing="ij")
    seeds = np.array([[1.0, 0.0, 0.0], [np.cos(0.2), np.sin(0.2), 0.0]])
    length = np.pi * np.sqrt(2.0)
    for method in ("rk2", "rk45"):
        for stride in (1, 4, 16, 64, 256, 4096):
            controls = dict(ds=length / 2048, max_steps=2048)
            if method == "rk45":
                controls.update(
                    method=method,
                    ds=0.3,
                    max_steps=20000,
                    max_length=length,
                    rtol=1e-9,
                    atol=1e-11,
                )
            result = trace_field_lines(
                -y,
                x,
                np.ones_like(x),
                xy,
                xy,
                zg,
                seeds,
                save_stride=stride,
                periodic_x=False,
                periodic_y=False,
                dtype=np.float64,
                **controls,
                periodic_convention="endpoint_included",
            )
            winding = pair_winding(result, 257)
            rows.append(
                dict(
                    study="traced-helix",
                    method=method,
                    save_stride=stride,
                    saved_points_per_line=int(result.line_count[0]),
                    nsample=257,
                    expected_winding=1.0,
                    measured_winding=winding,
                    absolute_error=abs(winding - 1.0),
                )
            )
    return rows


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    scalar = scalar_study()
    winding = winding_study()
    write_csv(output / "scalar.csv", scalar)
    write_csv(output / "winding.csv", winding)
    root = Path(__file__).resolve().parents[1]
    environment = dict(
        python=platform.python_version(),
        numpy=np.__version__,
        numba=numba.__version__,
        threads=numba.get_num_threads(),
        dtype="float64",
        scalar_reference=scalar_reference(),
        source_sha256={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__).resolve(), *sorted((root / "src").rglob("*.py"))]
        },
    )
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    synthetic = [
        r
        for r in winding
        if r["study"] == "synthetic"
        and r["saved_points_per_line"] in (5, 17)
        and r["nsample"] in (5, 17, 257)
    ]
    traced = [r for r in winding if r["study"] == "traced-helix"]
    report = f"""# Diagnostic sensitivity to saved-point density

Reproduce: `python validation/diagnostic_sensitivity.py`.
Raw tables: `scalar.csv`, `winding.csv`; controls, reference values and source
hashes: `environment.json`. Float64 storage, two Numba threads by default.

## Scalar diagnostics

Trace the exact straight line `(0,0,z)`, z in [-1,1], with a Gaussian scalar
`exp(-0.5*((z-7/32)/(1/32))**2)`. Quadrature supplies the truncated integral,
centroid and continuous weighted percentiles. Sampling uses the actual scalar
interpolator on 1025 z nodes; all saved coordinates in this study are nodes,
so interpolation adds no scalar-value error. RK2 always takes 512 steps per
branch while only save_stride changes. RK45 changes max_step, with stride=1.
Both methods have essentially exact trajectory geometry here.

{format_table(scalar, ["study", "save_stride", "saved_points", "max_saved_gap", "integral_relative_error", "maximum_relative_error", "width_absolute_error"])}

Small trajectory error does not guarantee accurate scalar diagnostics.
Saving too sparsely can miss a localized scalar peak. Tightening RK45 rtol
cannot help this constant-direction field: its trajectory error is zero,
independently of the Gaussian scalar. A suitable maximum step or independent
output sampling is needed to resolve that scalar. Independent uniform output
sampling is not yet part of the API.

The percentile diagnostic inverts piecewise-linear absolute density within
segments. Its error can still depend on output spacing even when trapezoidal
integrals are accurate. Compare width with the continuous reference without
claiming higher-order percentile convergence. Full CSV includes mean and
centroid errors and trajectory residuals.

## Winding: geometric sampling and angular resampling are separate

Two synthetic curves have horizontal separation `(cos(6*pi*u),sin(6*pi*u))`,
u in [0,1], hence exactly 3 turns. They isolate saved geometry from tracing.
The winding implementation linearly interpolates saved x/y coordinates and
unwraps successive angle differences using nsample normalized-arc samples.
This study explicitly requests the historical normalized/XY proxy; the
current default uses seed-relative arc and a mean-tangent plane.

{format_table(synthetic, ["saved_points_per_line", "nsample", "expected_winding", "measured_winding", "absolute_error"])}

Increasing nsample cannot reconstruct turns already lost in sparse saved
geometry. Conversely, fine geometry can still alias when nsample is too
small. Near coincidence and periodic coordinate jumps are additional issues
not removed by these tests; this remains a local proxy, not Gauss linking.

An end-to-end affine-field check traces neighboring equal-radius helices
in `B=(-y,x,1)` for a complete stored turn (both branches together), with
expected winding 1. The two seeds have the same radius, so equal normalized
arc length compares equal phase progression. All cases use nsample=257.

{format_table(traced, ["method", "save_stride", "saved_points_per_line", "measured_winding", "absolute_error"])}

## Practical interpretation

Converge scalar diagnostics in saved arc-length spacing separately from the
trajectory tolerance. For winding, independently refine saved geometry and
nsample, and inspect near coincidences and periodic seams. No universal
save_stride or nsample threshold follows from this small analytic study.
The error tables document limitations of sampling, not a solver speed claim.
"""
    (output / "REPORT.md").write_text(report)
    print(
        f"Wrote {output / 'REPORT.md'}: {len(scalar)} scalar and {len(winding)} winding cases."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "diagnostics",
    )
    main(parser.parse_args().output_dir)
