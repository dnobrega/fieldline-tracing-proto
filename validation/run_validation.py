"""Generate boundary and separated-error studies, CSV data and an English report.

Run from the project root: python validation/run_validation.py
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numba
import numpy as np
import scipy
from scipy.integrate import quad
from scipy.optimize import brentq

from fieldline_tracing_proto.tracing import trace_field_lines


def parabola_arc(y):
    """Primitive of sqrt(1+4*y**2), for B=(-2y,1,0)."""
    return 0.5 * y * np.sqrt(1 + 4 * y * y) + 0.25 * np.arcsinh(2 * y)


def parabola_trace(clearance, rtol=1e-9, max_step=np.inf):
    grid = np.linspace(-1.0, 1.0, 9)
    x, y, z = np.meshgrid(grid, grid, np.array([-1.0, 1.0]), indexing="ij")
    return trace_field_lines(
        -2 * y,
        np.ones_like(x),
        np.zeros_like(x),
        grid,
        grid,
        np.array([-1.0, 1.0]),
        np.array([[0.75 - clearance, -0.5, 0.0]]),
        0.3,
        10000,
        method="rk45",
        dtype=np.float64,
        periodic_x=False,
        periodic_y=False,
        rtol=rtol,
        atol=rtol * 0.01,
        min_step=1e-14,
        boundary_tol=1e-10,
        max_length=1.3,
        max_step=max_step,
        periodic_convention="endpoint_included",
    )


def cubic_reference(grid=None, length=1.2, alpha=0.75):
    """Independent quadrature/root reference for B=(1,0,alpha*x**2).

    If grid is provided, reference the *piecewise linear sampled field*,
    not the analytic polynomial. Its breakpoints are explicit quadrature nodes.
    """

    def bz(x):
        return (
            alpha * x * x if grid is None else np.interp(x, grid, alpha * grid * grid)
        )

    def arc(x):
        nodes = None if grid is None else grid[(grid > 0) & (grid < x)]
        return quad(
            lambda u: np.sqrt(1 + bz(u) ** 2),
            0.0,
            x,
            points=nodes,
            epsabs=2e-13,
            epsrel=2e-13,
            limit=300,
        )[0]

    endpoint = brentq(lambda x: arc(x) - length, 0.0, 1.5, xtol=1e-14)
    nodes = None if grid is None else grid[(grid > 0) & (grid < endpoint)]
    height = quad(
        bz, 0.0, endpoint, points=nodes, epsabs=2e-13, epsrel=2e-13, limit=300
    )[0]
    return np.array([endpoint, 0.0, height])


def cubic_trace(nodes, rtol, max_step=0.3):
    grid = np.linspace(-2.0, 2.0, nodes)
    yz = np.array([-2.0, 2.0])
    x, y, z = np.meshgrid(grid, yz, yz, indexing="ij")
    return trace_field_lines(
        np.ones_like(x),
        np.zeros_like(x),
        0.75 * x * x,
        grid,
        yz,
        yz,
        np.zeros((1, 3)),
        0.3,
        20000,
        method="rk45",
        dtype=np.float64,
        periodic_x=False,
        periodic_y=False,
        rtol=rtol,
        atol=rtol * 0.01,
        max_length=1.2,
        max_step=max_step,
        periodic_convention="endpoint_included",
    )


def convergence_study():
    exact = cubic_reference()
    rows = []
    # Fixed mesh: integration error should fall toward the interpolant reference,
    # while total error levels off at the field interpolation error.
    cases = [("tolerance", 17, tol, 0.3) for tol in (1e-3, 1e-5, 1e-7, 1e-9, 1e-11)]
    # Tight tolerance: change only the field's spatial resolution.
    cases += [("grid", nodes, 1e-11, 0.05) for nodes in (9, 17, 33, 65)]
    for study, nodes, rtol, max_step in cases:
        reference = cubic_reference(np.linspace(-2.0, 2.0, nodes))
        result = cubic_trace(nodes, rtol, max_step)
        position = np.array([result.xfl[-1], result.yfl[-1], result.zfl[-1]])
        rows.append(
            dict(
                study=study,
                nodes_x=nodes,
                spacing=4 / (nodes - 1),
                rtol=rtol,
                atol=rtol * 0.01,
                max_step=max_step,
                length=1.2,
                integration_error=float(np.linalg.norm(position - reference)),
                interpolation_error=float(np.linalg.norm(reference - exact)),
                total_error=float(np.linalg.norm(position - exact)),
                accepted=int(result.nsteps_plus[0]),
                rejected=int(result.nrejected_plus[0]),
                exit_plus=int(result.exit_code_plus[0]),
            )
        )
    return rows


def boundary_study():
    rows = []
    for clearance in (1e-3, 1e-6, 1e-9, 0.0, -1e-6):
        for tol in (1e-6, 1e-9, 1e-12):
            result = parabola_trace(clearance, tol)
            expected_code = 6 if clearance < 0 else 11
            expected_length = (
                parabola_arc(-np.sqrt(-clearance)) - parabola_arc(-0.5)
                if clearance < 0
                else 1.3
            )
            rows.append(
                dict(
                    scenario="parabola",
                    clearance=clearance,
                    rtol=tol,
                    max_step="unlimited",
                    expected_exit=expected_code,
                    actual_exit=int(result.exit_code_plus[0]),
                    expected_length=expected_length,
                    actual_length=float(result.length_plus[0]),
                    length_error=abs(float(result.length_plus[0]) - expected_length),
                    endpoint_x=float(result.xfl[-1]),
                    endpoint_y=float(result.yfl[-1]),
                    accepted=int(result.nsteps_plus[0]),
                    rejected=int(result.nrejected_plus[0]),
                    matches_expected_exit=int(result.exit_code_plus[0])
                    == expected_code,
                )
            )
    for step in (0.01, 0.001):
        result = parabola_trace(-1e-6, 1e-9, step)
        expected_length = parabola_arc(-0.001) - parabola_arc(-0.5)
        rows.append(
            dict(
                scenario="parabola",
                clearance=-1e-6,
                rtol=1e-9,
                max_step=step,
                expected_exit=6,
                actual_exit=int(result.exit_code_plus[0]),
                expected_length=expected_length,
                actual_length=float(result.length_plus[0]),
                length_error=abs(float(result.length_plus[0]) - expected_length),
                endpoint_x=float(result.xfl[-1]),
                endpoint_y=float(result.yfl[-1]),
                accepted=int(result.nsteps_plus[0]),
                rejected=int(result.nrejected_plus[0]),
                matches_expected_exit=int(result.exit_code_plus[0]) == 6,
            )
        )
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_table(rows, columns):
    def fmt(value):
        return f"{value:.3e}" if isinstance(value, float) else str(value)

    head = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    return "\n".join(
        [head, rule]
        + ["| " + " | ".join(fmt(row[key]) for key in columns) + " |" for row in rows]
    )


def generate_report(output):
    output.mkdir(parents=True, exist_ok=True)
    convergence = convergence_study()
    boundaries = boundary_study()
    write_csv(output / "convergence.csv", convergence)
    write_csv(output / "boundaries.csv", boundaries)
    environment = dict(
        python=platform.python_version(),
        platform=platform.platform(),
        machine=platform.machine(),
        numpy=np.__version__,
        numba=numba.__version__,
        scipy=scipy.__version__,
        numba_threads=numba.get_num_threads(),
        storage_dtype="float64",
        boundary_cases_all_match=all(r["matches_expected_exit"] for r in boundaries),
        crossing_cases_all_detected=all(
            r["matches_expected_exit"] for r in boundaries if r["clearance"] < 0
        ),
        interior_cases_all_complete=all(
            r["matches_expected_exit"] for r in boundaries if r["clearance"] > 0
        ),
        exact_contact_face_exits=sum(
            r["actual_exit"] == 6 for r in boundaries if r["clearance"] == 0
        ),
    )
    root = Path(__file__).resolve().parents[1]
    environment["source_sha256"] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [
            root / "src" / "fieldline_tracing_proto" / "tracing.py",
            root / "src" / "fieldline_tracing_proto" / "grid.py",
            root / "src" / "fieldline_tracing_proto" / "events.py",
            root / "tests/test_events.py",
            Path(__file__).resolve(),
            root / "tests/test_scientific_validation.py",
        ]
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    tolerance = [r for r in convergence if r["study"] == "tolerance"]
    grid = [r for r in convergence if r["study"] == "grid"]
    ratios = [
        grid[i - 1]["interpolation_error"] / grid[i]["interpolation_error"]
        for i in range(1, len(grid))
    ]
    missed_crossings = [
        r for r in boundaries if r["clearance"] < 0 and not r["matches_expected_exit"]
    ]
    contacts = [r for r in boundaries if r["clearance"] == 0]
    contact_exits = [r for r in contacts if r["actual_exit"] == 6]
    report = f"""# Scientific validation: boundaries and separated errors

This report is generated by `validation/run_validation.py`; numerical data
are in `convergence.csv` and `boundaries.csv`, with environment versions in
`environment.json`. Reproduce from the project root:

```sh
python validation/run_validation.py
python -m pytest -q
```

## Assessment

The original stage-only detector missed all three unrestricted short
exit/re-entry cases. The quartic within-step event search now detects the
first exit in those cases, and the former expected-failure tests pass.
The full current sweep has **{len(missed_crossings)} missed true crossings**.
All positive-clearance cases complete as expected. These results validate
the tested configurations, not every possible field or boundary event.

Exact contact remains sensitive to integration error: {len(contact_exits)} of
{len(contacts)} mathematical tangencies terminate with a nearby numerical face
exit. This discrepancy is retained explicitly in the CSV and environment
metadata (`boundary_cases_all_match`), rather than counted as exact geometric
agreement. An exactly tangent polynomial is not an exit in the event search;
an accumulated trajectory error can nevertheless turn a mathematical contact
into a crossing of the numerical trajectory.

## 1. Integration tolerance at fixed mesh

Use the smooth, divergence-free analytic field `B=(1,0,0.75*x²)` with seed
`(0,0,0)` and per-branch arc length 1.2. Its exact geometric trajectory is
`z=0.25*x³`, `y=0`. Scalar quadrature and a bracketed root solve determine
the endpoint at fixed arc length. The grid-sampled field is piecewise linear
in x, so a second reference integrates that *same interpolated field*, with
all cell boundaries explicitly supplied to the quadrature. No tracing
routine is used to build either reference.

The table reports Euclidean endpoint errors for the plus branch. Integration
error compares RK45 to the sampled-field reference; interpolation error
compares the sampled-field reference to the analytic-field reference; total
error compares RK45 to the analytic-field reference. These norms need not
add: errors are vectors. The mesh has 17 x nodes on [-2,2], and two nodes on
each transverse axis (the field is independent of those axes). `max_step=0.3`.

{format_table(tolerance, ["rtol", "integration_error", "interpolation_error", "total_error", "accepted", "rejected"])}

The tightest settings reduce integration error by several orders of magnitude,
while the total error approaches an interpolation-error floor. The first
change from rtol=1e-3 to 1e-5 actually increases the measured integration
error; convergence is not monotonic at these loose settings. Tightening a
local tolerance cannot recover information absent from the input mesh, and
individual ratios need not follow a clean fifth-order law across trilinear
cell interfaces.

## 2. Mesh resolution at tight tolerance

Here `rtol=1e-11`, `atol=1e-13`, `max_step=0.05` and the field/seed/length
are fixed. Only the x-grid resolution changes.

{format_table(grid, ["nodes_x", "spacing", "integration_error", "interpolation_error", "total_error"])}

Successive interpolation-error ratios on halving spacing are
{", ".join(f"{ratio:.3f}" for ratio in ratios)}. This is consistent with
second-order spatial interpolation error for this smooth polynomial case;
it is not a guarantee for arbitrary simulation data.

## 3. Tangency, near-tangency and missed excursions

Use the affine, divergence-free field `B=(-2y,1,0)`, exactly reproduced by
trilinear interpolation. Starting at `(0.75-clearance,-0.5,0)` gives
`x=1-clearance-y²`. The domain is [-1,1] on x/y/z. Positive clearance stays
inside; zero clearance touches x=1 and returns inside. Negative clearance
leaves the domain at `y=-sqrt(-clearance)` and must stop at that first exit.
The exact arc length follows the primitive
`F(y)=0.5*y*sqrt(1+4y²)+0.25*asinh(2y)`.

A maximum branch length of 1.3 ends the interior/touching cases before any
other face is reached. The table shows all true-exit cases, plus exact
contacts whose exit code differs from the mathematical prediction:

{format_table([r for r in boundaries if r["clearance"] < 0] + contact_exits, ["clearance", "rtol", "max_step", "expected_exit", "actual_exit", "length_error"])}

The excursion extends only 1e-6 outside the box. The dense search detects its
first outward crossing even with unlimited maximum step. It isolates extrema
of the quartic continuous extension and brackets roots on the resulting
monotone intervals; endpoint-only sign changes are not required. A detected
crossing evaluates the continuous extension at its first outward root and
projects the terminal point onto the selected face.
Out-of-domain stages still trigger safe step shortening. No interpolated
outside point is saved and field data are not extrapolated.

Event-time accuracy remains limited by trajectory accuracy. At this shallow
crossing, |dx/ds| is about 0.002, so a small position error is amplified when
converted to an arc-length error. Event-length
error varies with integration settings. Accordingly its regression
checks the first-exit code and a tolerance-scaled event-length budget,
rather than imposing 1e-7 accuracy regardless of the requested integration
tolerance. `boundary_tol` controls terminal/retry resolution, not global
trajectory or event-location error.

For an exact touch, the squared transverse displacement is sensitive to tiny
normal-position error. Small normal displacements can make the numerical trajectory exit close
to the mathematical contact. The test
explicitly distinguishes exact contacts from positive-clearance near misses:
a contact may complete or return a face exit within a tolerance-scaled
neighborhood of the analytic contact. It does not claim preservation of
contact topology. Tightening tolerances is useful but is not a proof of
classification at a mathematically exact tangency.

All 17 configurations are retained in `boundaries.csv`.

## 4. Additional regression cases

`tests/test_scientific_validation.py` also checks simultaneous corner exits
and their deterministic x-first code, an outward/inward boundary seed,
multiple periodic crossings on all three axes, continuous travel tangent to
a face, approach to a magnetic null with a weak-field threshold, and both
branches against the independent cubic references. `tests/test_events.py`
adds narrow polynomial excursions on both faces, true polynomial contacts,
quartic exit/re-entry, stationary points and first roots against independent
NumPy polynomial solvers, and the continuous extension against SciPy dense
output. End-to-end excursions cover all coordinate axes, both faces and both
branches, including sparse output strides. Periodic axes are excluded from
the event search. No expected-failure marker remains.

## Implementation limits and next validation

The event search is applied to error-accepted quartic trajectories. Excursions
smaller than floating-point noise (64 machine eps times a coordinate and
coefficient magnitude scale) are not classified as crossings. Neither the
continuous extension nor the embedded estimate bounds errors in the sampled
field. The root search uses bracketed scalar operations, with no SciPy or
complex-eigenvalue solver dependency at runtime. Terminal face output is
projected exactly; its accuracy remains limited by the numerical trajectory.

Further release validation should cover longer trajectories and real datasets,
near-degenerate events at other scales, and performance costs of the new
per-step search. The exact-contact sensitivity above must stay documented.
The resolved missed-exit regression no longer requires a user-selected small
maximum step to pass.

The study does not establish behavior on real datasets, matched-accuracy
performance/memory costs, or diagnostic/winding convergence. Those remain
open validation tasks.
"""
    (output / "REPORT.md").write_text(report)
    print(
        f"Wrote {output / 'REPORT.md'}: {len(convergence)} convergence cases, "
        f"{len(boundaries)} boundary cases, {len(missed_crossings)} missed crossings, "
        f"{len(contact_exits)} accuracy-sensitive exact-contact face exits."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "boundaries",
    )
    generate_report(parser.parse_args().output_dir)
