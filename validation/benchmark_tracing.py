"""Calibrate equal trajectory-error budgets, then benchmark in isolated workers.

Run: python validation/benchmark_tracing.py
No runtime dependency beyond the project's development environment is added.
"""

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import resource
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numba
import numpy as np

from fieldline_tracing_proto import __version__
from fieldline_tracing_proto.tracing import trace_field_lines
from validation.run_validation import format_table, write_csv

LENGTH = 2.0


def geometry(field, nseeds):
    grid = np.linspace(-3.0, 3.0, 17)
    x, y, z = np.meshgrid(grid, grid, grid, indexing="ij")
    phase = np.linspace(0.0, 2 * np.pi, nseeds, endpoint=False)
    seeds = np.column_stack([np.cos(phase), np.sin(phase), np.zeros(nseeds)])
    bz = np.zeros_like(x) if field == "circle" else np.full_like(x, 0.5)
    return (-y, x, bz, grid, grid, grid, seeds)


def solve(data, controls, periodic=False):
    return trace_field_lines(
        *data,
        dtype=np.float64,
        save_stride=1,
        periodic_x=periodic,
        periodic_y=periodic,
        periodic_z=periodic,
        **controls,
        periodic_convention="endpoint_included",
    )


def trajectory_error(result, field):
    """Maximum Euclidean error over *every saved point* and both branches."""
    pitch = 0.0 if field == "circle" else 0.5
    speed = np.sqrt(1 + pitch * pitch)
    largest = 0.0
    for i in range(result.nlines):
        x, y, z, s = result.get_line(i)
        signed_s = s - result.length_minus[i]
        phase = np.arctan2(result.seeds[i, 1], result.seeds[i, 0])
        theta = phase + signed_s / speed
        error = np.sqrt(
            (x - np.cos(theta)) ** 2
            + (y - np.sin(theta)) ** 2
            + (z - pitch * signed_s / speed) ** 2
        )
        largest = max(largest, float(error.max()))
    return largest


def calibrate(field, target, method):
    data = geometry(field, 16)
    if method == "rk2":
        candidates = [
            dict(ds=LENGTH / n, max_steps=n)
            for n in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
        ]
    else:
        candidates = [
            dict(
                method="rk45",
                ds=0.8,
                max_steps=20000,
                max_length=LENGTH,
                rtol=tol,
                atol=tol * 0.01,
            )
            for tol in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12)
        ]
    for controls in candidates:
        result = solve(data, controls)
        error = trajectory_error(result, field)
        if error <= 0.8 * target:
            return controls, error
        del result
    raise RuntimeError(f"No calibrated configuration for {field}/{method}/{target}")


def peak_rss_mib():
    # macOS ru_maxrss is bytes; Linux reports KiB.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024**2 if sys.platform == "darwin" else 1024)


def worker(case):
    data = geometry(case["field"], case["nseeds"])
    baseline_rss = peak_rss_mib()
    start = time.perf_counter()
    result = solve(data, case["controls"], case["periodic"])
    first_call = time.perf_counter() - start
    del result
    gc.collect()
    times = []
    for _ in range(case["repeats"]):
        start = time.perf_counter()
        result = solve(data, case["controls"], case["periodic"])
        times.append(time.perf_counter() - start)
        del result
        gc.collect()
    # Untimed verification keeps analytic-error calculation out of warm timing.
    result = solve(data, case["controls"], case["periodic"])
    peak_after_solve = peak_rss_mib()
    error = trajectory_error(result, case["field"])
    if error > case["target"]:
        raise RuntimeError(f"Actual benchmark failed its error budget: {error}")
    if not (
        np.allclose(result.length_minus, LENGTH, atol=1e-12, rtol=0)
        and np.allclose(result.length_plus, LENGTH, atol=1e-12, rtol=0)
    ):
        raise RuntimeError("Benchmark methods did not traverse equal lengths")
    persistent_bytes = sum(
        value.nbytes for value in vars(result).values() if isinstance(value, np.ndarray)
    )
    return dict(
        field=case["field"],
        target=case["target"],
        nseeds=case["nseeds"],
        method=case["method"],
        boundary_mode="periodic" if case["periodic"] else "bounded",
        ds=case["controls"]["ds"],
        rtol=case["controls"].get("rtol", "not-applicable"),
        max_steps=case["controls"]["max_steps"],
        save_stride=1,
        measured_max_error=error,
        saved_points=result.total_points,
        accepted_steps_total=int(
            np.sum(result.nsteps_minus) + np.sum(result.nsteps_plus)
        ),
        rejected_trials_total=int(
            np.sum(result.nrejected_minus) + np.sum(result.nrejected_plus)
        ),
        per_integration_nfev=int(np.sum(result.nfev_minus) + np.sum(result.nfev_plus)),
        work_nfev=json.loads(result.settings_json).get("work_nfev"),
        overflow_retraced=json.loads(result.settings_json).get("overflow_retraced"),
        first_call_seconds=first_call,
        repeats=case["repeats"],
        warm_median_seconds=float(np.median(times)),
        warm_min_seconds=min(times),
        warm_p25_seconds=float(np.percentile(times, 25)),
        warm_p75_seconds=float(np.percentile(times, 75)),
        output_array_mib=persistent_bytes / 1024**2,
        worker_peak_rss_before_first_call_mib=baseline_rss,
        worker_peak_rss_through_solves_mib=peak_after_solve,
        worker_peak_rss_including_verification_mib=peak_rss_mib(),
    )


def main(output, repeats, seed_counts):
    output.mkdir(parents=True, exist_ok=True)
    cases, calibration = [], []
    for field in ("circle", "helix"):
        for target in (1e-4, 1e-6, 1e-8):
            for method in ("rk2", "rk45"):
                controls, error = calibrate(field, target, method)
                calibration.append(
                    dict(
                        field=field,
                        target=target,
                        method=method,
                        calibration_max_error=error,
                        controls_json=json.dumps(controls, sort_keys=True),
                    )
                )
                for nseeds in seed_counts:
                    cases.append(
                        dict(
                            field=field,
                            target=target,
                            method=method,
                            nseeds=nseeds,
                            controls=controls,
                            periodic=False,
                            repeats=repeats,
                        )
                    )
                if method == "rk45" and target == 1e-6:
                    cases.append(
                        dict(
                            field=field,
                            target=target,
                            method=method,
                            nseeds=max(seed_counts),
                            controls=controls,
                            periodic=True,
                            repeats=repeats,
                        )
                    )
    # Reproducible interleaving reduces systematic method/order bias.
    random.Random(19).shuffle(cases)
    rows = []
    for i, case in enumerate(cases):
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-json",
                json.dumps(case),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        row = json.loads(completed.stdout)
        rows.append(row)
        print(
            f"[{i + 1}/{len(cases)}] {row['field']} {row['method']} error<={row['target']:.0e} "
            f"seeds={row['nseeds']} {row['boundary_mode']}: {row['warm_median_seconds']:.4f}s",
            flush=True,
        )
    rows.sort(
        key=lambda r: (
            r["field"],
            r["target"],
            r["nseeds"],
            r["method"],
            r["boundary_mode"],
        )
    )
    write_csv(output / "timings.csv", rows)
    write_csv(output / "calibration.csv", calibration)
    root = Path(__file__).resolve().parents[1]
    (output / "environment.json").write_text(
        json.dumps(
            dict(
                version=__version__,
                python=platform.python_version(),
                platform=platform.platform(),
                machine=platform.machine(),
                numpy=np.__version__,
                numba=numba.__version__,
                threads=numba.get_num_threads(),
                dtype="float64",
                field_shape=[17, 17, 17],
                branch_length=LENGTH,
                repeats=repeats,
                seed_counts=seed_counts,
                source_sha256={
                    str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [
                        Path(__file__).resolve(),
                        *sorted((root / "src").rglob("*.py")),
                    ]
                },
            ),
            indent=2,
        )
        + "\n"
    )
    pairs = []
    for field in ("circle", "helix"):
        for target in (1e-4, 1e-6, 1e-8):
            for nseeds in seed_counts:
                a = next(
                    r
                    for r in rows
                    if r["field"] == field
                    and r["target"] == target
                    and r["nseeds"] == nseeds
                    and r["method"] == "rk2"
                )
                b = next(
                    r
                    for r in rows
                    if r["field"] == field
                    and r["target"] == target
                    and r["nseeds"] == nseeds
                    and r["method"] == "rk45"
                    and r["boundary_mode"] == "bounded"
                )
                pairs.append(
                    dict(
                        field=field,
                        target=target,
                        nseeds=nseeds,
                        rk2_error=a["measured_max_error"],
                        rk45_error=b["measured_max_error"],
                        rk2_seconds=a["warm_median_seconds"],
                        rk45_seconds=b["warm_median_seconds"],
                        rk2_time_over_rk45=a["warm_median_seconds"]
                        / b["warm_median_seconds"],
                        rk2_output_mib=a["output_array_mib"],
                        rk45_output_mib=b["output_array_mib"],
                        rk2_worker_peak_rss_mib=a["worker_peak_rss_through_solves_mib"],
                        rk45_worker_peak_rss_mib=b[
                            "worker_peak_rss_through_solves_mib"
                        ],
                    )
                )
    write_csv(output / "comparisons.csv", pairs)
    overhead = []
    for periodic in (r for r in rows if r["boundary_mode"] == "periodic"):
        bounded = next(
            r
            for r in rows
            if r["field"] == periodic["field"]
            and r["method"] == "rk45"
            and r["target"] == periodic["target"]
            and r["nseeds"] == periodic["nseeds"]
            and r["boundary_mode"] == "bounded"
        )
        overhead.append(
            dict(
                field=periodic["field"],
                nseeds=periodic["nseeds"],
                bounded_seconds=bounded["warm_median_seconds"],
                periodic_seconds=periodic["warm_median_seconds"],
                bounded_over_periodic=bounded["warm_median_seconds"]
                / periodic["warm_median_seconds"],
            )
        )
    report = f"""# RK2/RK45 performance at matched trajectory-error budgets ({__version__})

Reproduce: `python validation/benchmark_tracing.py`.
Raw results: `timings.csv`, `comparisons.csv`, `calibration.csv` and
`environment.json`. Default: 32/128 seeds, two Numba threads, float64,
17³ grid, 2 coordinate units per branch, save_stride=1, {repeats} timed repetitions.

## Equal-error-budget comparison

Use affine, divergence-free fields `B=(-y,x,0)` and `B=(-y,x,0.5)` with
unit-radius seeds at evenly distributed phases. Trilinear interpolation is
exact for these fields, so analytic circles/helices isolate integration error.
Calibration picks the first tested configuration whose maximum Euclidean
error over every stored point, every seed and both branches is <=80% of the
budget (16 calibration seeds). Each timed case is then independently checked
against the full budget using its actual 32/128 seeds. Both methods traverse
the same length; comparisons are by equal *error budget*, not identical
measured errors or step counts. Candidate lists are discrete, not optimized
for either solver. See calibration.csv for settings and measured errors.

{format_table(pairs, ["field", "target", "nseeds", "rk2_seconds", "rk45_seconds", "rk2_time_over_rk45", "rk2_output_mib", "rk45_output_mib"])}

A ratio above one means RK45 was faster in that case. These local timings do
not establish a universal speed advantage on simulation data or nonsmooth
interpolated fields. Small workloads are particularly sensitive to overhead
and host load. Timing quantiles and minimums are retained in timings.csv.

## Timing and memory methodology

Each configuration runs in a separate worker process. One untimed first call
loads/compiles kernels and warms the solver. Warm timing includes input
validation, the selected storage strategy, allocation and result construction;
it excludes analytic verification, imports and garbage collection. First-call
seconds are reported separately, but may use Numba's disk cache, so they are
not a pure fresh-compilation benchmark. No speed assertion is enforced by tests.

`output_array_mib` is the exact sum of ndarray bytes retained in TraceResult;
it excludes strings, Python overhead, input fields and temporary allocations.
Saving every accepted point allows the methods to produce different point
counts; this output/memory difference is part of this workload, not a
comparison at equal output spacing or equal diagnostic accuracy.

`worker_peak_rss_through_solves_mib` is the whole worker's peak resident set
through warming and measured solves, before analytic verification. It includes
Python, imports, Numba/JIT/cache, inputs, allocator retention and solver work;
it is **not** the incremental solver peak. The baseline worker peak and the
peak including verification are also recorded. Absolute RSS and retained
output bytes are distinct measures and must not be subtracted to infer
Numba temporary memory. macOS/Linux ru_maxrss units are normalized to MiB.

## Boundary-search cost check

{format_table(overhead, ["field", "nseeds", "bounded_seconds", "periodic_seconds", "bounded_over_periodic"])}

Bounded and all-periodic cases use the same field, geometry, tolerances and
length, with trajectories staying inside the same box. Periodic axes skip the
dense boundary-event search, while bounded axes execute it. This paired check
also changes boundary/wrapping checks, so the ratio is not a pure isolated
cost of the polynomial root finder. It measures the effect of boundary mode
for these interior trajectories, not the cost of an actual boundary exit.

## Interpretation and limits

Choose integration controls from an accuracy requirement, then choose output
density from the intended diagnostics. The separate diagnostic study shows
why fewer accurate RK45 points need not give equally accurate scalar integrals
or winding. Matched diagnostic accuracy, nonuniform/noisy real fields,
float32, other thread counts, cold compilation and cross-machine performance
remain outside this benchmark. Input fields and output remain in memory;
no external dataset or disk I/O is timed.
"""
    (output / "REPORT.md").write_text(report)
    print(
        f"Wrote {output / 'REPORT.md'} with {len(rows)} isolated configurations.",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "tracing",
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed-counts", type=int, nargs="+", default=[32, 128])
    args = parser.parse_args()
    if args.worker_json:
        print(json.dumps(worker(json.loads(args.worker_json))))
    else:
        if args.repeats < 3 or not args.seed_counts or min(args.seed_counts) < 1:
            parser.error("Use at least 3 repetitions and positive seed counts")
        main(args.output_dir, args.repeats, sorted(set(args.seed_counts)))
