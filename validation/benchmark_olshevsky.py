"""High-resolution, isolated-worker benchmark on the 14-null analytic field.

See validation/olshevsky/README.md. SciPy is a benchmark-only dependency.
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

os.environ.setdefault("NUMBA_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import numba
import numpy as np
from fieldline_tracing_proto import trace_field_lines, __version__
from validation.olshevsky.field import create_grid, olshevsky2020_field
from validation.olshevsky.reference import make_seeds, compare_reference, field_at


def peak_mib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024**2 if sys.platform == "darwin" else 1024)


def hashes():
    files = (
        list((ROOT / "src").rglob("*.py"))
        + list((ROOT / "validation/olshevsky").glob("*.py"))
        + [Path(__file__)]
    )
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(files)
    }


def worker(args):
    n, dtype, order = args.worker
    n = int(n)
    dtype = np.dtype(dtype)
    max_threads = numba.config.NUMBA_NUM_THREADS
    if max(args.threads) > max_threads:
        raise ValueError(f"Requested threads exceed NUMBA_NUM_THREADS={max_threads}")
    numba.set_num_threads(max(args.threads))
    field_gib = 3 * n**3 * dtype.itemsize / 1024**3
    if field_gib > args.max_field_gib:
        raise ValueError(f"Field requires {field_gib:.3f} GiB, above --max-field-gib")
    baseline_peak = peak_mib()
    grid = create_grid(n)
    h = float(grid[1] - grid[0])
    general, near, roots = make_seeds(args.seeds, args.near_per_null, args.radius)
    populations = [("general", general), ("near_null", near)]
    ids = np.array([0, n // 3, n // 2, n - 1])
    # Compile field generation separately; no full-size conversion or meshgrid.
    t = time.perf_counter()
    olshevsky2020_field(
        grid[:2],
        grid[:2],
        grid[:2],
        dtype=dtype,
        out=tuple(np.empty((2, 2, 2), dtype=dtype, order=order) for _ in range(3)),
    )
    generator_first = time.perf_counter() - t
    t = time.perf_counter()
    fields = tuple(np.empty((n, n, n), dtype=dtype, order=order) for _ in range(3))
    olshevsky2020_field(grid, grid, grid, out=fields)
    generation = time.perf_counter() - t
    for i in ids:
        p = np.array([grid[i], grid[(i + 3) % n], grid[(i + 5) % n]])
        got = np.array([a[i, (i + 3) % n, (i + 5) % n] for a in fields])
        np.testing.assert_allclose(
            got,
            field_at(p),
            rtol=2e-6 if dtype == np.dtype("float32") else 1e-12,
            atol=2e-6 if dtype == np.dtype("float32") else 1e-12,
        )
    generation_peak = peak_mib()
    rows = []
    # Deterministic interleaving reduces fixed method-order bias.
    cases = [
        (label, seeds, method, threads)
        for label, seeds in populations
        for method in args.methods
        for threads in args.threads
    ]
    np.random.default_rng(14).shuffle(cases)
    for label, seeds, method, threads in cases:
        numba.set_num_threads(threads)
        ds = args.ds if args.ds is not None else 0.2 * h
        kwargs = dict(
            ds=ds,
            max_steps=args.max_steps,
            max_length=args.length,
            method=method,
            dtype=dtype,
            save_stride=args.save_stride,
            periodic_x=False,
            periodic_y=False,
            periodic_z=False,
            storage=args.storage,
            memory_limit_mb=args.memory_limit_mb,
            array_order="K",
            bmin=1e-12,
        )
        if method == "rk45":
            kwargs.update(
                rtol=args.rtol, atol=args.atol, max_step=h * args.max_step_cells
            )
        if args.buffer_points is not None:
            kwargs["buffer_points"] = args.buffer_points

        def solve():
            return trace_field_lines(*fields, grid, grid, grid, seeds, **kwargs)

        t = time.perf_counter()
        result = solve()
        first = time.perf_counter() - t
        expected_codes = np.stack(
            (result.exit_code_minus, result.exit_code_plus)
        ).copy()
        expected_steps = np.stack((result.nsteps_minus, result.nsteps_plus)).copy()
        expected_lengths = np.stack((result.length_minus, result.length_plus)).copy()
        expected_points = result.total_points
        # Do not retain a previous full output during a measured solve.
        del result
        samples = []
        for _ in range(args.repeats):
            gc.collect()
            t = time.perf_counter()
            result = solve()
            samples.append(time.perf_counter() - t)
            np.testing.assert_array_equal(
                np.stack((result.exit_code_minus, result.exit_code_plus)),
                expected_codes,
            )
            np.testing.assert_array_equal(
                np.stack((result.nsteps_minus, result.nsteps_plus)), expected_steps
            )
            np.testing.assert_array_equal(
                np.stack((result.length_minus, result.length_plus)), expected_lengths
            )
            assert result.total_points == expected_points
            if not all(
                np.isfinite(a).all()
                for a in (result.xfl, result.yfl, result.zfl, result.sfl)
            ):
                raise AssertionError("Nonfinite output")
            if _ < args.repeats - 1:
                del result
        # This is the cumulative worker peak, not incremental solver memory.
        solve_peak = peak_mib()
        settings = json.loads(result.settings_json)
        codes, counts = np.unique(expected_codes, return_counts=True)
        row = dict(
            population=label,
            n=n,
            dtype=dtype.name,
            order=order,
            method=method,
            threads=threads,
            seeds=len(seeds),
            ds=ds,
            max_step=h * args.max_step_cells if method == "rk45" else None,
            first_seconds=first,
            warm_samples_seconds=samples,
            warm_median_seconds=float(np.median(samples)),
            points=result.total_points,
            accepted_steps=int(expected_steps.sum()),
            nfev=int(result.nfev_minus.sum() + result.nfev_plus.sum()),
            rejected_steps=int(
                result.nrejected_minus.sum() + result.nrejected_plus.sum()
            ),
            exit_histogram={str(int(c)): int(k) for c, k in zip(codes, counts)},
            max_steps_branches=int(np.sum(np.abs(expected_codes) == 7)),
            result_sha256={
                name: hashlib.sha256(getattr(result, name).tobytes()).hexdigest()
                for name in (
                    "xfl",
                    "yfl",
                    "zfl",
                    "sfl",
                    "line_start",
                    "line_count",
                    "exit_code_minus",
                    "exit_code_plus",
                    "length_minus",
                    "length_plus",
                    "nsteps_minus",
                    "nsteps_plus",
                    "nfev_minus",
                    "nfev_plus",
                    "nrejected_minus",
                    "nrejected_plus",
                    "endpoints_minus",
                    "endpoints_plus",
                    "exit_faces_minus",
                    "exit_faces_plus",
                )
            },
            nonstandard_terminations=[
                dict(
                    seed_index=int(i),
                    direction=direction,
                    code=int(codes[i]),
                    endpoint=getattr(result, "endpoints_" + direction)[i].tolist(),
                )
                for direction in ("minus", "plus")
                for codes in (getattr(result, "exit_code_" + direction),)
                for i in np.flatnonzero((np.abs(codes) >= 7) & (np.abs(codes) != 11))
            ],
            output_array_mib=sum(
                a.nbytes for a in vars(result).values() if isinstance(a, np.ndarray)
            )
            / 1024**2,
            worker_peak_through_solve_mib=solve_peak,
            **{
                k: settings.get(k)
                for k in (
                    "work_nfev",
                    "storage",
                    "overflow_retraced",
                    "peak_branch_buffer_bytes",
                    "buffer_capacity_peak",
                )
            },
        )
        if args.reference != "none":
            # For near-null seeds, cover distinct null neighborhoods rather than
            # taking only seeds near the first root.
            refids = np.linspace(
                0, len(seeds) - 1, min(args.reference_seeds, len(seeds)), dtype=int
            )
            t = time.perf_counter()
            row["reference"] = compare_reference(
                result,
                (grid, grid, grid),
                fields,
                refids,
                args.reference,
                args.reference_max_step,
            )
            row["reference_seconds"] = time.perf_counter() - t
        rows.append(row)
        print(
            f"{n} {dtype.name} {order} {label} {method} t={threads}: {np.median(samples):.5f}s, {result.total_points} points",
            flush=True,
        )
        del result
    info = dict(
        version=__version__,
        python=platform.python_version(),
        numpy=np.__version__,
        numba=numba.__version__,
        platform=platform.platform(),
        logical_cpus=os.cpu_count(),
        threading_layer=numba.threading_layer(),
        numba_cache_dir=os.environ.get("NUMBA_CACHE_DIR"),
        field_gib=field_gib,
        baseline_worker_peak_mib=baseline_peak,
        generation_peak_mib=generation_peak,
        worker_peak_including_reference_mib=peak_mib(),
        generator_first_seconds=generator_first,
        field_allocation_and_generation_seconds=generation,
        seeds_sha256={
            label: hashlib.sha256(s.tobytes()).hexdigest() for label, s in populations
        },
        analytic_roots=roots.tolist(),
        analytic_root_residuals=[float(np.linalg.norm(field_at(p))) for p in roots],
        configuration={
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
            if k not in ("worker", "worker_json")
        },
        source_sha256=hashes(),
    )
    return dict(environment=info, cases=rows)


def report(out, documents):
    rows = [r for d in documents for r in d["cases"]]
    lines = [
        "# Olshevsky 14-null field: tracing benchmark",
        "",
        "Synthetic workload; see ../olshevsky/README.md for methodology and limits.",
        "Warm medians exclude generation, first calls and reference integrations.",
        "These controls are not calibrated to equal error budgets. No speed ranking is asserted.",
        "",
        "| N | dtype/order | Population | Method | Threads | Seeds | Median s | Points |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in sorted(
        rows,
        key=lambda r: (
            r["n"],
            r["dtype"],
            r["order"],
            r["population"],
            r["threads"],
            r["method"],
        ),
    ):
        lines.append(
            f"| {r['n']} | {r['dtype']}/{r['order']} | {r['population']} | {r['method']} | {r['threads']} | {r['seeds']} | {r['warm_median_seconds']:.6f} | {r['points']} |"
        )
    lines += [
        "",
        "## Reference errors",
        "",
        "Maximum Euclidean position difference at saved arc positions, on selected seeds.",
        "Analytic reference combines interpolation and integration effects; sampled-field reference",
        "estimates integration effects. Neither is an exact trajectory or global error bound.",
        "",
        "| N | dtype/order | Population | Method | Threads | Reference | Selected-seed max error |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        for kind in ("analytic", "sampled"):
            values = [
                v["max_position_error"]
                for v in r.get("reference", [])
                if v["reference"] == kind
            ]
            if values:
                lines.append(
                    f"| {r['n']} | {r['dtype']}/{r['order']} | {r['population']} | {r['method']} | {r['threads']} | {kind} | {max(values):.6e} |"
                )
    lines += [
        "",
        "Raw JSON records controls, source hashes, seed hashes, first calls, evaluations,",
        "rejections, exit histograms, overflow work, retained output bytes and process peaks.",
        "Process peaks are cumulative per worker; later cases can inherit earlier peaks.",
        "They include imports/JIT/allocator retention and cannot isolate solver allocation.",
        "The worker is isolated per grid/dtype/layout. References may affect peaks of later cases.",
        "See generation_peak_mib and worker_peak_including_reference_mib separately.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", nargs="+", type=int, default=[64, 128, 256])
    p.add_argument(
        "--dtypes", nargs="+", choices=["float32", "float64"], default=["float32"]
    )
    p.add_argument("--orders", nargs="+", choices=["C", "F"], default=["F"])
    p.add_argument("--threads", nargs="+", type=int, default=[2])
    p.add_argument(
        "--methods", nargs="+", choices=["rk2", "rk45"], default=["rk2", "rk45"]
    )
    p.add_argument("--seeds", type=int, default=128)
    p.add_argument("--near-per-null", type=int, default=8)
    p.add_argument("--radius", type=float, default=0.02)
    p.add_argument("--length", type=float, default=1.0)
    p.add_argument("--ds", type=float)
    p.add_argument("--max-step-cells", type=float, default=1.0)
    p.add_argument("--rtol", type=float, default=1e-7)
    p.add_argument("--atol", type=float, default=1e-10)
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--save-stride", type=int, default=1)
    p.add_argument(
        "--storage", choices=["auto", "single_pass", "two_pass"], default="auto"
    )
    p.add_argument("--memory-limit-mb", type=float, default=64.0)
    p.add_argument("--buffer-points", type=int)
    p.add_argument("--max-field-gib", type=float, default=4.0)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument(
        "--reference", choices=["none", "analytic", "both"], default="analytic"
    )
    p.add_argument("--reference-seeds", type=int, default=2)
    p.add_argument("--reference-max-step", type=float, default=0.01)
    p.add_argument("--out", type=Path, default=ROOT / "validation/output/olshevsky")
    p.add_argument("--worker", nargs=3, help=argparse.SUPPRESS)
    p.add_argument("--worker-json", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    positive = [
        *args.n,
        *args.threads,
        args.seeds,
        args.near_per_null,
        args.radius,
        args.length,
        args.max_step_cells,
        args.rtol,
        args.atol,
        args.max_steps,
        args.save_stride,
        args.memory_limit_mb,
        args.max_field_gib,
        args.repeats,
        args.reference_seeds,
        args.reference_max_step,
    ]
    if any(not np.isfinite(v) or v <= 0 for v in positive) or min(args.n) < 4:
        p.error("Sizes/counts/controls must be positive and finite; N >= 4")
    if args.ds is not None and (not np.isfinite(args.ds) or args.ds <= 0):
        p.error("--ds must be finite and positive")
    if args.worker:
        args.worker_json.write_text(json.dumps(worker(args), indent=2) + "\n")
        return
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    documents = []
    original = sys.argv[1:]
    for n in args.n:
        for dtype in args.dtypes:
            for order in args.orders:
                destination = args.out / f"n{n}_{dtype}_{order}.json"
                if destination.exists():
                    raise FileExistsError(
                        f"Refusing to overwrite {destination}; use a new --out directory"
                    )
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    *original,
                    "--worker",
                    str(n),
                    dtype,
                    order,
                    "--worker-json",
                    str(destination),
                ]
                subprocess.run(command, check=True)
                documents.append(json.loads(destination.read_text()))
                report(args.out, documents)
    print(f"Report: {args.out / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
