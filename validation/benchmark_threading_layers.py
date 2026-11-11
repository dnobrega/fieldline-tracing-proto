"""Run the same Olshevsky workload under explicit OpenMP/TBB backends.

No global thread-layer settings are changed. Each backend has fresh workers.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def compare(out, layers):
    by_layer = {}
    for layer in layers:
        rows = {}
        for file in (out / layer).glob("n*.json"):
            document = json.loads(file.read_text())
            assert document["environment"]["threading_layer"] == layer
            for row in document["cases"]:
                key = (
                    row["n"],
                    row["dtype"],
                    row["order"],
                    row["population"],
                    row["method"],
                    row["threads"],
                )
                rows[key] = row
        by_layer[layer] = rows
    baseline = by_layer[layers[0]]
    for layer in layers[1:]:
        assert baseline.keys() == by_layer[layer].keys()
        for key, original in baseline.items():
            row = by_layer[layer][key]
            assert original["result_sha256"] == row["result_sha256"], (
                key,
                layer,
                "numerical output differs",
            )
            assert original["exit_histogram"] == row["exit_histogram"]
    lines = [
        "# Explicit threading-layer comparison",
        "",
        "Each row has the same field/seeds/controls across backends. Output and",
        "scientific-counter hashes must match exactly. Warm solve times only;",
        "no speed ranking is required. Host load and scheduling affect timings.",
        "See each layer directory for source hashes, settings and first calls.",
        "",
        "| N | Population | Method | Threads | "
        + " | ".join(layer + " seconds" for layer in layers)
        + " |",
        "| --- | --- | --- | --- | " + " | ".join("---" for layer in layers) + " |",
    ]
    for key in sorted(baseline):
        n, _dtype, _order, population, method, threads = key
        times = " | ".join(
            f"{by_layer[layer][key]['warm_median_seconds']:.6f}" for layer in layers
        )
        lines.append(f"| {n} | {population} | {method} | {threads} | {times} |")
    problems = [
        {
            "layer": layer,
            "n": key[0],
            "population": key[3],
            "method": key[4],
            "threads": key[5],
            "branches": row["nonstandard_terminations"],
        }
        for layer, rows in by_layer.items()
        for key, row in rows.items()
        if row["nonstandard_terminations"]
    ]
    lines += [
        "",
        "Exact output/counter equality across requested backends: **passed**.",
        f"Cases containing nonstandard terminations: **{len(problems)}**.",
        "These are fixed-control synthetic workloads, not matched-error method comparisons.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))
    (out / "comparison.json").write_text(
        json.dumps(
            {
                "layers": layers,
                "cases": len(baseline),
                "exact_output_and_counter_equality": True,
                "nonstandard_terminations": problems,
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layers",
        nargs="+",
        choices=["omp", "tbb", "workqueue"],
        default=["omp", "tbb"],
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "validation/output/threading"
    )
    # All remaining controls are passed unchanged to the field benchmark.
    args, controls = parser.parse_known_args()
    if len(set(args.layers)) != len(args.layers):
        parser.error("Layer names must be distinct")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    for layer in args.layers:
        environment = os.environ.copy()
        environment["NUMBA_THREADING_LAYER"] = layer
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "validation/benchmark_olshevsky.py"),
                *controls,
                "--out",
                str(args.out / layer),
            ],
            env=environment,
            check=True,
        )
    compare(args.out, args.layers)
    print(args.out / "REPORT.md")


if __name__ == "__main__":
    main()
