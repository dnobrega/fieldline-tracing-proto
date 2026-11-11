"""Analytic circular field: compare accuracy and warm RK2/RK45 timing.

Run from the project root:
    .venv/bin/python examples/compare_integrators.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np

from fieldline_tracing_proto.tracing import trace_field_lines


def main():
    grid = np.linspace(-3.0, 3.0, 17)
    x, y, z = np.meshgrid(grid, grid, grid, indexing="ij")
    seeds = np.array([[1.0, 0.0, 0.0]])
    cases = [
        ("RK2", dict(ds=0.01, max_steps=200)),
        (
            "RK45",
            dict(
                ds=0.8,
                max_steps=1000,
                method="rk45",
                max_length=2.0,
                rtol=1e-9,
                atol=1e-11,
            ),
        ),
    ]
    for name, controls in cases:

        def solve():
            return trace_field_lines(
                -y,
                x,
                np.zeros_like(x),
                grid,
                grid,
                grid,
                seeds,
                dtype=np.float64,
                periodic_x=False,
                periodic_y=False,
                **controls,
                periodic_convention="endpoint_included",
            )

        solve()  # Compile/warm kernels before timing.
        start = time.perf_counter()
        result = solve()
        elapsed = time.perf_counter() - start
        length = float(result.length_plus[0])
        error = np.hypot(
            result.xfl[-1] - np.cos(length), result.yfl[-1] - np.sin(length)
        )
        print(
            f"{name}: endpoint error={error:.3e}, accepted={result.nsteps_plus[0]}, "
            f"rejected={result.nrejected_plus[0]}, warm time={elapsed:.4f}s"
        )
    print("Timings are illustrative: one seed and different accuracy targets.")


if __name__ == "__main__":
    main()
