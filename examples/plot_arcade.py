"""Trace a potential magnetic arcade on a grid with nonuniform z spacing."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fieldline_tracing_proto import trace_field_lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("arcade.png"))
    args = parser.parse_args()

    x = np.linspace(-1.5, 1.5, 151)
    y = np.array([-0.5, 0.5])
    z = 1.35 * np.linspace(0.0, 1.0, 101) ** 1.4
    X, _, Z = np.meshgrid(x, y, z, indexing="ij")
    decay = np.exp(-Z)
    bx = np.cos(X) * decay
    bz = -np.sin(X) * decay
    heights = np.linspace(0.05, 1.15, 9)
    seeds = np.column_stack((np.zeros_like(heights), np.zeros_like(heights), heights))
    result = trace_field_lines(
        bx,
        np.zeros_like(bx),
        bz,
        x,
        y,
        z,
        seeds,
        ds=0.01,
        max_steps=4000,
        method="rk45",
        rtol=1e-7,
        periodic_x=False,
        periodic_y=False,
        periodic_z=False,
    )
    if np.any(result.exit_code_minus != -1) or np.any(result.exit_code_plus != 1):
        raise RuntimeError("Expected both branches of every loop to reach zmin.")

    fig, ax = plt.subplots(figsize=(7.2, 3.6), layout="constrained")
    for i, color in enumerate(plt.cm.viridis(np.linspace(0.18, 0.8, len(seeds)))):
        xline, _, zline, _ = result.get_line(i)
        ax.plot(xline, zline, color=color, linewidth=1.8)
    ax.scatter(seeds[:, 0], seeds[:, 2], s=15, color="#1e293b", zorder=3, label="Seeds")
    ax.axhline(0, color="#64748b", linewidth=0.9)
    ax.set(xlabel="x [dimensionless]", ylabel="z [dimensionless]", ylim=(-0.04, 1.32))
    ax.set_aspect("equal")
    ax.set_title(
        "Magnetic field lines in a potential arcade", loc="left", weight="bold"
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(args.output, dpi=170, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
