"""XZ magnetic-field quick look near a null, using bifrost_proto.

Notebook usage:
    %run /absolute/path/to/examples/plot_bifrost_xz.py
    fig, ax = plot_bifrost_xz(dd, snap, res['pos'][0], half_width=3.)

The white curves are 2-D streamlines of (Bx,Bz) evaluated in the selected
plane. They are not projected or plane-confined 3-D RK field lines when
By != 0. The null marker is its x/z projection onto the nearest Y plane.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator


def plot_bifrost_xz(dd, snap, null, *, half_width=3.0, density=1.5):
    """Plot |B| and in-plane streamlines; coordinates/half_width are in Mm.

    Reuses an existing BifrostData object. half_width=None shows the full
    XZ extent; otherwise the data read is cropped around the null projection.
    Matplotlib/SciPy/bifrost_proto belong to the analysis environment, not
    the fieldline tracer's runtime dependencies.
    """
    null = np.asarray(null, dtype=float)
    if null.shape != (3,) or not np.all(np.isfinite(null)):
        raise ValueError("null must contain finite physical coordinates (x,y,z)")
    xall, yall, zall = (
        np.asarray(f(), dtype=float) for f in (dd.get_x, dd.get_y, dd.get_z)
    )
    yplane = float(yall[np.argmin(np.abs(yall - null[1]))])
    xsel = zsel = None
    if half_width is not None:
        if not np.isfinite(half_width) or half_width <= 0:
            raise ValueError("half_width must be positive or None")
        xsel = slice(
            max(xall[0], null[0] - half_width), min(xall[-1], null[0] + half_width)
        )
        zsel = slice(
            max(zall[0], null[2] - half_width), min(zall[-1], null[2] + half_width)
        )
    selection = dict(x=xsel, y=yplane, z=zsel)
    x = np.asarray(dd.get_x(x=xsel), dtype=float)
    z = np.asarray(dd.get_z(z=zsel, uniform_z=True), dtype=float)
    if x.size < 2 or z.size < 2:
        raise ValueError("The selected extent must include at least two X and Z nodes")
    bx = np.asarray(dd.get_var("b_x", snap, **selection, center=True, uniform_z=True))
    bz = np.asarray(dd.get_var("b_z", snap, **selection, center=True, uniform_z=True))
    if bx.shape != (len(x), len(z)) or bz.shape != bx.shape:
        raise ValueError("Expected Bifrost XZ slices in (nx,nz) order")

    # streamplot requires evenly spaced coordinates. Sample both components
    # on one uniform grid, also handling a potentially nonuniform X mesh.
    xu, zu = np.linspace(x[0], x[-1], len(x)), np.linspace(z[0], z[-1], len(z))
    X, Z = np.meshgrid(xu, zu, indexing="xy")
    positions = np.column_stack((X.ravel(), Z.ravel()))
    u, v = (
        RegularGridInterpolator((x, z), a)(positions).reshape(X.shape) for a in (bx, bz)
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    dd.plot("modb", snap, **selection, ax=ax, cmap="magma", norm=LogNorm(), title=False)
    ax.streamplot(
        xu, zu, u, v, density=density, color="white", linewidth=0.7, arrowsize=0.9
    )
    ax.scatter(
        null[0],
        null[2],
        marker="x",
        s=70,
        color="cyan",
        zorder=5,
        label="Null projection",
    )
    ax.set_title(f"XZ: y = {yplane:.5f} Mm (null y = {null[1]:.5f} Mm)")
    ax.set_xlabel("x [Mm]")
    ax.set_ylabel("z [Mm]")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    return fig, ax
