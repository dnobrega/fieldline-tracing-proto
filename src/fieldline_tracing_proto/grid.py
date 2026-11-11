"""Native XYZ rectilinear grids, periodic seam interpolation and safe lookup."""

import numpy as np
from numba import njit


class PeriodicInferenceWarning(UserWarning):
    """Physical periodicity was inferred rather than fully specified."""


EXIT_ZMIN = 1
EXIT_ZMAX = 2
EXIT_YMIN = 3
EXIT_YMAX = 4
EXIT_XMIN = 5
EXIT_XMAX = 6
EXIT_MAX_STEPS = 7
EXIT_BNORM_TOO_SMALL = 8
EXIT_NAN = 9
EXIT_INTERP_ERROR = 10
EXIT_MAX_LENGTH = 11
EXIT_STEP_UNDERFLOW = 12
EXIT_MAX_ATTEMPTS = 13


def is_uniform_grid(grid, rtol=None, atol=0.0):
    """Recognize an affine lattice at its original coordinate precision."""
    raw = np.asarray(grid)
    if raw.size < 3:
        return True
    g = raw.astype(np.float64)
    if not np.all(np.isfinite(g)) or not np.all(np.diff(g) > 0):
        return False
    h = (g[-1] - g[0]) / (g.size - 1)
    eps = np.finfo(raw.dtype).eps if raw.dtype.kind == "f" else np.finfo(float).eps
    tolerance = max(
        atol,
        (1e-6 if rtol is None else rtol) * abs(h),
        4 * eps * max(abs(g[0]), abs(g[-1]), abs(h)),
    )
    tolerance = min(tolerance, 0.25 * np.min(np.diff(g)))
    return bool(np.max(np.abs(g - np.linspace(g[0], g[-1], g.size))) <= tolerance)


_FLOAT64_DTYPE = np.dtype(np.float64)


def _inv_spacing(grid, dtype=_FLOAT64_DTYPE):
    return dtype.type((len(grid) - 1) / (float(grid[-1]) - float(grid[0])))


def validate_cartesian_grid(shape, xgrid, ygrid, zgrid):
    for name, g, size in zip(("xgrid", "ygrid", "zgrid"), (xgrid, ygrid, zgrid), shape):
        if g.ndim != 1 or g.size != size:
            raise ValueError(f"{name} length/shape must match (nx,ny,nz).")
        if g.size < 2:
            raise ValueError(f"{name} must contain at least two nodes.")
        if not np.all(np.isfinite(g)):
            raise ValueError(f"{name} contains NaN/Inf.")
        if not np.all(np.diff(g) > 0):
            raise ValueError(f"{name} must be strictly increasing.")


def prepare_periods(grids, flags, period=None, convention="endpoint_excluded"):
    """Uniform endpoint-excluded axes infer n*dx; nonuniform axes require P."""
    if convention not in ("endpoint_excluded", "endpoint_included"):
        raise ValueError("Unknown periodic_convention.")
    supplied = (None,) * 3 if period is None else tuple(period)
    if len(supplied) != 3:
        raise ValueError("period must have three entries (Px,Py,Pz).")
    out = []
    for g, active, p in zip(grids, flags, supplied):
        if not active:
            if p is not None and p != 0:
                raise ValueError("A period requires a periodic axis.")
            out.append(0.0)
            continue
        span = float(g[-1]) - float(g[0])
        if p is None:
            if convention == "endpoint_included":
                p = span
            else:
                if not is_uniform_grid(g):
                    raise ValueError(
                        "Nonuniform periodic axes require an explicit physical period."
                    )
                p = span * len(g) / (len(g) - 1)
        p = float(p)
        if not np.isfinite(p) or p <= 0 or p < span:
            raise ValueError(
                "Period must be finite, positive and at least the node span."
            )
        out.append(p)
    return tuple(out)


def prepare_array(a, dtype, order="K"):
    """Keep C, Fortran or strided layout; copy only for explicit order/dtype."""
    if order not in ("K", "C", "F"):
        raise ValueError("array_order must be 'K', 'C' or 'F'.")
    return np.asarray(a, dtype=dtype, order=order)


def lookup_table(g, uniform):
    if uniform:
        return np.empty(0, dtype=np.int64)
    q = np.linspace(g[0], g[-1], max(2, 2 * len(g)))
    return np.clip(np.searchsorted(g, q, side="right") - 1, 0, len(g) - 2).astype(
        np.int64
    )


@njit(cache=True, forceinline=True)
def _wrap_coord(q, qmin, qmax, period=0.0):
    if not np.isfinite(q):
        return np.nan
    if period > 0:
        if qmin <= q < qmin + period:
            # Keep the previous canonicalization's rounding without fmod.
            offset = q - qmin
            if offset >= period:
                return qmin
            value = qmin + offset
            return qmin if value >= qmin + period else value
        value = qmin + (q - qmin) % period
        return qmin if value >= qmin + period else value
    L = qmax - qmin
    if L <= 0:
        return q
    if q < qmin:
        return qmin + (q - qmin) % L
    if q > qmax:
        return qmax - (qmax - q) % L
    return q


@njit(cache=True)
def _exit_code_for_coord(q, qmin, qmax, periodic, offset):
    if periodic:
        return 0
    if q < qmin:
        return offset + 1
    if q > qmax:
        return offset + 2
    return 0


@njit(cache=True)
def _find_index_1d(grid, q, is_uniform, inv_d):
    n = grid.size
    if n < 2 or q < grid[0] or q > grid[-1]:
        return -1
    if q == grid[-1]:
        return n - 2
    if not is_uniform:
        return max(0, np.searchsorted(grid, q, side="right") - 1)
    i = min(n - 2, max(0, int((q - grid[0]) * inv_d)))
    # Rounded float32 lattices need correction against the actual nodes.
    while i > 0 and q < grid[i]:
        i -= 1
    while i < n - 2 and q >= grid[i + 1]:
        i += 1
    return i


@njit(cache=True, forceinline=True)
def _uniform_axis_cell(g, q, active, inverse, period):
    if active:
        q = _wrap_coord(q, g[0], g[-1], period)
    elif q < g[0] or q > g[-1]:
        return -1, 0.0
    if active and period > g[-1] - g[0] and q >= g[-1]:
        return g.size - 1, (q - g[-1]) / (g[0] + period - g[-1])
    i = min(g.size - 2, max(0, int((q - g[0]) * inverse)))
    while i > 0 and q < g[i]:
        i -= 1
    while i < g.size - 2 and q >= g[i + 1]:
        i += 1
    return i, (q - g[i]) / (g[i + 1] - g[i])


@njit(cache=True)
def _axis_cell(g, q, active, uniform, inverse, period, table):
    if active:
        q = _wrap_coord(q, g[0], g[-1], period)
    elif q < g[0] or q > g[-1]:
        return -1, 0.0
    if active and period > g[-1] - g[0] and q >= g[-1]:
        return g.size - 1, (q - g[-1]) / (g[0] + period - g[-1])
    if not uniform and table.size > 0:
        b = min(
            table.size - 1, max(0, int((q - g[0]) / (g[-1] - g[0]) * (table.size - 1)))
        )
        i = table[b]
        steps = 0
        while i > 0 and q < g[i] and steps < 8:
            i -= 1
            steps += 1
        while i < g.size - 2 and q >= g[i + 1] and steps < 8:
            i += 1
            steps += 1
        if steps == 8:
            i = _find_index_1d(g, q, False, 0.0)
    else:
        i = _find_index_1d(g, q, uniform, inverse)
    if i < 0:
        return -1, 0.0
    return i, (q - g[i]) / (g[i + 1] - g[i])


_EMPTY = np.empty(0, dtype=np.int64)


@njit(cache=True, forceinline=True)
def _interpolation_stencil(
    xgrid,
    ygrid,
    zgrid,
    x,
    y,
    z,
    periodic_x,
    periodic_y,
    periodic_z,
    uniform_x,
    uniform_y,
    uniform_z,
    inv_dx,
    inv_dy,
    inv_dz,
    periods=(0.0, 0.0, 0.0),
    tables=(_EMPTY, _EMPTY, _EMPTY),
):
    if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
        return -1, -1, -1, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), False, EXIT_NAN
    if not periodic_x:
        if x < xgrid[0]:
            return (
                -1,
                -1,
                -1,
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                False,
                EXIT_XMIN,
            )
        if x > xgrid[-1]:
            return (
                -1,
                -1,
                -1,
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                False,
                EXIT_XMAX,
            )
    if not periodic_y:
        if y < ygrid[0]:
            return (
                -1,
                -1,
                -1,
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                False,
                EXIT_YMIN,
            )
        if y > ygrid[-1]:
            return (
                -1,
                -1,
                -1,
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                False,
                EXIT_YMAX,
            )
    if not periodic_z:
        if z < zgrid[0]:
            return (
                -1,
                -1,
                -1,
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                False,
                EXIT_ZMIN,
            )
        if z > zgrid[-1]:
            return (
                -1,
                -1,
                -1,
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                False,
                EXIT_ZMAX,
            )
    if uniform_x:
        ix, tx = _uniform_axis_cell(xgrid, x, periodic_x, inv_dx, periods[0])
    else:
        ix, tx = _axis_cell(xgrid, x, periodic_x, False, inv_dx, periods[0], tables[0])
    if uniform_y:
        iy, ty = _uniform_axis_cell(ygrid, y, periodic_y, inv_dy, periods[1])
    else:
        iy, ty = _axis_cell(ygrid, y, periodic_y, False, inv_dy, periods[1], tables[1])
    if uniform_z:
        iz, tz = _uniform_axis_cell(zgrid, z, periodic_z, inv_dz, periods[2])
    else:
        iz, tz = _axis_cell(zgrid, z, periodic_z, False, inv_dz, periods[2], tables[2])
    if min(ix, iy, iz) < 0:
        return (
            -1,
            -1,
            -1,
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            False,
            EXIT_INTERP_ERROR,
        )
    return (
        ix,
        iy,
        iz,
        (
            (1 - tz) * (1 - ty) * (1 - tx),
            (1 - tz) * (1 - ty) * tx,
            (1 - tz) * ty * (1 - tx),
            (1 - tz) * ty * tx,
            tz * (1 - ty) * (1 - tx),
            tz * (1 - ty) * tx,
            tz * ty * (1 - tx),
            tz * ty * tx,
        ),
        True,
        0,
    )


@njit(cache=True, forceinline=True)
def _apply_indices(A, ix, iy, iz, jx, jy, jz, w):
    return (
        A[ix, iy, iz] * w[0]
        + A[jx, iy, iz] * w[1]
        + A[ix, jy, iz] * w[2]
        + A[jx, jy, iz] * w[3]
        + A[ix, iy, jz] * w[4]
        + A[jx, iy, jz] * w[5]
        + A[ix, jy, jz] * w[6]
        + A[jx, jy, jz] * w[7]
    )


@njit(cache=True, forceinline=True)
def _apply_stencil(A, ix, iy, iz, w):
    jx = ix + 1 if ix + 1 < A.shape[0] else 0
    jy = iy + 1 if iy + 1 < A.shape[1] else 0
    jz = iz + 1 if iz + 1 < A.shape[2] else 0
    return _apply_indices(A, ix, iy, iz, jx, jy, jz, w)
