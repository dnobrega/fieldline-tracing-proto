# -------------------------------------------------------------------
# Sampling scalar fields on traced ragged field lines
# -------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from .grid import (
    EXIT_NAN,
    _apply_stencil,
    _interpolation_stencil,
    _inv_spacing,
    is_uniform_grid,
    lookup_table,
    prepare_array,
    prepare_periods,
    validate_cartesian_grid,
)
from .tracing import TraceResult


@dataclass
class ScalarSampleResult:
    """Scalar values sampled on an existing ragged TraceResult."""

    values_all: np.ndarray
    line_start: np.ndarray
    line_count: np.ndarray
    codes_all: np.ndarray

    @property
    def nlines(self) -> int:
        return self.line_count.size

    @property
    def total_points(self) -> int:
        return self.values_all.size

    def get_values(self, i: int) -> np.ndarray:
        """Return sampled scalar values for line i."""
        i0 = int(self.line_start[i])
        i1 = i0 + int(self.line_count[i])
        return self.values_all[i0:i1]

    def get_codes(self, i: int) -> np.ndarray:
        """Return sampling codes for line i."""
        i0 = int(self.line_start[i])
        i1 = i0 + int(self.line_count[i])
        return self.codes_all[i0:i1]


def validate_scalar_grid(
    scalar: np.ndarray,
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    zgrid: np.ndarray,
) -> None:
    """Validate scalar field shape and grid compatibility."""
    if scalar.ndim != 3:
        raise ValueError("scalar must have shape (nx, ny, nz).")

    validate_cartesian_grid(scalar.shape, xgrid, ygrid, zgrid)


def sample_scalar_on_lines(
    result: TraceResult,
    scalar,
    xgrid,
    ygrid,
    zgrid,
    *,
    periodic_x=None,
    periodic_y=None,
    periodic_z=None,
    period=None,
    periodic_convention=None,
    dtype=None,
    bad_value=np.nan,
    field_order="xyz",
    array_order="K",
    exact_anchors=True,
):
    """Sample a native XYZ scalar, inheriting trace flags, periods and dtype.

    Explicit flag/period changes warn because sampling another domain may be
    intentional. Positions and grids are always evaluated in float64.
    Legacy NPZ settings without explicit periods use repeated endpoints.
    """
    import json
    import warnings

    settings = json.loads(getattr(result, "settings_json", "{}"))
    flags = []
    for key, value, default in zip(
        ("periodic_x", "periodic_y", "periodic_z"),
        (periodic_x, periodic_y, periodic_z),
        (True, True, False),
    ):
        inherited = settings.get(key, default)
        if value is not None and key in settings and bool(value) != inherited:
            warnings.warn(
                f"{key} differs from trace metadata.", UserWarning, stacklevel=2
            )
        flags.append(inherited if value is None else bool(value))
    if dtype is None:
        dtype = np.asarray(result.xfl).dtype
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("dtype must be float32 or float64.")
    if period is None and "period" in settings:
        period = [
            (p if p > 0 else None) if flags[i] else 0.0
            for i, p in enumerate(settings["period"])
        ]
    elif period is not None and "period" in settings:
        if tuple(0.0 if p is None else p for p in period) != tuple(settings["period"]):
            warnings.warn(
                "period differs from trace metadata.", UserWarning, stacklevel=2
            )
    convention = periodic_convention or settings.get(
        "periodic_convention", "endpoint_included" if settings else "endpoint_excluded"
    )
    raw = tuple(np.asarray(g) for g in (xgrid, ygrid, zgrid))
    scalar = np.asarray(scalar)
    if field_order == "zyx":
        warnings.warn(
            "field_order='zyx' is a deprecated compatibility adapter.",
            DeprecationWarning,
            stacklevel=2,
        )
        scalar = scalar.transpose(2, 1, 0)
    elif field_order != "xyz":
        raise ValueError("Unknown field_order.")
    validate_scalar_grid(scalar, *raw)
    uniform = tuple(is_uniform_grid(g) for g in raw)
    grids = tuple(np.ascontiguousarray(g, dtype=np.float64) for g in raw)
    validate_scalar_grid(scalar, *grids)
    periods = prepare_periods(raw, flags, period, convention)
    tables = tuple(lookup_table(g, u) for g, u in zip(grids, uniform))
    scalar = prepare_array(
        scalar,
        scalar.dtype
        if scalar.dtype in (np.dtype("float32"), np.dtype("float64"))
        else dtype,
        array_order,
    )
    values = np.empty(len(result.xfl), dtype=dtype)
    codes = np.empty(len(result.xfl), dtype=np.int16)
    if not isinstance(exact_anchors, (bool, np.bool_)):
        raise ValueError("exact_anchors must be boolean.")
    seed_points = getattr(result, "seeds", None)
    seed_index = getattr(result, "seed_index", None)
    restore = bool(
        exact_anchors
        and np.asarray(result.xfl).dtype == np.dtype("float32")
        and seed_points is not None
        and seed_index is not None
        and len(values) > 0
    )
    coordinates = (result.xfl, result.yfl, result.zfl)
    points = [
        np.array(a, dtype=np.float64, copy=True, order="C")
        if restore
        else np.ascontiguousarray(a, dtype=np.float64)
        for a in coordinates
    ]
    if restore:
        starts = np.asarray(result.line_start, dtype=np.int64)
        counts = np.asarray(result.line_count, dtype=np.int64)
        indices = np.asarray(seed_index, dtype=np.int64)
        seeds = np.asarray(seed_points, dtype=np.float64)
        n = len(counts)
        if starts.shape != (n,) or indices.shape != (n,) or seeds.shape != (n, 3):
            raise ValueError("Invalid trace anchor metadata.")
        if (
            np.any(starts < 0)
            or np.any(counts < 1)
            or np.any(starts > len(values))
            or np.any(counts > len(values) - starts)
        ):
            raise ValueError("Invalid trace anchor layout.")
        if np.any(indices < starts) or np.any(indices >= starts + counts):
            raise ValueError("Invalid trace seed indices.")
        minus = getattr(result, "endpoints_minus", None)
        plus = getattr(result, "endpoints_plus", None)
        has_minus = minus is not None
        has_plus = plus is not None
        minus = np.asarray(minus, dtype=np.float64) if has_minus else np.empty((0, 3))
        plus = np.asarray(plus, dtype=np.float64) if has_plus else np.empty((0, 3))
        if (has_minus and minus.shape != (n, 3)) or (has_plus and plus.shape != (n, 3)):
            raise ValueError("Invalid trace endpoint metadata.")
        _restore_trace_anchors(
            *points, starts, counts, indices, seeds, minus, plus, has_minus, has_plus
        )
    inverse = tuple(float(_inv_spacing(g)) for g in grids)
    _sample_scalar_all_parallel(
        scalar,
        *grids,
        *points,
        values,
        codes,
        *flags,
        *uniform,
        *inverse,
        float(bad_value),
        np.asarray(periods),
        *tables,
    )
    return ScalarSampleResult(values, result.line_start, result.line_count, codes)


def sample_ragged_to_padded(
    sample: ScalarSampleResult,
    *,
    fill_value: float = np.nan,
    return_codes: bool = False,
):
    """
    Convert a ScalarSampleResult from ragged format to a padded matrix.

    Parameters
    ----------
    sample : ScalarSampleResult
        Output from sample_scalar_on_lines.
    fill_value : float
        Fill value for missing points.
    return_codes : bool
        If True, also return padded sampling-code matrix.

    Returns
    -------
    values_mat : ndarray
        Shape (nlines, max_points), filled with fill_value.
    codes_mat : ndarray, optional
        Shape (nlines, max_points), filled with -1 where no point exists.
    """

    nlines = sample.line_count.size
    max_points = int(sample.line_count.max()) if nlines > 0 else 0

    values_mat = np.full(
        (nlines, max_points),
        fill_value,
        dtype=sample.values_all.dtype,
    )

    if return_codes:
        codes_mat = np.full(
            (nlines, max_points),
            -1,
            dtype=sample.codes_all.dtype,
        )

    for i in range(nlines):
        i0 = int(sample.line_start[i])
        n = int(sample.line_count[i])
        i1 = i0 + n

        values_mat[i, :n] = sample.values_all[i0:i1]

        if return_codes:
            codes_mat[i, :n] = sample.codes_all[i0:i1]

    if return_codes:
        return values_mat, codes_mat

    return values_mat


from .grid import _EMPTY


@njit(cache=True)
def _interp_scalar_sampling(
    A,
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
    ix, iy, iz, w, ok, code = _interpolation_stencil(
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
        periods,
        tables,
    )
    if not ok:
        return 0.0, False, code
    value = _apply_stencil(A, ix, iy, iz, w)
    if not np.isfinite(value):
        return 0.0, False, EXIT_NAN
    return value, True, 0


@njit(parallel=True, cache=True)
def _sample_scalar_all_parallel(
    scalar,
    xgrid,
    ygrid,
    zgrid,
    xfl,
    yfl,
    zfl,
    values,
    codes,
    px,
    py,
    pz,
    ux,
    uy,
    uz,
    dx,
    dy,
    dz,
    bad,
    periods,
    tx,
    ty,
    tz,
):
    for i in prange(len(xfl)):
        value, ok, code = _interp_scalar_sampling(
            scalar,
            xgrid,
            ygrid,
            zgrid,
            xfl[i],
            yfl[i],
            zfl[i],
            px,
            py,
            pz,
            ux,
            uy,
            uz,
            dx,
            dy,
            dz,
            (periods[0], periods[1], periods[2]),
            (tx, ty, tz),
        )
        values[i] = value if ok else bad
        codes[i] = code


@njit(cache=True, forceinline=True)
def _restore_anchor(x, y, z, j, a, b, c):
    # Only undo the tracer's storage rounding; never override edited geometry.
    if (
        x[j] == np.float64(np.float32(a))
        and y[j] == np.float64(np.float32(b))
        and z[j] == np.float64(np.float32(c))
    ):
        x[j] = a
        y[j] = b
        z[j] = c


@njit(cache=True)
def _restore_trace_anchors(
    x, y, z, starts, counts, indices, seeds, minus, plus, has_minus, has_plus
):
    for i in range(len(counts)):
        j = indices[i]
        lo = starts[i]
        hi = lo + counts[i] - 1
        _restore_anchor(x, y, z, j, seeds[i, 0], seeds[i, 1], seeds[i, 2])
        if has_minus and lo < j:
            _restore_anchor(x, y, z, lo, minus[i, 0], minus[i, 1], minus[i, 2])
        if has_plus and j < hi:
            _restore_anchor(x, y, z, hi, plus[i, 0], plus[i, 1], plus[i, 2])
