"""Bidirectional Cartesian field-line tracing and ragged result persistence.

Integrates dr/ds = +/- B/|B| using midpoint RK2 or adaptive Dormand–Prince
RK45. Input component arrays are (nx, ny, nz); seed columns are (x,y,z).
Lengths and grids share coordinate units. See docs/RK45.md and docs/API.md
for error control, boundary semantics, exit codes and storage conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
from typing import Tuple
from numbers import Integral, Real

import numpy as np


from .grid import (
    PeriodicInferenceWarning,
    EXIT_ZMIN,
    EXIT_ZMAX,
    EXIT_YMIN,
    EXIT_YMAX,
    EXIT_XMIN,
    EXIT_XMAX,
    EXIT_MAX_STEPS,
    EXIT_BNORM_TOO_SMALL,
    EXIT_NAN,
    EXIT_INTERP_ERROR,
    EXIT_MAX_LENGTH,
    EXIT_STEP_UNDERFLOW,
    EXIT_MAX_ATTEMPTS,
    validate_cartesian_grid,
    is_uniform_grid,
    _inv_spacing,
    prepare_array,
    prepare_periods,
    lookup_table,
)


@dataclass
class TraceResult:
    xfl: np.ndarray
    yfl: np.ndarray
    zfl: np.ndarray
    sfl: np.ndarray
    line_start: np.ndarray
    line_count: np.ndarray
    seeds: np.ndarray
    exit_code_minus: np.ndarray
    exit_code_plus: np.ndarray
    nsteps_minus: np.ndarray
    nsteps_plus: np.ndarray
    length_minus: np.ndarray
    length_plus: np.ndarray
    seed_index: np.ndarray
    method: str = "rk2"
    settings_json: str = "{}"
    nrejected_minus: np.ndarray = None
    nrejected_plus: np.ndarray = None
    nfev_minus: np.ndarray = None
    nfev_plus: np.ndarray = None
    original_seeds: np.ndarray = None
    endpoints_minus: np.ndarray = None
    endpoints_plus: np.ndarray = None
    exit_faces_minus: np.ndarray = None
    exit_faces_plus: np.ndarray = None

    @property
    def nlines(self) -> int:
        # Number of lines
        return int(self.line_count.size)

    @property
    def total_points(self) -> int:
        # Number of points
        return int(self.xfl.size)

    def get_line(self, i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Function to get the coordinates of the line number #i
        i0 = int(self.line_start[i])
        i1 = i0 + int(self.line_count[i])
        return self.xfl[i0:i1], self.yfl[i0:i1], self.zfl[i0:i1], self.sfl[i0:i1]


def validate_inputs(bx, by, bz, xgrid, ygrid, zgrid, seeds) -> None:
    # Verifications of shapes just in case the user messes it up with the input arrays
    if bx.shape != by.shape or bx.shape != bz.shape:
        raise ValueError("bx, by, bz must have identical shape (nx, ny, nz).")
    if bx.ndim != 3:
        raise ValueError("bx, by, bz must be 3D arrays with shape (nx, ny, nz).")
    validate_cartesian_grid(bx.shape, xgrid, ygrid, zgrid)
    if seeds.ndim != 2 or seeds.shape[1] != 3:
        raise ValueError("seeds must have shape (nseeds, 3), columns (x0, y0, z0).")
    if not np.all(np.isfinite(seeds)):
        raise ValueError("seeds contains NaN/Inf.")


def trace_field_lines(
    bx,
    by,
    bz,
    xgrid,
    ygrid,
    zgrid,
    seeds,
    ds,
    max_steps,
    *,
    save_stride=1,
    periodic_x=None,
    periodic_y=None,
    periodic_z=False,
    period=None,
    periodic_convention="endpoint_excluded",
    dtype=None,
    method="rk2",
    bmin=1e-12,
    rtol=None,
    atol=None,
    min_step=None,
    max_step=None,
    max_step_cells=None,
    max_length=None,
    max_attempts=None,
    boundary_tol=None,
    error_scale=None,
    array_order="K",
    field_layout="preserve",
    storage="auto",
    memory_limit_mb=64.0,
    block_size=None,
    buffer_points=None,
    seed_order="balanced",
    field_order="xyz",
    wrap_output=True,
    warn_periodic=True,
):
    """Trace native (nx,ny,nz) fields, preserving C/F layout by default.

    RK45's omitted max_step uses local mesh spacing, and rtol scales a fixed
    physical length (cell widths by default), never absolute coordinates.
    Explicit max_step=inf is an opt-out of the grid resolution safeguard.
    Single-pass bounded blocks are used when they fit memory_limit_mb;
    storage='two_pass' provides a low-temporary-memory fallback.
    """
    import warnings

    if method not in ("rk2", "rk45"):
        raise ValueError("method must be 'rk2' or 'rk45'.")
    if isinstance(ds, np.ndarray) and ds.ndim == 0:
        ds = ds.item()
    if isinstance(ds, (bool, np.bool_)) or not isinstance(ds, Real):
        raise ValueError("ds must be a finite positive real scalar.")
    try:
        ds = float(ds)
    except (OverflowError, ValueError):
        raise ValueError("ds must be a finite positive real scalar.") from None
    if not np.isfinite(ds) or ds <= 0:
        raise ValueError("ds must be a finite positive real scalar.")
    for name, value in (
        ("max_steps", max_steps),
        ("save_stride", save_stride),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Integral)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer.")
    if max_steps > np.iinfo(np.int32).max:
        raise ValueError("max_steps exceeds the int32 counter limit.")
    if not np.isscalar(bmin) or not np.isfinite(bmin) or bmin < 0:
        raise ValueError("bmin must be finite and nonnegative.")
    if field_order not in ("xyz", "zyx"):
        raise ValueError("field_order must be 'xyz' or 'zyx'.")
    fields_in = [np.asarray(a) for a in (bx, by, bz)]
    if field_order == "zyx":
        warnings.warn(
            "field_order='zyx' is a compatibility adapter; native fields are (nx,ny,nz).",
            DeprecationWarning,
            stacklevel=2,
        )
        fields_in = [a.transpose(2, 1, 0) for a in fields_in]
    raw_grids = tuple(np.asarray(g) for g in (xgrid, ygrid, zgrid))
    seeds = np.asarray(seeds)
    validate_inputs(*fields_in, *raw_grids, seeds)
    uniform = tuple(is_uniform_grid(g) for g in raw_grids)
    grids = tuple(np.ascontiguousarray(g, dtype=np.float64) for g in raw_grids)
    validate_cartesian_grid(fields_in[0].shape, *grids)
    defaulted = [
        name
        for name, value in zip(("x", "y"), (periodic_x, periodic_y))
        if value is None
    ]
    flags = (
        True if periodic_x is None else bool(periodic_x),
        True if periodic_y is None else bool(periodic_y),
        bool(periodic_z),
    )
    periods = prepare_periods(raw_grids, flags, period, periodic_convention)
    original = np.array(seeds, dtype=np.float64, copy=True)
    seeds = original.copy()
    for axis, (g, active, p) in enumerate(zip(grids, flags, periods)):
        if active:
            seeds[:, axis] = g[0] + (seeds[:, axis] - g[0]) % p
            seeds[seeds[:, axis] >= g[0] + p, axis] = g[0]
        elif np.any((seeds[:, axis] < g[0]) | (seeds[:, axis] > g[-1])):
            raise ValueError(f"Seed outside nonperiodic axis {axis}.")
    if dtype is None:
        dtype = np.result_type(*(a.dtype for a in fields_in))
        if dtype not in (np.dtype("float32"), np.dtype("float64")):
            dtype = np.dtype("float64")
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("dtype must be float32 or float64.")
    arrays = [prepare_array(a, dtype, array_order) for a in fields_in]
    if field_layout not in ("preserve", "interleaved"):
        raise ValueError("Unknown field_layout.")
    if field_layout == "interleaved":
        packed = np.stack(arrays, axis=-1)
        arrays = [packed[..., i] for i in range(3)]
    widths = np.array([np.min(np.diff(g)) for g in grids])
    for i, p in enumerate(periods):
        seam = p - (grids[i][-1] - grids[i][0])
        if p > 0 and seam > 0:
            widths[i] = min(widths[i], seam)
    spacing = float(widths.min())
    if method == "rk2" and any(
        v is not None
        for v in (
            rtol,
            atol,
            min_step,
            max_step,
            max_step_cells,
            max_attempts,
            error_scale,
        )
    ):
        raise ValueError("Adaptive controls require method='rk45'.")
    rtol = 1e-6 if rtol is None else float(rtol)
    atol = 1e-9 * spacing if atol is None else float(atol)
    min_step = (
        32 * np.finfo(np.float64).eps * spacing if min_step is None else float(min_step)
    )
    if max_step_cells is not None and max_step is not None:
        raise ValueError("Specify only one of max_step or max_step_cells.")
    cells = 1.0 if max_step_cells is None else float(max_step_cells)
    if not np.isfinite(cells) or cells <= 0:
        raise ValueError("max_step_cells must be finite and positive.")
    cap = 0.0 if max_step is None else float(max_step)
    length = np.inf if max_length is None else float(max_length)
    tol = 1e-9 * spacing if boundary_tol is None else float(boundary_tol)
    attempts = 10 * int(max_steps) + 100 if max_attempts is None else max_attempts
    if (
        isinstance(attempts, (bool, np.bool_))
        or not isinstance(attempts, Integral)
        or attempts <= 0
        or attempts > 2**53
    ):
        raise ValueError("max_attempts must be an integer in [1, 2**53].")
    for name, value in (
        ("rtol", rtol),
        ("atol", atol),
        ("min_step", min_step),
        ("boundary_tol", tol),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if np.isnan(cap) or cap < 0 or (max_step is not None and cap == 0):
        raise ValueError("max_step must be positive.")
    if np.isnan(length) or length <= 0:
        raise ValueError("max_length must be positive.")
    if cap > 0 and min_step > cap:
        raise ValueError("min_step exceeds max_step.")
    scale = (
        widths
        if error_scale is None
        else np.broadcast_to(np.asarray(error_scale, dtype=float), (3,))
    )
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("error_scale must contain positive physical lengths.")
    scales = np.asarray(atol + rtol * scale, dtype=np.float64)
    inverse = tuple(float(_inv_spacing(g)) for g in grids)
    tables = tuple(lookup_table(g, u) for g, u in zip(grids, uniform))
    field = (
        *arrays,
        *grids,
        np.asarray(flags),
        np.asarray(uniform),
        np.asarray(inverse),
        np.asarray(periods),
        float(bmin),
        tables,
    )
    options = np.array(
        [
            method == "rk45",
            float(ds),
            int(max_steps),
            # Larger strides all mean "only terminal saves". Keep the
            # internal integer exactly representable in the float option ABI.
            min(int(save_stride), int(max_steps) + 1),
            min_step,
            cap,
            length,
            int(attempts),
            tol,
            *scales,
            dtype == np.dtype("float32"),
            bool(wrap_output),
            cells,
        ],
        dtype=np.float64,
    )
    if not isinstance(warn_periodic, (bool, np.bool_)):
        raise ValueError("warn_periodic must be boolean.")
    raw_period = (None, None, None) if period is None else tuple(period)
    inferred = [
        name
        for name, active, value in zip(("x", "y", "z"), flags, raw_period)
        if active and value is None
    ]
    if warn_periodic and (inferred or defaulted):
        warnings.warn(
            f"Periodic assumptions: default X/Y flags={defaulted}; inferred period axes={inferred}. "
            "For cropped/nonperiodic data set flags explicitly; for periodic data pass physical period=(Px,Py,Pz). "
            "Set warn_periodic=False only when these assumptions are intentional.",
            PeriodicInferenceWarning,
            stacklevel=2,
        )
    from ._storage import integrate_storage

    counts, stats, endpoints, vectors, storage_info = integrate_storage(
        field,
        seeds,
        options,
        dtype,
        storage,
        memory_limit_mb,
        block_size,
        seed_order,
        buffer_points,
    )
    n = len(seeds)
    sizes = counts.sum(axis=1) + 1
    starts = (
        np.r_[0, np.cumsum(sizes)[:-1]].astype(np.int64)
        if n
        else np.empty(0, dtype=np.int64)
    )
    settings = dict(
        algorithm_revision="xyz-flat-boundary-v3",
        field_order="xyz",
        input_field_order=field_order,
        array_order=array_order,
        field_layout=field_layout,
        component_layouts=[
            "F" if a.flags.f_contiguous else "C" if a.flags.c_contiguous else "A"
            for a in arrays
        ],
        wrap_output=bool(wrap_output),
        **storage_info,
        memory_limit_mb=float(memory_limit_mb),
        seed_order=seed_order,
        ds=float(ds),
        max_steps=int(max_steps),
        save_stride=int(save_stride),
        periodic_x=flags[0],
        periodic_y=flags[1],
        periodic_z=flags[2],
        period=list(periods),
        periodic_convention=periodic_convention,
        period_inferred_axes=inferred,
        periodic_flags_defaulted=defaulted,
        warn_periodic=bool(warn_periodic),
        dtype=str(dtype),
        arc_dtype="float64",
        bmin=float(bmin),
        rtol=rtol,
        atol=atol,
        error_scale=list(map(float, scale)),
        min_step=min_step,
        max_step="local_grid" if cap == 0 else None if np.isinf(cap) else cap,
        max_step_cells=cells if cap == 0 else None,
        max_length=None if np.isinf(length) else length,
        max_attempts=int(attempts),
        boundary_tol=tol,
    )
    codes = stats[:, :, 0].astype(np.int16)
    return TraceResult(
        *vectors,
        starts,
        sizes,
        seeds,
        -codes[:, 0],
        codes[:, 1],
        stats[:, 0, 1].astype(np.int32),
        stats[:, 1, 1].astype(np.int32),
        stats[:, 0, 2],
        stats[:, 1, 2],
        starts + counts[:, 0],
        method,
        json.dumps(settings, sort_keys=True),
        stats[:, 0, 3].astype(np.int64),
        stats[:, 1, 3].astype(np.int64),
        stats[:, 0, 4].astype(np.int64),
        stats[:, 1, 4].astype(np.int64),
        original,
        endpoints[:, 0],
        endpoints[:, 1],
        np.where(codes[:, 0] <= 6, codes[:, 0], 0),
        np.where(codes[:, 1] <= 6, codes[:, 1], 0),
    )


def ragged_to_padded(result: TraceResult, fill_value=np.nan):
    """Convert ragged lines to padded matrices with shape (nlines, max_points)."""
    nlines = result.nlines
    max_points = int(result.line_count.max()) if nlines else 0
    dtype = result.xfl.dtype
    xmat = np.full((nlines, max_points), fill_value, dtype=dtype)
    ymat = np.full((nlines, max_points), fill_value, dtype=dtype)
    zmat = np.full((nlines, max_points), fill_value, dtype=dtype)
    smat = np.full((nlines, max_points), fill_value, dtype=result.sfl.dtype)
    for i in range(nlines):
        i0 = int(result.line_start[i])
        n = int(result.line_count[i])
        i1 = i0 + n
        xmat[i, :n] = result.xfl[i0:i1]
        ymat[i, :n] = result.yfl[i0:i1]
        zmat[i, :n] = result.zfl[i0:i1]
        smat[i, :n] = result.sfl[i0:i1]
    return xmat, ymat, zmat, smat


def save_trace_npz(
    filename: str, result: TraceResult, *, compressed: bool = True
) -> None:
    saver = np.savez_compressed if compressed else np.savez
    payload = {
        f.name: getattr(result, f.name)
        for f in fields(TraceResult)
        if getattr(result, f.name) is not None
    }
    saver(filename, schema_version=np.array(3), **payload)


def load_trace_npz(filename: str) -> TraceResult:
    """Load schema 2/3 results or legacy RK2 files, without pickle/object arrays."""
    with np.load(filename, allow_pickle=False) as data:
        if "schema_version" in data and int(data["schema_version"]) not in (2, 3):
            raise ValueError("Unsupported trace schema version.")
        payload = {f.name: data[f.name] for f in fields(TraceResult) if f.name in data}
        for name in ("method", "settings_json"):
            if name in payload:
                payload[name] = str(payload[name].item())
    return TraceResult(**payload)


def print_exit_summary(result: TraceResult) -> None:
    for name, arr in (
        ("exit_code_minus", result.exit_code_minus),
        ("exit_code_plus", result.exit_code_plus),
    ):
        vals, counts = np.unique(arr, return_counts=True)
        print(name)
        for v, c in zip(vals, counts):
            print(f"  {int(v):4d}: {int(c)}")


def exit_code_meaning(code: int) -> str:
    branch = "plus" if code > 0 else "minus"
    meanings = {
        EXIT_ZMIN: "ended at zmin",
        EXIT_ZMAX: "ended at zmax",
        EXIT_YMIN: "ended at ymin",
        EXIT_YMAX: "ended at ymax",
        EXIT_XMIN: "ended at xmin",
        EXIT_XMAX: "ended at xmax",
        EXIT_MAX_STEPS: "reached max_steps",
        EXIT_BNORM_TOO_SMALL: "|B| too small",
        EXIT_NAN: "NaN/Inf encountered",
        EXIT_INTERP_ERROR: "interpolation/domain error",
        EXIT_MAX_LENGTH: "reached max_length",
        EXIT_STEP_UNDERFLOW: "adaptive step underflow",
        EXIT_MAX_ATTEMPTS: "reached max_attempts",
    }
    return f"{branch}: {meanings.get(abs(int(code)), 'unknown')}"


# Private aliases retain the developer inspection entry points.
from ._kernels import (
    _rhs as _rhs,
    _rhs_xyz as _rhs_xyz,
    _branch as _branch,
    _rk2_trial as _rk2_trial,
    _rk45_trial as _rk45_trial,
)
