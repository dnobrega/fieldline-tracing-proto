"""Strict-arithmetic local winding with explicit correspondence and frame.

Default: seed-relative arc length on each pair's common interval, in a fixed
plane normal to the mean seed tangent. This is a specified local proxy, not
a parameterization-independent or Gauss linking invariant.
"""

import json
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numba import njit, prange


@dataclass
class LocalWindingResult:
    mean_wind: np.ndarray
    mean_abs_wind: np.ndarray
    max_abs_wind: np.ndarray
    std_wind: np.ndarray
    n_pairs: np.ndarray
    valid_lines: np.ndarray = None
    invalid_reason: np.ndarray = None


def compute_local_winding(
    result,
    seed_shape,
    *,
    neighbor_offsets=((1, 0), (0, 1), (1, 1), (1, -1)),
    nsample=200,
    min_sep=1e-8,
    dtype=np.float64,
    parameterization="seed_arc",
    plane="mean_tangent",
    period=None,
    sparse_policy="raise",
):
    """Compare oriented pairs with finite, strictly monotone stored geometry.

    parameterization='normalized', plane='xy' selects the historical proxy
    explicitly. The default seed_arc mode requires seed_index. Periodic
    separation uses a continuous cover, anchored to the minimum-image seed
    separation, not independently rewrapped angle samples. Sparse wrapped
    output that cannot safely be unwrapped is rejected; prefer wrap_output=False.
    """
    if (
        isinstance(nsample, (bool, np.bool_))
        or not isinstance(nsample, Integral)
        or nsample < 2
    ):
        raise ValueError("nsample must be an integer >=2.")
    if not np.isfinite(min_sep) or min_sep < 0:
        raise ValueError("min_sep must be finite and nonnegative.")
    if len(seed_shape) != 2 or any(
        not isinstance(n, Integral) or isinstance(n, (bool, np.bool_)) or n < 0
        for n in seed_shape
    ):
        raise ValueError("Invalid seed_shape.")
    if sparse_policy not in ("raise", "mask"):
        raise ValueError("sparse_policy must be raise or mask.")
    ny, nx = map(int, seed_shape)
    n = len(result.line_count)
    if n != ny * nx:
        raise ValueError("seed_shape does not match the number of lines.")
    if parameterization not in ("seed_arc", "normalized"):
        raise ValueError("Unknown parameterization.")
    if plane not in ("xy", "mean_tangent"):
        raise ValueError("Unknown plane.")
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("dtype must be float32 or float64.")
    offsets = np.asarray(neighbor_offsets, dtype=np.int64)
    if offsets.size == 0:
        offsets = np.empty((0, 2), dtype=np.int64)
    if offsets.ndim != 2 or offsets.shape[1] != 2:
        raise ValueError("neighbor_offsets must contain (dy,dx) pairs.")
    start = np.asarray(result.line_start, dtype=np.int64)
    count = np.asarray(result.line_count, dtype=np.int64)
    s = np.asarray(result.sfl, dtype=np.float64)
    z = np.asarray(getattr(result, "zfl", np.zeros_like(result.xfl)), dtype=np.float64)
    points = np.column_stack((result.xfl, result.yfl, z)).astype(np.float64, copy=False)
    if (
        len(s) != len(points)
        or len(start) != n
        or np.any(start < 0)
        or np.any(count < 0)
        or np.any(start + count > len(s))
    ):
        raise ValueError("Invalid ragged geometry.")
    if parameterization == "seed_arc" and not hasattr(result, "seed_index"):
        raise ValueError(
            "seed_arc requires seed_index; select normalized explicitly for synthetic legacy curves."
        )
    seed = np.asarray(getattr(result, "seed_index", start), dtype=np.int64)
    if (
        len(seed) != n
        or np.any(seed < start)
        or np.any(seed >= start + np.maximum(1, count))
    ):
        raise ValueError("Invalid seed_index.")
    settings = json.loads(getattr(result, "settings_json", "{}"))
    periods = np.asarray(
        settings.get("period", [0.0, 0.0, 0.0]) if period is None else period,
        dtype=float,
    )
    if periods.shape != (3,) or not np.all(np.isfinite(periods)) or np.any(periods < 0):
        raise ValueError("period must contain three finite nonnegative entries.")
    # Independent writes during parallel preparation require disjoint segments.
    occupied = count > 0
    ordered = (
        np.argsort(start[occupied]) if np.any(np.diff(start[occupied]) < 0) else None
    )
    lo = start[occupied]
    hi = lo + count[occupied]
    if ordered is not None:
        lo = lo[ordered]
        hi = hi[ordered]
    if np.any(hi[:-1] > lo[1:]):
        raise ValueError("Ragged line segments must not overlap.")
    valid = np.zeros(n, dtype=np.bool_)
    reason = np.zeros(n, dtype=np.int8)
    seed_s = np.zeros(n)
    tangent = np.zeros((n, 3))
    seed_positions = np.zeros((n, 3))
    _prepare_lines(
        points,
        s,
        start,
        count,
        seed,
        periods,
        bool(settings.get("wrap_output", True)),
        plane == "mean_tangent",
        valid,
        reason,
        seed_s,
        tangent,
        seed_positions,
    )
    nsparse = int(np.count_nonzero(reason == 2))
    if nsparse:
        if sparse_policy == "raise":
            raise ValueError(
                "Sparse wrapped periodic geometry is ambiguous; trace with wrap_output=False or save more densely."
            )
        import warnings

        warnings.warn(
            f"{nsparse} sparse wrapped line(s) excluded from winding; inspect valid_lines/invalid_reason.",
            UserWarning,
            stacklevel=2,
        )
    samples = (
        np.empty((n, nsample, 3))
        if parameterization == "normalized"
        else np.empty((0, 0, 3))
    )
    if parameterization == "normalized":
        _resample_normalized(points, s, start, count, valid, nsample, samples)
    arrays = [np.empty(n, dtype=np.float64) for _ in range(4)]
    pairs = np.zeros(n, dtype=np.int32)
    _winding_all(
        points,
        s,
        start,
        count,
        seed_s,
        tangent,
        seed_positions,
        periods,
        valid,
        samples,
        parameterization == "normalized",
        plane == "xy",
        ny,
        nx,
        offsets,
        nsample,
        float(min_sep),
        *arrays,
        pairs,
    )
    return LocalWindingResult(
        *(a.astype(dtype, copy=False) for a in arrays), pairs, valid, reason
    )


@njit(cache=True)
def _angle_diff(theta_new, theta_old):
    d = theta_new - theta_old
    while d > np.pi:
        d -= 2 * np.pi
    while d < -np.pi:
        d += 2 * np.pi
    return d


@njit(cache=True, forceinline=True)
def _point_at(points, s, lo, hi, k, target):
    while k < hi - 2 and s[k + 1] < target:
        k += 1
    f = min(1.0, max(0.0, (target - s[k]) / (s[k + 1] - s[k])))
    return (
        points[k, 0] * (1 - f) + points[k + 1, 0] * f,
        points[k, 1] * (1 - f) + points[k + 1, 1] * f,
        points[k, 2] * (1 - f) + points[k + 1, 2] * f,
    ), k


@njit(parallel=True, cache=True)
def _resample_normalized(points, s, start, count, valid, nsample, out):
    for i in prange(len(count)):
        if not valid[i]:
            continue
        lo = start[i]
        hi = lo + count[i]
        k = lo
        for m in range(nsample):
            target = s[lo] + (s[hi - 1] - s[lo]) * m / (nsample - 1)
            p, k = _point_at(points, s, lo, hi, k, target)
            out[i, m, 0] = p[0]
            out[i, m, 1] = p[1]
            out[i, m, 2] = p[2]


@njit(cache=True)
def _pair(
    points,
    s,
    start,
    count,
    seed_s,
    tangent,
    seed_positions,
    periods,
    samples,
    normalized,
    xy,
    i,
    j,
    nsample,
    min_sep,
):
    nx = tangent[i, 0] + tangent[j, 0]
    ny = tangent[i, 1] + tangent[j, 1]
    nz = tangent[i, 2] + tangent[j, 2]
    if xy:
        e1 = (1.0, 0.0, 0.0)
        e2 = (0.0, 1.0, 0.0)
    else:
        norm = np.hypot(np.hypot(nx, ny), nz)
        if norm <= 32 * np.finfo(np.float64).eps or not np.isfinite(norm):
            return np.nan, False
        nx /= norm
        ny /= norm
        nz /= norm
        if abs(nx) <= abs(ny) and abs(nx) <= abs(nz):
            e1 = (0.0, nz, -ny)
        elif abs(ny) <= abs(nz):
            e1 = (-nz, 0.0, nx)
        else:
            e1 = (ny, -nx, 0.0)
        norm = np.hypot(np.hypot(e1[0], e1[1]), e1[2])
        e1 = (e1[0] / norm, e1[1] / norm, e1[2] / norm)
        e2 = (ny * e1[2] - nz * e1[1], nz * e1[0] - nx * e1[2], nx * e1[1] - ny * e1[0])
    sx = (
        -np.rint((seed_positions[j, 0] - seed_positions[i, 0]) / periods[0])
        * periods[0]
        if periods[0] > 0
        else 0.0
    )
    sy = (
        -np.rint((seed_positions[j, 1] - seed_positions[i, 1]) / periods[1])
        * periods[1]
        if periods[1] > 0
        else 0.0
    )
    sz = (
        -np.rint((seed_positions[j, 2] - seed_positions[i, 2]) / periods[2])
        * periods[2]
        if periods[2] > 0
        else 0.0
    )
    li = start[i]
    lj = start[j]
    hi = li + count[i]
    hj = lj + count[j]
    ki = li
    kj = lj
    low = max(s[li] - seed_s[i], s[lj] - seed_s[j])
    high = min(s[hi - 1] - seed_s[i], s[hj - 1] - seed_s[j])
    if not normalized and high <= low:
        return np.nan, False
    total = 0.0
    previous = 0.0
    for m in range(nsample):
        if normalized:
            p = (samples[i, m, 0], samples[i, m, 1], samples[i, m, 2])
            q = (samples[j, m, 0], samples[j, m, 1], samples[j, m, 2])
        else:
            sigma = low + (high - low) * m / (nsample - 1)
            p, ki = _point_at(points, s, li, hi, ki, sigma + seed_s[i])
            q, kj = _point_at(points, s, lj, hj, kj, sigma + seed_s[j])
        x = q[0] + sx - p[0]
        y = q[1] + sy - p[1]
        z = q[2] + sz - p[2]
        dx = x * e1[0] + y * e1[1] + z * e1[2]
        dy = x * e2[0] + y * e2[1] + z * e2[2]
        if not np.isfinite(dx) or not np.isfinite(dy) or np.hypot(dx, dy) <= min_sep:
            return np.nan, False
        angle = np.arctan2(dy, dx)
        if m > 0:
            total += _angle_diff(angle, previous)
        previous = angle
    value = total / (2 * np.pi)
    return value, np.isfinite(value)


@njit(parallel=True, cache=True)
def _winding_all(
    points,
    s,
    start,
    count,
    seed_s,
    tangent,
    seed_positions,
    periods,
    valid,
    samples,
    normalized,
    xy,
    ny,
    nx,
    offsets,
    nsample,
    min_sep,
    means,
    absolute,
    maximum,
    std,
    pairs,
):
    for i in prange(ny * nx):
        n = 0
        mean = 0.0
        m2 = 0.0
        abs_sum = 0.0
        maxabs = 0.0
        if valid[i]:
            iy = i // nx
            ix = i % nx
            for o in range(len(offsets)):
                jy = iy + offsets[o, 0]
                jx = ix + offsets[o, 1]
                if jy < 0 or jy >= ny or jx < 0 or jx >= nx:
                    continue
                j = jy * nx + jx
                if not valid[j]:
                    continue
                value, ok = _pair(
                    points,
                    s,
                    start,
                    count,
                    seed_s,
                    tangent,
                    seed_positions,
                    periods,
                    samples,
                    normalized,
                    xy,
                    i,
                    j,
                    nsample,
                    min_sep,
                )
                if not ok:
                    continue
                n += 1
                difference = value - mean
                mean += difference / n
                m2 += difference * (value - mean)
                abs_sum += abs(value)
                maxabs = max(maxabs, abs(value))
        pairs[i] = n
        if n:
            means[i] = mean
            absolute[i] = abs_sum / n
            maximum[i] = maxabs
            std[i] = np.sqrt(max(0.0, m2 / n))
        else:
            means[i] = np.nan
            absolute[i] = np.nan
            maximum[i] = np.nan
            std[i] = np.nan


@njit(parallel=True, cache=True)
def _prepare_lines(
    points,
    s,
    start,
    count,
    seed,
    periods,
    wrapped,
    need_tangent,
    valid,
    reason,
    seed_s,
    tangent,
    seed_positions,
):
    # reason: 0 valid; 1 invalid/short/nonmonotone; 2 sparse wrapped;
    # 3 degenerate tangent. No per-line arrays or Python calls.
    for i in prange(len(count)):
        lo = start[i]
        hi = lo + count[i]
        if count[i] < 2:
            reason[i] = 1
            continue
        good = True
        maxgap = 0.0
        for k in range(lo, hi):
            if not np.isfinite(s[k]):
                good = False
                break
            for a in range(3):
                if not np.isfinite(points[k, a]):
                    good = False
            if k > lo:
                gap = s[k] - s[k - 1]
                if gap <= 0 or not np.isfinite(gap):
                    good = False
                maxgap = max(maxgap, gap)
        if not good:
            reason[i] = 1
            continue
        j = seed[i]
        seed_s[i] = s[j]
        for a in range(3):
            seed_positions[i, a] = points[j, a]
        sparse = False
        if wrapped:
            for a in range(3):
                if periods[a] > 0 and maxgap >= periods[a] / 2:
                    sparse = True
        if sparse:
            reason[i] = 2
            continue
        if wrapped:
            for a in range(3):
                period = periods[a]
                if period <= 0:
                    continue
                previous = points[lo, a]
                points[lo, a] = 0.0
                cumulative = 0.0
                for k in range(lo + 1, hi):
                    current = points[k, a]
                    delta = current - previous
                    delta -= np.rint(delta / period) * period
                    cumulative += delta
                    points[k, a] = cumulative
                    previous = current
                anchor = points[j, a]
                for k in range(lo, hi):
                    points[k, a] = points[k, a] - anchor + seed_positions[i, a]
        left = max(lo, j - 1)
        right = min(hi - 1, j + 1)
        tx = points[right, 0] - points[left, 0]
        ty = points[right, 1] - points[left, 1]
        tz = points[right, 2] - points[left, 2]
        norm = np.hypot(np.hypot(tx, ty), tz)
        if not np.isfinite(norm) or (norm == 0 and need_tangent):
            reason[i] = 3
            continue
        if norm > 0:
            tangent[i, 0] = tx / norm
            tangent[i, 1] = ty / norm
            tangent[i, 2] = tz / norm
        valid[i] = True
