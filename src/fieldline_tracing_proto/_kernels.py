"""Flat-argument stepping; tuple adapters at the end are inspection-only.

The hot path passes component/grid arrays separately and metadata as scalar
tuples. The scientific RK/event algorithms retain strict arithmetic and caching.
"""

import numpy as np
from numba import njit, prange
from .grid import (
    EXIT_MAX_STEPS,
    EXIT_BNORM_TOO_SMALL,
    EXIT_NAN,
    EXIT_MAX_LENGTH,
    EXIT_STEP_UNDERFLOW,
    EXIT_MAX_ATTEMPTS,
    _wrap_coord,
    _interpolation_stencil,
    _apply_indices,
)


# Dormand–Prince 5(4) tableau (Dormand & Prince, 1980).
_DP_A = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1 / 5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3 / 40, 9 / 40, 0.0, 0.0, 0.0, 0.0, 0.0],
        [44 / 45, -56 / 15, 32 / 9, 0.0, 0.0, 0.0, 0.0],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0.0, 0.0, 0.0],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0.0, 0.0],
        [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0],
    ]
)
_DP_E = np.array(
    [
        35 / 384 - 5179 / 57600,
        0.0,
        500 / 1113 - 7571 / 16695,
        125 / 192 - 393 / 640,
        -2187 / 6784 + 92097 / 339200,
        11 / 84 - 187 / 2100,
        -1 / 40,
    ]
)


from .grid import _axis_cell, _uniform_axis_cell
from .events import (
    _DP_P,
    _first_face_crossing_into,
    _dense_first_exit_into,
    _poly,
    _range_proves_inside,
)


@njit(cache=True, forceinline=True)
def _advance(p, v, step):
    return (p[0] + step * v[0], p[1] + step * v[1], p[2] + step * v[2])


@njit(cache=True, forceinline=True)
def _wrap_position(Bx, By, Bz, Xg, Yg, Zg, Fl, Un, Inv, Per, Bmin, Tx, Ty, Tz, p):
    return (
        _wrap_coord(p[0], Xg[0], Xg[-1], Per[0]) if Fl[0] else p[0],
        _wrap_coord(p[1], Yg[0], Yg[-1], Per[1]) if Fl[1] else p[1],
        _wrap_coord(p[2], Zg[0], Zg[-1], Per[2]) if Fl[2] else p[2],
    )


@njit(cache=True, forceinline=True)
def _roundoff_clamp(q, g, periodic):
    if not periodic and (q < g[0] or q > g[-1]):
        noise = (
            32
            * np.finfo(np.float64).eps
            * max(abs(g[0]), abs(g[-1]), abs(q), np.finfo(np.float64).tiny)
        )
        if q < g[0] and q >= g[0] - noise:
            return g[0]
        if q > g[-1] and q <= g[-1] + noise:
            return g[-1]
    return q


@njit(cache=True, forceinline=True)
def _rhs_xyz_flat(Bx, By, Bz, Xg, Yg, Zg, Fl, Un, Inv, Per, Bmin, Tx, Ty, Tz, x, y, z):
    """Scalar unit field; no allocation or vector arithmetic in the hot path."""
    if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
        return 0.0, 0.0, 0.0, EXIT_NAN
    x = _roundoff_clamp(x, Xg, Fl[0])
    y = _roundoff_clamp(y, Yg, Fl[1])
    z = _roundoff_clamp(z, Zg, Fl[2])
    ix, iy, iz, w, ok, code = _interpolation_stencil(
        Xg,
        Yg,
        Zg,
        x,
        y,
        z,
        Fl[0],
        Fl[1],
        Fl[2],
        Un[0],
        Un[1],
        Un[2],
        Inv[0],
        Inv[1],
        Inv[2],
        (Per[0], Per[1], Per[2]),
        (Tx, Ty, Tz),
    )
    if not ok:
        return 0.0, 0.0, 0.0, code
    jx = ix + 1 if ix + 1 < Xg.size else 0
    jy = iy + 1 if iy + 1 < Yg.size else 0
    jz = iz + 1 if iz + 1 < Zg.size else 0
    bx = _apply_indices(Bx, ix, iy, iz, jx, jy, jz, w)
    by = _apply_indices(By, ix, iy, iz, jx, jy, jz, w)
    bz = _apply_indices(Bz, ix, iy, iz, jx, jy, jz, w)
    norm = np.hypot(np.hypot(bx, by), bz)
    if not np.isfinite(norm):
        return 0.0, 0.0, 0.0, EXIT_NAN
    if norm <= Bmin:
        return 0.0, 0.0, 0.0, EXIT_BNORM_TOO_SMALL
    return bx / norm, by / norm, bz / norm, 0


@njit(cache=True)
def _rhs_flat(Bx, By, Bz, Xg, Yg, Zg, Fl, Un, Inv, Per, Bmin, Tx, Ty, Tz, p):
    # Private convenience adapter; stepping calls the scalar helper directly.
    ux, uy, uz, code = _rhs_xyz_flat(
        Bx, By, Bz, Xg, Yg, Zg, Fl, Un, Inv, Per, Bmin, Tx, Ty, Tz, p[0], p[1], p[2]
    )
    return np.array([ux, uy, uz]), code


@njit(cache=True, forceinline=True)
def _cell_width(g, p, active, uniform, inverse, period, table):
    if uniform:
        i, _ = _uniform_axis_cell(g, p, active, inverse, period)
    else:
        i, _ = _axis_cell(g, p, active, False, inverse, period, table)
    if i < 0:
        return 0.0
    return g[i + 1] - g[i] if i < g.size - 1 else g[0] + period - g[-1]


@njit(cache=True, forceinline=True)
def _local_cap(Bx, By, Bz, Xg, Yg, Zg, Fl, Un, Inv, Per, Bmin, Tx, Ty, Tz, p):
    return min(
        _cell_width(
            Xg,
            p[0],
            Fl[0],
            Un[0],
            Inv[0],
            Per[0],
            Tx,
        ),
        _cell_width(
            Yg,
            p[1],
            Fl[1],
            Un[1],
            Inv[1],
            Per[1],
            Ty,
        ),
        _cell_width(
            Zg,
            p[2],
            Fl[2],
            Un[2],
            Inv[2],
            Per[2],
            Tz,
        ),
    )


@njit(cache=True, forceinline=True)
def _far_from_faces(
    Bx, By, Bz, Xg, Yg, Zg, Fl, Un, Inv, Per, Bmin, Tx, Ty, Tz, p, distance
):
    return (
        (Fl[0] or (p[0] - Xg[0] > distance and Xg[-1] - p[0] > distance))
        and (Fl[1] or (p[1] - Yg[0] > distance and Yg[-1] - p[1] > distance))
        and (Fl[2] or (p[2] - Zg[0] > distance and Zg[-1] - p[2] > distance))
    )


@njit(cache=True, forceinline=True)
def _face_axis(g, p, v, periodic, offset):
    if periodic or v == 0:
        return np.inf, 0
    upper = v > 0
    face = g[-1] if upper else g[0]
    d = (face - p) / v
    return (d, offset + int(upper)) if d >= 0 else (np.inf, 0)


@njit(cache=True, forceinline=True)
def _face_limit(
    Bx, By, Bz, Xg, Yg, Zg, Fl, Un, Inv, Per, Bmin, Tx, Ty, Tz, p, velocity, sign
):
    d, code = _face_axis(Xg, p[0], sign * velocity[0], Fl[0], 5)
    dy, cy = _face_axis(Yg, p[1], sign * velocity[1], Fl[1], 3)
    dz, cz = _face_axis(Zg, p[2], sign * velocity[2], Fl[2], 1)
    if dy < d:
        d, code = dy, cy
    if dz < d:
        d, code = dz, cz
    return d, code


@njit(cache=True, forceinline=True)
def _project_axis(g, q, periodic):
    if periodic:
        return q, True
    noise = (
        64
        * np.finfo(np.float64).eps
        * max(abs(g[0]), abs(g[-1]), abs(q), np.finfo(np.float64).tiny)
    )
    if q < g[0] - noise or q > g[-1] + noise:
        return q, False
    return min(g[-1], max(g[0], q)), True


@njit(cache=True, forceinline=True)
def _project_face(Bx, By, Bz, Xg, Yg, Zg, Fl, Un, Inv, Per, Bmin, Tx, Ty, Tz, p, code):
    x = Xg[-1] if code == 6 else Xg[0] if code == 5 else p[0]
    y = Yg[-1] if code == 4 else Yg[0] if code == 3 else p[1]
    z = Zg[-1] if code == 2 else Zg[0] if code == 1 else p[2]
    x, okx = _project_axis(Xg, x, Fl[0])
    y, oky = _project_axis(Yg, y, Fl[1])
    z, okz = _project_axis(Zg, z, Fl[2])
    return (x, y, z), okx and oky and okz


@njit(cache=True, forceinline=True)
def _rk45_trial_flat(
    Bx,
    By,
    Bz,
    Xg,
    Yg,
    Zg,
    Fl,
    Un,
    Inv,
    Per,
    Bmin,
    Tx,
    Ty,
    Tz,
    p,
    h,
    k1,
    scales,
    stages,
    point,
    c,
    split,
    roots,
    boundary_tol=0.0,
):
    stages[0, 0], stages[0, 1], stages[0, 2] = k1
    for stage in range(1, 7):
        for axis in range(3):
            delta = 0.0
            for j in range(stage):
                delta += _DP_A[stage, j] * stages[j, axis]
            point[axis] = p[axis] + h * delta
        ux, uy, uz, code = _rhs_xyz_flat(
            Bx,
            By,
            Bz,
            Xg,
            Yg,
            Zg,
            Fl,
            Un,
            Inv,
            Per,
            Bmin,
            Tx,
            Ty,
            Tz,
            point[0],
            point[1],
            point[2],
        )
        if code:
            return p, k1, np.inf, code, stage, 2.0, 0, p
        stages[stage, 0], stages[stage, 1], stages[stage, 2] = ux, uy, uz
    error = 0.0
    for axis in range(3):
        e = 0.0
        for stage in range(7):
            e += _DP_E[stage] * stages[stage, axis]
        error += (h * e / scales[axis]) ** 2
    error = np.sqrt(error / 3.0)
    fraction, event = 2.0, 0
    trial = (point[0], point[1], point[2])
    endpoint = trial
    if error <= 1:
        fraction, event = _dense_first_exit_into(
            p,
            stages,
            h,
            Xg,
            Yg,
            Zg,
            Fl[0],
            Fl[1],
            Fl[2],
            c,
            split,
            roots,
            trial,
            boundary_tol,
        )
        if event:
            for axis in range(3):
                point[axis] = p[axis]
                for power in range(4):
                    coefficient = 0.0
                    for j in range(7):
                        coefficient += stages[j, axis] * _DP_P[j, power]
                    point[axis] += h * coefficient * fraction ** (power + 1)
            endpoint = (point[0], point[1], point[2])
    return (
        trial,
        (stages[6, 0], stages[6, 1], stages[6, 2]),
        error,
        0,
        6,
        fraction,
        event,
        endpoint,
    )


@njit(cache=True, forceinline=True)
def _rk2_trial_flat(
    Bx,
    By,
    Bz,
    Xg,
    Yg,
    Zg,
    Fl,
    Un,
    Inv,
    Per,
    Bmin,
    Tx,
    Ty,
    Tz,
    p,
    h,
    k1,
    c,
    split,
    roots,
    Lo,
    Hi,
):
    midp = _advance(p, k1, 0.5 * h)
    ux, uy, uz, code = _rhs_xyz_flat(
        Bx,
        By,
        Bz,
        Xg,
        Yg,
        Zg,
        Fl,
        Un,
        Inv,
        Per,
        Bmin,
        Tx,
        Ty,
        Tz,
        midp[0],
        midp[1],
        midp[2],
    )
    mid = (ux, uy, uz)
    if code:
        return p, k1, np.inf, code, 1, 2.0, 0, p
    point = _advance(p, mid, h)
    first, event = 2.0, 0
    for axis in range(3):
        if Fl[axis]:
            continue
        linear = h * k1[axis]
        quadratic = h * (mid[axis] - k1[axis])
        if _range_proves_inside(
            (p[axis], linear, quadratic, 0.0, 0.0), Lo[axis], Hi[axis]
        ):
            continue
        c[0] = p[axis]
        c[1] = linear
        c[2] = quadratic
        c[3] = 0.0
        c[4] = 0.0
        theta, side = _first_face_crossing_into(c, Lo[axis], Hi[axis], split, roots)
        if side and theta < first - 64 * np.finfo(np.float64).eps:
            first = theta
            event = (5, 3, 1)[axis] + int(side > 0)
    if event:
        endpoint = (
            _poly((p[0], h * k1[0], h * (mid[0] - k1[0]), 0.0, 0.0), first),
            _poly((p[1], h * k1[1], h * (mid[1] - k1[1]), 0.0, 0.0), first),
            _poly((p[2], h * k1[2], h * (mid[2] - k1[2]), 0.0, 0.0), first),
        )
        return point, mid, 0.0, 0, 1, first, event, endpoint
    ux, uy, uz, code = _rhs_xyz_flat(
        Bx,
        By,
        Bz,
        Xg,
        Yg,
        Zg,
        Fl,
        Un,
        Inv,
        Per,
        Bmin,
        Tx,
        Ty,
        Tz,
        point[0],
        point[1],
        point[2],
    )
    return point, (ux, uy, uz), 0.0, code, 2, 2.0, 0, point


@njit(cache=True, forceinline=True)
def _quantized(p, small):
    if small:
        return (
            np.float64(np.float32(p[0])),
            np.float64(np.float32(p[1])),
            np.float64(np.float32(p[2])),
        )
    return p


@njit(cache=True, forceinline=True)
def _save_changed(
    Bx,
    By,
    Bz,
    Xg,
    Yg,
    Zg,
    Fl,
    Un,
    Inv,
    Per,
    Bmin,
    Tx,
    Ty,
    Tz,
    p,
    q,
    last_unwrapped,
    lastp,
    wrapped,
    arc_delta,
    small,
):
    if q[0] != lastp[0] or q[1] != lastp[1] or q[2] != lastp[2]:
        return True
    if wrapped:
        for a in range(3):
            if Per[a] > 0 and abs(p[a] - last_unwrapped[a]) >= 0.5 * Per[a]:
                return True
    # A closed path can return to the same stored coordinates after substantial
    # arc. Merge only distances compatible with output-coordinate roundoff.
    eps = np.finfo(np.float32).eps if small else np.finfo(np.float64).eps
    tiny = np.finfo(np.float32).tiny if small else np.finfo(np.float64).tiny
    scale = max(
        abs(q[0]),
        abs(q[1]),
        abs(q[2]),
        abs(lastp[0]),
        abs(lastp[1]),
        abs(lastp[2]),
        tiny,
    )
    return arc_delta > 4 * eps * scale


_BUFFER_FULL = 0  # Internal retry status; never a physical exit code.


@njit(cache=True)
def _one_sided_cell(g, p, velocity, active, uniform, inverse, period, table):
    """Cell approached by this branch, including periodic seam one-sidedness."""
    if uniform:
        i, t = _uniform_axis_cell(g, p, active, inverse, period)
    else:
        i, t = _axis_cell(g, p, active, False, inverse, period, table)
    if i < 0:
        return -1, 0.0, 0.0
    q = _wrap_coord(p, g[0], g[-1], period) if active else p
    if velocity < 0 and q == g[i]:
        if i > 0:
            i -= 1
            t = 1.0
        elif active:
            i = g.size - 1 if period > g[-1] - g[0] else g.size - 2
            t = 1.0
    width = g[i + 1] - g[i] if i < g.size - 1 else g[0] + period - g[-1]
    return i, t, width


@njit(cache=True)
def _face_component(a, ix, iy, iz, tx, ty, tz, dx, dy, dz, u):
    """One-sided trilinear curvature and cancellation in the normal field."""
    jx = ix + 1 if ix + 1 < a.shape[0] else 0
    jy = iy + 1 if iy + 1 < a.shape[1] else 0
    jz = iz + 1 if iz + 1 < a.shape[2] else 0
    c000, c100 = np.float64(a[ix, iy, iz]), np.float64(a[jx, iy, iz])
    c010, c110 = np.float64(a[ix, jy, iz]), np.float64(a[jx, jy, iz])
    c001, c101 = np.float64(a[ix, iy, jz]), np.float64(a[jx, iy, jz])
    c011, c111 = np.float64(a[ix, jy, jz]), np.float64(a[jx, jy, jz])
    x0, y0, z0 = 1 - tx, 1 - ty, 1 - tz
    samples = (
        c000 * (x0 * y0 * z0),
        c100 * (tx * y0 * z0),
        c010 * (x0 * ty * z0),
        c110 * (tx * ty * z0),
        c001 * (x0 * y0 * tz),
        c101 * (tx * y0 * tz),
        c011 * (x0 * ty * tz),
        c111 * (tx * ty * tz),
    )
    normal, amplitude = 0.0, 0.0
    for sample in samples:
        normal += sample
        amplitude += abs(sample)
    # Input quantization is distinct from roundoff in the float64 stencil.
    # At a node a small, nonzero stored value still has a resolvable sign.
    eps = np.finfo(np.float64).eps
    input_eps = np.finfo(a.dtype).eps
    uncertain = (
        np.isfinite(normal)
        and np.isfinite(amplitude)
        and amplitude > 0.0
        and abs(normal) <= (input_eps + 16 * eps) * amplitude
    )
    # Accumulate signed differences separately so cancellation does not turn a
    # numerically indeterminate tangent into a claimed immediate exit.
    terms = (
        u[0] * (1 - ty) * (1 - tz) * (c100 - c000) / dx,
        u[0] * ty * (1 - tz) * (c110 - c010) / dx,
        u[0] * (1 - ty) * tz * (c101 - c001) / dx,
        u[0] * ty * tz * (c111 - c011) / dx,
        u[1] * (1 - tx) * (1 - tz) * (c010 - c000) / dy,
        u[1] * tx * (1 - tz) * (c110 - c100) / dy,
        u[1] * (1 - tx) * tz * (c011 - c001) / dy,
        u[1] * tx * tz * (c111 - c101) / dy,
        u[2] * (1 - tx) * (1 - ty) * (c001 - c000) / dz,
        u[2] * tx * (1 - ty) * (c101 - c100) / dz,
        u[2] * (1 - tx) * ty * (c011 - c010) / dz,
        u[2] * tx * ty * (c111 - c110) / dz,
    )
    value, magnitude = 0.0, 0.0
    for term in terms:
        value += term
        magnitude += abs(term)
    return value, 64 * eps * max(magnitude, np.finfo(np.float64).tiny), uncertain


@njit(cache=True)
def _initial_face_exit(
    Bx, By, Bz, Xg, Yg, Zg, Fl, Un, Inv, Per, Tx, Ty, Tz, p, u, sign, Lo, Hi
):
    """Resolve boundary starts using one-sided local curvature.

    At B_normal=0, the sign of normal acceleration is the sign of
    (u dot grad) B_normal / |B|. Branch signs cancel in that acceleration,
    but select different one-sided cells at a nonsmooth grid node.
    """
    tangent = u
    for axis in range(3):
        if Fl[axis]:
            continue
        outward = -1.0 if p[axis] == Lo[axis] else 1.0 if p[axis] == Hi[axis] else 0.0
        if outward == 0:
            continue
        normal = sign * u[axis]
        code = (5, 3, 1)[axis] + int(outward > 0)
        ix, tx, dx = _one_sided_cell(
            Xg, p[0], sign * u[0], Fl[0], Un[0], Inv[0], Per[0], Tx
        )
        iy, ty, dy = _one_sided_cell(
            Yg, p[1], sign * u[1], Fl[1], Un[1], Inv[1], Per[1], Ty
        )
        iz, tz, dz = _one_sided_cell(
            Zg, p[2], sign * u[2], Fl[2], Un[2], Inv[2], Per[2], Tz
        )
        if ix < 0 or iy < 0 or iz < 0:
            if outward * normal > 0:
                return code, tangent
            continue
        a = Bx if axis == 0 else By if axis == 1 else Bz
        planar = (
            0.0 if axis == 0 else u[0],
            0.0 if axis == 1 else u[1],
            0.0 if axis == 2 else u[2],
        )
        curvature, noise, uncertain = _face_component(
            a, ix, iy, iz, tx, ty, tz, dx, dy, dz, planar
        )
        if (
            (normal == 0 or uncertain)
            and np.isfinite(curvature)
            and np.isfinite(noise)
            and abs(curvature) > noise
        ):
            if outward * curvature > 0:
                return code, tangent
            tangent = (
                0.0 if axis == 0 else tangent[0],
                0.0 if axis == 1 else tangent[1],
                0.0 if axis == 2 else tangent[2],
            )
        elif outward * normal > 0:
            return code, tangent
    return 0, tangent


@njit(cache=True)
def _branch_flat(
    Bx,
    By,
    Bz,
    Xg,
    Yg,
    Zg,
    Fl,
    Un,
    Inv,
    Per,
    Bmin,
    Tx,
    Ty,
    Tz,
    seed,
    options,
    sign,
    xs,
    ys,
    zs,
    ss,
    seedpos,
    direction,
    write,
    capacity,
):
    adaptive = options[0] > 0
    ds = options[1]
    maxsteps = int(options[2])
    stride = int(options[3])
    minh, maxh, maxL, maxtries, tol = (
        options[4],
        options[5],
        options[6],
        int(options[7]),
        options[8],
    )
    scales = (options[9], options[10], options[11])
    small = options[12] > 0
    p = (seed[0], seed[1], seed[2])
    Lo = (Xg[0], Yg[0], Zg[0])
    Hi = (Xg[-1], Yg[-1], Zg[-1])
    k1 = (0.0, 0.0, 0.0)
    stages = np.empty((7, 3))
    point_work = np.empty(3)
    c = np.empty(5)
    split = np.empty(4)
    roots = np.empty(10)
    cached = False
    nfev = 0
    rejected = 0
    attempts = 0
    nsteps = 0
    length = 0.0
    nsaved = 0
    laststep = 0
    lastp = _quantized(p, small)
    last_unwrapped = p
    last_length = 0.0
    h = ds
    code = EXIT_MAX_STEPS
    after_rejection = False
    noise_floor = (
        32
        * np.finfo(np.float64).eps
        * max(abs(Xg[0]), abs(Xg[-1]), np.finfo(np.float64).tiny),
        32
        * np.finfo(np.float64).eps
        * max(abs(Yg[0]), abs(Yg[-1]), np.finfo(np.float64).tiny),
        32
        * np.finfo(np.float64).eps
        * max(abs(Zg[0]), abs(Zg[-1]), np.finfo(np.float64).tiny),
    )
    while nsteps < maxsteps:
        if length >= maxL:
            code = EXIT_MAX_LENGTH
            break
        if attempts >= maxtries:
            code = EXIT_MAX_ATTEMPTS
            break
        if not cached:
            ux, uy, uz, code0 = _rhs_xyz_flat(
                Bx,
                By,
                Bz,
                Xg,
                Yg,
                Zg,
                Fl,
                Un,
                Inv,
                Per,
                Bmin,
                Tx,
                Ty,
                Tz,
                p[0],
                p[1],
                p[2],
            )
            k1 = (ux, uy, uz)
            nfev += 1
            if code0:
                code = code0
                break
            cached = True
            initial_exit, k1 = _initial_face_exit(
                Bx,
                By,
                Bz,
                Xg,
                Yg,
                Zg,
                Fl,
                Un,
                Inv,
                Per,
                Tx,
                Ty,
                Tz,
                p,
                k1,
                sign,
                Lo,
                Hi,
            )
            if initial_exit:
                code = initial_exit
                break
        limit = (
            _local_cap(Bx, By, Bz, Xg, Yg, Zg, Fl, Un, Inv, Per, Bmin, Tx, Ty, Tz, p)
            * options[14]
            if adaptive and maxh == 0
            else maxh
            if adaptive
            else ds
        )
        trial = min(h, limit, maxL - length)
        # Unit-speed displacement cannot reach a face from this enclosure.
        # Uncertain/near-face cases retain the original tangent predictor.
        if _far_from_faces(
            Bx,
            By,
            Bz,
            Xg,
            Yg,
            Zg,
            Fl,
            Un,
            Inv,
            Per,
            Bmin,
            Tx,
            Ty,
            Tz,
            p,
            2 * max(trial, tol),
        ):
            face_h, face_code = np.inf, 0
        else:
            face_h, face_code = _face_limit(
                Bx, By, Bz, Xg, Yg, Zg, Fl, Un, Inv, Per, Bmin, Tx, Ty, Tz, p, k1, sign
            )
        face_limited = face_h <= trial
        if face_h == 0 and face_code:
            code = face_code
            break
        if face_code and face_h <= tol:
            axis = 2 - (face_code - 1) // 2
            face = Hi[axis] if face_code % 2 == 0 else Lo[axis]
            noise = (
                32
                * np.finfo(np.float64).eps
                * max(
                    abs(Lo[axis]),
                    abs(Hi[axis]),
                    abs(p[axis]),
                    np.finfo(np.float64).tiny,
                )
            )
            if face - noise <= p[axis] <= face + noise and length + face_h <= maxL:
                p, valid = _project_face(
                    Bx,
                    By,
                    Bz,
                    Xg,
                    Yg,
                    Zg,
                    Fl,
                    Un,
                    Inv,
                    Per,
                    Bmin,
                    Tx,
                    Ty,
                    Tz,
                    _advance(p, k1, sign * face_h),
                    face_code,
                )
                if valid:
                    length += face_h
                    nsteps += 1
                    code = face_code
                    break
        if trial <= 0 or length + trial == length:
            code = EXIT_STEP_UNDERFLOW
            break
        if adaptive and trial < minh and not face_limited and trial < maxL - length:
            code = EXIT_STEP_UNDERFLOW
            break
        if adaptive:
            point, end, error, status, calls, fraction, event, endpoint = (
                _rk45_trial_flat(
                    Bx,
                    By,
                    Bz,
                    Xg,
                    Yg,
                    Zg,
                    Fl,
                    Un,
                    Inv,
                    Per,
                    Bmin,
                    Tx,
                    Ty,
                    Tz,
                    p,
                    sign * trial,
                    k1,
                    scales,
                    stages,
                    point_work,
                    c,
                    split,
                    roots,
                    tol,
                )
            )
        else:
            point, end, error, status, calls, fraction, event, endpoint = (
                _rk2_trial_flat(
                    Bx,
                    By,
                    Bz,
                    Xg,
                    Yg,
                    Zg,
                    Fl,
                    Un,
                    Inv,
                    Per,
                    Bmin,
                    Tx,
                    Ty,
                    Tz,
                    p,
                    sign * trial,
                    k1,
                    c,
                    split,
                    roots,
                    Lo,
                    Hi,
                )
            )
        nfev += calls
        attempts += 1
        if status:
            # An invalid initial field stops above. A weak *trial* stage may
            # be avoidable; recover by shortening, with bounded attempts.
            if status == EXIT_NAN:
                code = status
                break
            rejected += 1
            h = face_h if 1 <= status <= 6 and 0 < face_h < trial else trial * 0.5
            after_rejection = True
            if h < minh or trial <= tol:
                code = status if status == EXIT_BNORM_TOO_SMALL else EXIT_STEP_UNDERFLOW
                break
            continue
        if adaptive and (not np.isfinite(error) or error > 1.0):
            rejected += 1
            h = trial * (max(0.2, 0.9 * error ** (-0.2)) if np.isfinite(error) else 0.2)
            after_rejection = True
            if h < minh:
                code = EXIT_STEP_UNDERFLOW
                break
            continue
        used = trial
        terminal = 0
        if event:
            point, valid = _project_face(
                Bx,
                By,
                Bz,
                Xg,
                Yg,
                Zg,
                Fl,
                Un,
                Inv,
                Per,
                Bmin,
                Tx,
                Ty,
                Tz,
                endpoint,
                event,
            )
            if not valid:
                rejected += 1
                h = face_h if 1 <= status <= 6 and 0 < face_h < trial else trial * 0.5
                after_rejection = True
                continue
            used = trial * fraction
            terminal = event
        else:
            outside = False
            for axis in range(3):
                normal = sign * end[axis]
                if Fl[axis]:
                    continue
                outside |= point[axis] < Lo[axis] or point[axis] > Hi[axis]
                if normal == 0:
                    continue
                fc = (5, 3, 1)[axis] + int(normal > 0)
                face = Hi[axis] if normal > 0 else Lo[axis]
                gap = face - point[axis]
                if abs(gap) > 2 * noise_floor[axis]:
                    continue
                correction = gap / normal
                noise = (
                    32
                    * np.finfo(np.float64).eps
                    * max(
                        abs(Lo[axis]),
                        abs(Hi[axis]),
                        abs(point[axis]),
                        np.finfo(np.float64).tiny,
                    )
                )
                # Use the same rounded interval as _roundoff_clamp. An
                # abs(gap) comparison can disagree by one ulp at its edge.
                if (
                    face - noise <= point[axis] <= face + noise
                    and abs(correction) <= tol
                    and used + correction >= 0
                    and length + used + correction <= maxL
                ):
                    projected, valid = _project_face(
                        Bx,
                        By,
                        Bz,
                        Xg,
                        Yg,
                        Zg,
                        Fl,
                        Un,
                        Inv,
                        Per,
                        Bmin,
                        Tx,
                        Ty,
                        Tz,
                        _advance(point, end, sign * correction),
                        fc,
                    )
                    if valid:
                        point = projected
                        used += correction
                        terminal = fc
                        break
            if outside and not terminal:
                rejected += 1
                h = trial * 0.5
                after_rejection = True
                if h < minh or trial <= tol:
                    code = EXIT_STEP_UNDERFLOW
                    break
                continue
        if used == 0:
            code = terminal if terminal else EXIT_STEP_UNDERFLOW
            break
        if point[0] == p[0] and point[1] == p[1] and point[2] == p[2]:
            code = terminal if terminal else EXIT_STEP_UNDERFLOW
            break
        length = min(maxL, length + used)
        p = point
        nsteps += 1
        k1 = end
        cached = True
        if nsteps % stride == 0 or terminal:
            q = _quantized(
                _wrap_position(
                    Bx, By, Bz, Xg, Yg, Zg, Fl, Un, Inv, Per, Bmin, Tx, Ty, Tz, p
                )
                if options[13] > 0
                else p,
                small,
            )
            if _save_changed(
                Bx,
                By,
                Bz,
                Xg,
                Yg,
                Zg,
                Fl,
                Un,
                Inv,
                Per,
                Bmin,
                Tx,
                Ty,
                Tz,
                p,
                q,
                last_unwrapped,
                lastp,
                options[13] > 0,
                length - last_length,
                small,
            ):
                if write:
                    if nsaved >= capacity:
                        return (
                            nsaved,
                            _BUFFER_FULL,
                            nsteps,
                            length,
                            rejected,
                            nfev,
                            p,
                            False,
                        )
                    idx = seedpos + direction * (nsaved + 1)
                    xs[idx], ys[idx], zs[idx], ss[idx] = q[0], q[1], q[2], sign * length
                nsaved += 1
                lastp = q
                last_unwrapped = p
                last_length = length
                laststep = nsteps
            elif terminal and nsaved > 0:
                if write:
                    ss[seedpos + direction * nsaved] = sign * length
                laststep = nsteps
        if terminal:
            code = terminal
            break
        if adaptive:
            factor = 5.0 if error == 0 else min(5.0, max(0.2, 0.9 * error ** (-0.2)))
            h = trial * (min(1.0, factor) if after_rejection else factor)
            after_rejection = False
        else:
            h = ds
    # Face and arc-limit events separated by only rounding in accumulated arc
    # are a tie. Normalize before final saving, and update any terminal save.
    if (
        1 <= code <= 6
        and np.isfinite(maxL)
        and length > 0
        and 0 <= maxL - length <= 8 * np.finfo(np.float64).eps * maxL
    ):
        length = maxL
        code = EXIT_MAX_LENGTH
        if write and nsaved > 0 and laststep == nsteps:
            ss[seedpos + direction * nsaved] = sign * length
    if nsteps > laststep:
        q = _quantized(
            _wrap_position(
                Bx, By, Bz, Xg, Yg, Zg, Fl, Un, Inv, Per, Bmin, Tx, Ty, Tz, p
            )
            if options[13] > 0
            else p,
            small,
        )
        if _save_changed(
            Bx,
            By,
            Bz,
            Xg,
            Yg,
            Zg,
            Fl,
            Un,
            Inv,
            Per,
            Bmin,
            Tx,
            Ty,
            Tz,
            p,
            q,
            last_unwrapped,
            lastp,
            options[13] > 0,
            length - last_length,
            small,
        ):
            if write:
                if nsaved >= capacity:
                    return (
                        nsaved,
                        _BUFFER_FULL,
                        nsteps,
                        length,
                        rejected,
                        nfev,
                        p,
                        False,
                    )
                idx = seedpos + direction * (nsaved + 1)
                xs[idx], ys[idx], zs[idx], ss[idx] = q[0], q[1], q[2], sign * length
            nsaved += 1
        elif nsaved > 0 and write:
            ss[seedpos + direction * nsaved] = sign * length
    if length >= maxL:
        code = EXIT_MAX_LENGTH
    return (
        nsaved,
        code,
        nsteps,
        length,
        rejected,
        nfev,
        (
            _wrap_position(
                Bx, By, Bz, Xg, Yg, Zg, Fl, Un, Inv, Per, Bmin, Tx, Ty, Tz, p
            )
            if options[13] > 0
            else p
        ),
        True,
    )


@njit(parallel=True, cache=True)
def _single_block(
    bx,
    by,
    bz,
    xg,
    yg,
    zg,
    flags,
    uniform,
    inverse,
    periods,
    bmin,
    tx,
    ty,
    tz,
    seeds,
    options,
    order,
    capacity,
    x,
    y,
    z,
    s,
    counts,
    stats,
    endpoints,
    overflow,
):
    for k in prange(len(order)):
        Fl = (flags[0], flags[1], flags[2])
        Un = (uniform[0], uniform[1], uniform[2])
        Inv = (inverse[0], inverse[1], inverse[2])
        Per = (periods[0], periods[1], periods[2])
        i = order[k]
        seed = seeds[i]
        stats[i] = 0.0
        counts[i] = 0
        overflow[i] = False
        x[i, capacity], y[i, capacity], z[i, capacity], s[i, capacity] = (
            seed[0],
            seed[1],
            seed[2],
            0.0,
        )
        for branch in range(2):
            sign = -1.0 if branch == 0 else 1.0
            n, code, steps, length, rejected, fev, end, ok = _branch_flat(
                bx,
                by,
                bz,
                xg,
                yg,
                zg,
                Fl,
                Un,
                Inv,
                Per,
                bmin,
                tx,
                ty,
                tz,
                seed,
                options,
                sign,
                x[i],
                y[i],
                z[i],
                s[i],
                capacity,
                int(sign),
                True,
                capacity,
            )
            counts[i, branch] = n
            stats[i, branch, 0] = code
            stats[i, branch, 1] = steps
            stats[i, branch, 2] = length
            stats[i, branch, 3] = rejected
            stats[i, branch, 4] = fev
            endpoints[i, branch] = end
            if not ok:
                overflow[i] = True
                break


@njit(parallel=True, cache=True)
def _count_lines(
    bx,
    by,
    bz,
    xg,
    yg,
    zg,
    flags,
    uniform,
    inverse,
    periods,
    bmin,
    tx,
    ty,
    tz,
    seeds,
    options,
    order,
    counts,
    stats,
    endpoints,
):
    empty = np.empty(0)
    for k in prange(len(order)):
        Fl = (flags[0], flags[1], flags[2])
        Un = (uniform[0], uniform[1], uniform[2])
        Inv = (inverse[0], inverse[1], inverse[2])
        Per = (periods[0], periods[1], periods[2])
        i = order[k]
        for branch in range(2):
            sign = -1.0 if branch == 0 else 1.0
            n, code, steps, length, rejected, fev, end, ok = _branch_flat(
                bx,
                by,
                bz,
                xg,
                yg,
                zg,
                Fl,
                Un,
                Inv,
                Per,
                bmin,
                tx,
                ty,
                tz,
                seeds[i],
                options,
                sign,
                empty,
                empty,
                empty,
                empty,
                0,
                int(sign),
                False,
                0,
            )
            counts[i, branch] = n
            stats[i, branch, 0] = code
            stats[i, branch, 1] = steps
            stats[i, branch, 2] = length
            stats[i, branch, 3] = rejected
            stats[i, branch, 4] = fev
            endpoints[i, branch] = end


@njit(parallel=True, cache=True)
def _fill_lines(
    bx,
    by,
    bz,
    xg,
    yg,
    zg,
    flags,
    uniform,
    inverse,
    periods,
    bmin,
    tx,
    ty,
    tz,
    seeds,
    options,
    order,
    starts,
    counts,
    stats,
    x,
    y,
    z,
    s,
    ok,
):
    for k in prange(len(order)):
        Fl = (flags[0], flags[1], flags[2])
        Un = (uniform[0], uniform[1], uniform[2])
        Inv = (inverse[0], inverse[1], inverse[2])
        Per = (periods[0], periods[1], periods[2])
        i = order[k]
        seedpos = starts[i] + counts[i, 0]
        x[seedpos], y[seedpos], z[seedpos], s[seedpos] = (
            seeds[i, 0],
            seeds[i, 1],
            seeds[i, 2],
            0.0,
        )
        valid = True
        for branch in range(2):
            sign = -1.0 if branch == 0 else 1.0
            n, code, steps, length, rejected, fev, end, good = _branch_flat(
                bx,
                by,
                bz,
                xg,
                yg,
                zg,
                Fl,
                Un,
                Inv,
                Per,
                bmin,
                tx,
                ty,
                tz,
                seeds[i],
                options,
                sign,
                x,
                y,
                z,
                s,
                seedpos,
                int(sign),
                True,
                counts[i, branch],
            )
            valid = (
                valid
                and good
                and n == counts[i, branch]
                and code == stats[i, branch, 0]
                and steps == stats[i, branch, 1]
                and length == stats[i, branch, 2]
                and rejected == stats[i, branch, 3]
                and fev == stats[i, branch, 4]
            )
        for j in range(starts[i], starts[i] + counts[i, 0] + counts[i, 1] + 1):
            s[j] += stats[i, 0, 2]
        ok[i] = valid


@njit(parallel=True, cache=True)
def _compact_block(
    bx,
    by,
    bz,
    bs,
    counts,
    stats,
    capacity,
    starts,
    retry_map,
    retry_starts,
    rx,
    ry,
    rz,
    rs,
    x,
    y,
    z,
    s,
):
    for i in prange(len(counts)):
        total = counts[i, 0] + counts[i, 1] + 1
        dest = starts[i]
        retry = retry_map[i]
        if retry >= 0:
            source = retry_starts[retry]
            for k in range(total):
                x[dest + k] = rx[source + k]
                y[dest + k] = ry[source + k]
                z[dest + k] = rz[source + k]
                s[dest + k] = rs[source + k]
        else:
            source = capacity - counts[i, 0]
            shift = stats[i, 0, 2]
            for k in range(total):
                x[dest + k] = bx[i, source + k]
                y[dest + k] = by[i, source + k]
                z[dest + k] = bz[i, source + k]
                s[dest + k] = bs[i, source + k] + shift


# Imported-helper edits require a fresh NUMBA_CACHE_DIR; see CONTRIBUTING.md.


@njit(cache=True)
def _rhs_xyz(field, x, y, z):
    return _rhs_xyz_flat(
        field[0],
        field[1],
        field[2],
        field[3],
        field[4],
        field[5],
        (field[6][0], field[6][1], field[6][2]),
        (field[7][0], field[7][1], field[7][2]),
        (field[8][0], field[8][1], field[8][2]),
        (field[9][0], field[9][1], field[9][2]),
        field[10],
        field[11][0],
        field[11][1],
        field[11][2],
        x,
        y,
        z,
    )


@njit(cache=True)
def _rhs(field, p):
    return _rhs_flat(
        field[0],
        field[1],
        field[2],
        field[3],
        field[4],
        field[5],
        (field[6][0], field[6][1], field[6][2]),
        (field[7][0], field[7][1], field[7][2]),
        (field[8][0], field[8][1], field[8][2]),
        (field[9][0], field[9][1], field[9][2]),
        field[10],
        field[11][0],
        field[11][1],
        field[11][2],
        p,
    )


@njit(cache=True)
def _branch(
    field, seed, options, sign, xs, ys, zs, ss, seedpos, direction, write, capacity
):
    return _branch_flat(
        field[0],
        field[1],
        field[2],
        field[3],
        field[4],
        field[5],
        (field[6][0], field[6][1], field[6][2]),
        (field[7][0], field[7][1], field[7][2]),
        (field[8][0], field[8][1], field[8][2]),
        (field[9][0], field[9][1], field[9][2]),
        field[10],
        field[11][0],
        field[11][1],
        field[11][2],
        seed,
        options,
        sign,
        xs,
        ys,
        zs,
        ss,
        seedpos,
        direction,
        write,
        capacity,
    )


@njit(cache=True)
def _rk45_trial(field, p, h, k1, scales, stages, point, c, split, roots):
    return _rk45_trial_flat(
        field[0],
        field[1],
        field[2],
        field[3],
        field[4],
        field[5],
        (field[6][0], field[6][1], field[6][2]),
        (field[7][0], field[7][1], field[7][2]),
        (field[8][0], field[8][1], field[8][2]),
        (field[9][0], field[9][1], field[9][2]),
        field[10],
        field[11][0],
        field[11][1],
        field[11][2],
        (np.float64(p[0]), np.float64(p[1]), np.float64(p[2])),
        h,
        (np.float64(k1[0]), np.float64(k1[1]), np.float64(k1[2])),
        scales,
        stages,
        point,
        c,
        split,
        roots,
    )


@njit(cache=True)
def _rk2_trial(field, p, h, k1, c, split, roots):
    return _rk2_trial_flat(
        field[0],
        field[1],
        field[2],
        field[3],
        field[4],
        field[5],
        (field[6][0], field[6][1], field[6][2]),
        (field[7][0], field[7][1], field[7][2]),
        (field[8][0], field[8][1], field[8][2]),
        (field[9][0], field[9][1], field[9][2]),
        field[10],
        field[11][0],
        field[11][1],
        field[11][2],
        (np.float64(p[0]), np.float64(p[1]), np.float64(p[2])),
        h,
        (np.float64(k1[0]), np.float64(k1[1]), np.float64(k1[2])),
        c,
        split,
        roots,
        (field[3][0], field[4][0], field[5][0]),
        (field[3][-1], field[4][-1], field[5][-1]),
    )
