"""Quartic Dormand–Prince boundary events, isolated between all extrema.

Polynomial roots are bracketed on monotone intervals; no dense time grid,
field extrapolation, SciPy runtime dependency or complex root solver is used.
"""

import numpy as np
from numba import njit

# Shampine's quartic continuous extension of the Dormand–Prince pair.
# Reference: Mathematics of Computation 46 (1986), 135–150.
_DP_P = np.array(
    [
        [
            1.0,
            -8048581381 / 2820520608,
            8663915743 / 2820520608,
            -12715105075 / 11282082432,
        ],
        [0.0, 0.0, 0.0, 0.0],
        [
            0.0,
            131558114200 / 32700410799,
            -68118460800 / 10900136933,
            87487479700 / 32700410799,
        ],
        [
            0.0,
            -1754552775 / 470086768,
            14199869525 / 1410260304,
            -10690763975 / 1880347072,
        ],
        [
            0.0,
            127303824393 / 49829197408,
            -318862633887 / 49829197408,
            701980252875 / 199316789632,
        ],
        [0.0, -282668133 / 205662961, 2019193451 / 616988883, -1453857185 / 822651844],
        [0.0, 40617522 / 29380423, -110615467 / 29380423, 69997945 / 29380423],
    ]
)


@njit(cache=True)
def _poly(c, t):
    return c[0] + t * (c[1] + t * (c[2] + t * (c[3] + t * c[4])))


@njit(cache=True)
def _cubic(a, b, c, d, t):
    return d + t * (c + t * (b + t * a))


@njit(cache=True)
def _sort_small(a, n):
    # Fixed-size workspaces avoid heap allocation in the boundary search.
    for i in range(1, n):
        value = a[i]
        j = i - 1
        while j >= 0 and a[j] > value:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = value


@njit(cache=True)
def _stationary_points_into(c, split, roots):
    a, b, cc, d = 4 * c[4], 3 * c[3], 2 * c[2], c[1]
    split[0], split[1] = 0.0, 1.0
    nsplit = 2
    A, B, C = 3 * a, 2 * b, cc
    scale = max(abs(A), abs(B), abs(C))
    if scale > 0:
        A, B, C = A / scale, B / scale, C / scale
        if abs(A) <= 8 * np.finfo(np.float64).eps:
            if B != 0:
                r = -C / B
                if 0 < r < 1:
                    split[nsplit] = r
                    nsplit += 1
        else:
            discriminant = B * B - 4 * A * C
            if discriminant >= 0:
                q = -0.5 * (B + np.copysign(np.sqrt(discriminant), B))
                r1 = q / A
                r2 = C / q if q != 0 else -B / (2 * A)
                for r in (r1, r2):
                    if 0 < r < 1:
                        split[nsplit] = r
                        nsplit += 1
    _sort_small(split, nsplit)
    roots[0], roots[1] = 0.0, 1.0
    count = 2
    for i in range(nsplit - 1):
        lo, hi = split[i], split[i + 1]
        flo, fhi = _cubic(a, b, cc, d, lo), _cubic(a, b, cc, d, hi)
        if flo == 0 and 0 < lo < 1:
            roots[count] = lo
            count += 1
        if fhi == 0 and 0 < hi < 1:
            roots[count] = hi
            count += 1
        if (flo < 0 < fhi) or (fhi < 0 < flo):
            for _ in range(54):
                mid = 0.5 * (lo + hi)
                fm = _cubic(a, b, cc, d, mid)
                if fm == 0:
                    lo, hi = mid, mid
                    break
                if (flo < 0 < fm) or (fm < 0 < flo):
                    hi = mid
                else:
                    lo, flo = mid, fm
            roots[count] = 0.5 * (lo + hi)
            count += 1
    _sort_small(roots, count)
    return count


@njit(cache=True)
def _stationary_points(c):
    split = np.empty(4)
    roots = np.empty(10)
    count = _stationary_points_into(c, split, roots)
    return roots[:count].copy()


@njit(cache=True, forceinline=True)
def _range_proves_inside(c, lower, upper):
    """Enclose the power polynomial using a forward-error bound.

    At most four additions form each sign-separated bound. Their error is
    <= gamma_4*sum(abs(c)), gamma_4 < 8*eps. A 16*eps margin also covers
    rounding of the magnitude/bounds and subnormals (normal-tiny floor).
    A failed proof requests full extrema/root isolation; it is never an exit.
    """
    low, high = c[0], c[0]
    magnitude = abs(c[0])
    for k in range(1, 5):
        value = c[k]
        if not np.isfinite(value):
            return False
        magnitude += abs(value)
        if value < 0:
            low += value
        elif value > 0:
            high += value
    if not (np.isfinite(low) and np.isfinite(high) and np.isfinite(magnitude)):
        return False
    margin = (
        16
        * np.finfo(np.float64).eps
        * max(magnitude, abs(lower), abs(upper), np.finfo(np.float64).tiny)
    )
    return low - margin > lower and high + margin < upper


@njit(cache=True)
def _outward_root(c, bound, outward, noise, points, count):
    for i in range(count - 1):
        lo, hi = points[i], points[i + 1]
        flo = outward * (_poly(c, lo) - bound)
        fhi = outward * (_poly(c, hi) - bound)
        if fhi <= noise:
            continue
        # The first interval ending outside contains the first crossing,
        # including a boundary seed whose outward derivative is positive.
        if flo >= 0:
            return lo
        for _ in range(54):
            mid = 0.5 * (lo + hi)
            if outward * (_poly(c, mid) - bound) > 0:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)
    return 2.0


@njit(cache=True)
def _first_face_crossing_into(c, lower, upper, split, points):
    if _range_proves_inside(c, lower, upper):
        return 2.0, 0
    count = _stationary_points_into(c, split, points)
    magnitude = abs(lower) + abs(upper)
    for k in range(5):
        magnitude += abs(c[k])
    noise = 64 * np.finfo(np.float64).eps * max(magnitude, np.finfo(np.float64).tiny)
    first, side = 2.0, 0
    for bound, outward in ((lower, -1), (upper, 1)):
        root = _outward_root(c, bound, outward, noise, points, count)
        if root < first:
            first, side = root, outward
    return first, side


@njit(cache=True)
def _close_face_crossing_into(c, bound, outward, split, points):
    """Resolve a roundoff-close endpoint using displacement from its face."""
    origin = c[0]
    c[0] = origin - bound
    magnitude = abs(c[0]) + abs(c[1]) + abs(c[2]) + abs(c[3]) + abs(c[4])
    noise = 64 * np.finfo(np.float64).eps * max(magnitude, np.finfo(np.float64).tiny)
    count = _stationary_points_into(c, split, points)
    root = _outward_root(c, 0.0, outward, noise, points, count)
    c[0] = origin
    return root


@njit(cache=True)
def _first_face_crossing(c, lower, upper):
    return _first_face_crossing_into(c, lower, upper, np.empty(4), np.empty(10))


@njit(cache=True)
def _dense_first_exit_into(
    origin,
    stages,
    h,
    xgrid,
    ygrid,
    zgrid,
    periodic_x,
    periodic_y,
    periodic_z,
    c,
    split,
    roots,
    endpoint=None,
    boundary_tol=0.0,
):
    first, code = 2.0, 0
    for axis in range(3):
        if (periodic_x, periodic_y, periodic_z)[axis]:
            continue
        grid = (xgrid, ygrid, zgrid)[axis]
        c[0] = origin[axis]
        for power in range(4):
            coefficient = 0.0
            for stage in range(7):
                coefficient += stages[stage, axis] * _DP_P[stage, power]
            c[power + 1] = h * coefficient
        fraction, side = _first_face_crossing_into(c, grid[0], grid[-1], split, roots)
        if side == 0 and endpoint is not None:
            # The stages may have passed via coordinate roundoff clamping.
            # Require an outward crossing in the dense polynomial itself.
            normal = stages[6, axis]
            if (
                endpoint[axis] <= grid[0]
                and h * normal < 0
                and (
                    origin[axis] == grid[0]
                    or abs((endpoint[axis] - grid[0]) / normal) > boundary_tol
                )
            ):
                fraction = _close_face_crossing_into(c, grid[0], -1, split, roots)
                side = -1 if fraction <= 1 else 0
            elif (
                endpoint[axis] >= grid[-1]
                and h * normal > 0
                and (
                    origin[axis] == grid[-1]
                    or abs((endpoint[axis] - grid[-1]) / normal) > boundary_tol
                )
            ):
                fraction = _close_face_crossing_into(c, grid[-1], 1, split, roots)
                side = 1 if fraction <= 1 else 0
        if side != 0 and fraction < first - 64 * np.finfo(np.float64).eps:
            first = fraction
            code = (5, 3, 1)[axis] + (1 if side > 0 else 0)
    return first, code


@njit(cache=True)
def _dense_first_exit(
    origin, stages, h, xgrid, ygrid, zgrid, periodic_x, periodic_y, periodic_z
):
    return _dense_first_exit_into(
        origin,
        stages,
        h,
        xgrid,
        ygrid,
        zgrid,
        periodic_x,
        periodic_y,
        periodic_z,
        np.empty(5),
        np.empty(4),
        np.empty(10),
    )
