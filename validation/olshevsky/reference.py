"""Direct point evaluation, independent of the gridded trilinear tracer."""

import numpy as np
from numba import njit
from .field import PARAMS_A1, EXACT_NULLS


@njit(cache=True)
def field_at(p):
    x, y, z = p
    sx, cx, sy, sz, cz = np.sin(x), np.cos(x), np.sin(y), np.sin(z), np.cos(z)
    s2y = np.sin(2 * y)
    b = np.array(
        [
            -2 * s2y + sz + 0.04 * (-2 * s2y + 2 * sx * cz + sy + 0.1195),
            2 * sx + sz + 0.04 * (3 * sx * cz - cx * sz + np.sqrt(30.0) / 7),
            sx + sy + 0.04 * (-2 * cx * sz + sy + 2 * s2y - 0.1378162),
        ]
    )
    for r in range(len(PARAMS_A1)):
        a, kappa, ell, X, Y, Z = PARAMS_A1[r]
        wx = a if r < 5 else ell
        wy = a if r < 2 or r >= 5 else ell
        wz = ell if r < 2 else a
        dx, dy, dz = x - X, y - Y, z - Z
        g = np.exp(-((dx / wx) ** 2) - (dy / wy) ** 2 - (dz / wz) ** 2)
        c = 2 * a * kappa / a**2 * g
        if r < 2:
            b[0] -= c * dy
            b[1] += c * dx
        elif r < 5:
            b[0] += c * dz
            b[2] -= c * dx
        else:
            b[1] -= c * dz
            b[2] += c * dy
    return b


def refined_nulls():
    from scipy.optimize import root

    roots = []
    for initial in EXACT_NULLS:
        fit = root(field_at, initial, tol=1e-11)
        # Solver progress flags can be pessimistic at floating-point precision.
        if np.linalg.norm(field_at(fit.x)) > 1e-10:
            raise RuntimeError("Analytic null refinement failed")
        roots.append(fit.x)
    roots = np.asarray(roots)
    if (
        np.min(
            np.linalg.norm(roots[:, None] - roots[None, :], axis=2) + np.eye(14) * 100
        )
        < 1e-5
    ):
        raise RuntimeError("Null refinement produced duplicate roots")
    return roots


def make_seeds(count, near_per_null, radius):
    from .field import DOMAIN

    roots = refined_nulls()
    rng = np.random.default_rng(20261109)
    low, high = DOMAIN
    general = rng.uniform(low + 0.1, high - 0.1, size=(count, 3))
    i = np.arange(near_per_null)
    zz = 1 - 2 * (i + 0.5) / near_per_null
    phi = i * np.pi * (3 - np.sqrt(5.0))
    directions = np.column_stack(
        (np.sqrt(1 - zz * zz) * np.cos(phi), np.sqrt(1 - zz * zz) * np.sin(phi), zz)
    )
    near = (roots[:, None, :] + radius * directions[None, :, :]).reshape(-1, 3)
    if np.any(near <= low) or np.any(near >= high):
        raise ValueError("Near-null radius places seeds outside the domain")
    return general, near, roots


def compare_reference(result, grids, fields, ids, mode, max_step):
    """Errors at stored arc positions; no matched-error speed claim."""
    from scipy.integrate import solve_ivp
    from scipy.interpolate import RegularGridInterpolator

    evaluators = {"analytic": field_at}
    if mode == "both":
        interpolators = [
            RegularGridInterpolator(grids, a, bounds_error=False, fill_value=None)
            for a in fields
        ]

        def sampled(p):
            return np.array([a(p[None, :])[0] for a in interpolators])

        evaluators["sampled"] = sampled
    records = []
    for idx in ids:
        x, y, z, s = result.get_line(int(idx))
        signed = s - result.length_minus[idx]
        points = np.column_stack((x, y, z))
        for name, evaluate in evaluators.items():
            largest = 0.0
            nfev = 0
            for sign in (-1, 1):
                mask = signed * sign >= 0
                targets = signed[mask] * sign
                if not len(targets) or targets.max() <= 0:
                    continue

                def rhs(t, p):
                    b = evaluate(p)
                    norm = np.linalg.norm(b)
                    if not np.isfinite(norm) or norm < 1e-14:
                        raise RuntimeError(
                            "Reference reached an unresolved weak-field point"
                        )
                    return sign * b / norm

                fit = solve_ivp(
                    rhs,
                    (0.0, float(targets.max())),
                    result.seeds[idx],
                    method="DOP853",
                    rtol=1e-10,
                    atol=1e-12,
                    max_step=max_step,
                    dense_output=True,
                )
                if not fit.success:
                    raise RuntimeError(fit.message)
                error = np.linalg.norm(fit.sol(targets).T - points[mask], axis=1)
                largest = max(largest, float(error.max()))
                nfev += fit.nfev
            records.append(
                dict(
                    seed_index=int(idx),
                    reference=name,
                    max_position_error=largest,
                    reference_nfev=nfev,
                )
            )
    return records
