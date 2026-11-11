"""Independent polynomial and continuous-extension checks for boundary search."""

import numpy as np
import pytest
from scipy.integrate import RK45

from fieldline_tracing_proto.events import (
    _DP_P,
    _first_face_crossing,
    _range_proves_inside,
    _stationary_points,
)
from fieldline_tracing_proto.tracing import trace_field_lines


@pytest.mark.parametrize("width", [1e-2, 1e-3, 1e-5])
@pytest.mark.parametrize("outward", [-1, 1])
def test_narrow_excursion_between_inside_endpoints(width, outward):
    # p(t) = 1 + width**2 - (t-0.47)**2, reflected for the lower face.
    c = outward * np.array([1 + width**2 - 0.47**2, 0.94, -1.0, 0.0, 0.0])
    fraction, side = _first_face_crossing(c, -1.0, 1.0)
    assert side == outward
    assert fraction == pytest.approx(0.47 - width, abs=2e-10)


def test_pure_contact_does_not_count_as_exit():
    c = np.array([0.75, 1.0, -1.0, 0.0, 0.0])
    _, side = _first_face_crossing(c, -1.0, 1.0)
    assert side == 0


def test_quartic_exit_and_reentry_uses_first_root():
    # p(t) = 16*1.1*t**2*(1-t)**2: both endpoints are at 0, maximum 1.1.
    c = np.array([0.0, 0.0, 17.6, -35.2, 17.6])
    fraction, side = _first_face_crossing(c, -1.0, 1.0)
    expected = 0.5 * (1 - np.sqrt(1 - 1 / np.sqrt(1.1)))
    assert side == 1
    assert fraction == pytest.approx(expected, abs=1e-13)


def test_flat_face_path_and_boundary_outward_start():
    assert _first_face_crossing(np.array([1.0, 0.0, 0.0, 0.0, 0.0]), -1.0, 1.0)[1] == 0
    assert _first_face_crossing(np.array([1.0, 1.0, 0.0, 0.0, 0.0]), -1.0, 1.0) == (
        0.0,
        1,
    )


def test_interior_range_can_skip_detailed_search():
    c = np.array([0.1, 0.2, -0.1, 0.03, -0.01])
    assert _range_proves_inside(c, -1.0, 1.0)
    assert _first_face_crossing(c, -1.0, 1.0) == (2.0, 0)


@pytest.mark.parametrize(
    "c",
    [
        [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],  # Do not certify a boundary seed as strictly interior.
        [np.nextafter(1.0, 0.0), 1e-16, 0.0, 0.0, 0.0],  # Uncertain at roundoff scale.
        [0.75 + 1e-6, 1.0, -1.0, 0.0, 0.0],  # Inside endpoints, outside in between.
        [0.0, 4.0, -4.0, 0.0, 0.0],  # Loose bound: fallback must preserve pure contact.
        [0.0, np.inf, 0.0, 0.0, 0.0],
    ],
)
def test_uncertain_ranges_fall_back_to_search(c):
    assert not _range_proves_inside(np.array(c), -1.0, 1.0)


@pytest.mark.parametrize("scale", [1e-100, 1.0, 1e100])
def test_range_filter_certificates_against_independent_extrema(scale):
    rng = np.random.default_rng(82)
    certified = 0
    for _ in range(200):
        c = np.r_[
            rng.uniform(-0.8, 0.8),
            rng.normal(size=4) * rng.choice([0.01, 0.1, 1.0, 10.0]),
        ]
        if _range_proves_inside(c * scale, -scale, scale):
            certified += 1
            # Normalize before finding extrema to avoid ill-scaled polynomial roots.
            roots = np.polynomial.polynomial.polyroots(np.arange(1, 5) * c[1:])
            points = [0.0, 1.0] + [
                r.real for r in roots if abs(r.imag) < 1e-10 and 0 < r.real < 1
            ]
            values = np.polynomial.polynomial.polyval(points, c)
            assert min(values) > -1 and max(values) < 1
    assert certified > 50  # Exercise successful certificates, not only fallback.


def test_cubic_stationary_roots_against_independent_polynomial_solver():
    rng = np.random.default_rng(52)
    for _ in range(100):
        c = rng.normal(size=5)
        reference = np.polynomial.polynomial.polyroots(np.arange(1, 5) * c[1:])
        real = sorted(
            r.real for r in reference if abs(r.imag) < 1e-10 and 0 < r.real < 1
        )
        actual = _stationary_points(c)
        np.testing.assert_allclose(actual, [0.0, *real, 1.0], atol=2e-12, rtol=0)


def test_first_crossing_against_independent_quartic_roots():
    rng = np.random.default_rng(113)
    for _ in range(100):
        c = np.r_[0.0, rng.normal(size=4) * 5]
        candidates = []
        for bound, outward in ((-1.0, -1), (1.0, 1)):
            shifted = c.copy()
            shifted[0] -= bound
            for root in np.polynomial.polynomial.polyroots(shifted):
                if abs(root.imag) < 1e-10 and 0 < root.real < 1:
                    derivative = np.polynomial.polynomial.polyval(
                        root.real, np.arange(1, 5) * c[1:]
                    )
                    if outward * derivative > 0:
                        candidates.append((root.real, outward))
        actual = _first_face_crossing(c, -1.0, 1.0)
        if candidates:
            expected = min(candidates)
            assert actual[0] == pytest.approx(expected[0], abs=2e-12)
            assert actual[1] == expected[1]
        else:
            assert actual[1] == 0


def test_continuous_extension_matches_scipy_at_interior_points():
    # Coefficients are independently exercised against SciPy's dense output,
    # beyond matching the fifth-order endpoint used by both implementations.
    def rhs(t, y):
        return np.array([-y[1], y[0], 0.3]) / np.linalg.norm([-y[1], y[0], 0.3])

    solver = RK45(
        rhs, 0.0, np.array([1.0, 0.0, 0.0]), 0.2, first_step=0.2, rtol=1e-5, atol=1e-8
    )
    solver.step()
    dense = solver.dense_output()
    for fraction in (0.0, 0.13, 0.5, 0.81, 1.0):
        powers = fraction ** np.arange(1, 5)
        ours = solver.y_old + solver.h_previous * (solver.K.T @ _DP_P) @ powers
        np.testing.assert_allclose(
            ours, dense(fraction * solver.h_previous), atol=1e-14
        )


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("stride", [1, 1000])
def test_excursion_on_each_axis_both_faces_and_branches(axis, sign, stride):
    # Place the parabolic normal coordinate on each axis; reflect the face.
    tangent = (axis + 1) % 3
    grid = np.linspace(-1.0, 1.0, 9)
    coords = np.meshgrid(grid, grid, grid, indexing="ij")
    xyz = list(coords)
    fields = [np.zeros_like(xyz[0]) for _ in range(3)]
    fields[axis] = -sign * 2 * xyz[tangent]
    fields[tangent] = np.ones_like(xyz[0])
    seed = np.zeros((2, 3))
    seed[:, axis] = sign * 0.750001
    seed[:, tangent] = [-0.5, 0.5]
    r = trace_field_lines(
        *fields,
        grid,
        grid,
        grid,
        seed,
        0.3,
        10000,
        method="rk45",
        dtype=np.float64,
        max_length=1.3,
        periodic_x=False,
        periodic_y=False,
        periodic_z=False,
        rtol=1e-9,
        atol=1e-11,
        boundary_tol=1e-10,
        min_step=1e-14,
        save_stride=stride,
        periodic_convention="endpoint_included",
    )
    code = (5, 3, 1)[axis] + (sign > 0)
    assert r.exit_code_plus[0] == code
    assert r.exit_code_minus[1] == -code
    assert r.length_plus[0] == pytest.approx(0.5728967866814932, abs=1e-7)
    assert r.length_minus[1] == pytest.approx(0.5728967866814932, abs=1e-7)
    endpoint = r.get_line(0)[axis][-1]
    assert abs(endpoint - sign) < 1e-9
