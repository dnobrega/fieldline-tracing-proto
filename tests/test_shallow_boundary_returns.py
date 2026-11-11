"""Boundary starts with cancellation and genuine shallow RK45 returns."""

import numpy as np
import pytest

from fieldline_tracing_proto import trace_field_lines


def trace_face_seed(x, z, seed_x, side, normal_field, dtype, method, **controls):
    y = np.array([-1.0, 1.0])
    bx = np.ones((x.size, y.size, z.size), dtype=dtype)
    bz = np.empty_like(bx)
    bz[:] = normal_field[:, None, None]
    face = z[0] if side < 0 else z[-1]
    return trace_field_lines(
        bx,
        np.zeros_like(bx),
        bz,
        x,
        y,
        z,
        [[seed_x, 0.0, face]],
        0.013,
        1000,
        method=method,
        periodic_x=False,
        periodic_y=False,
        periodic_z=False,
        dtype=dtype,
        max_length=0.06,
        **controls,
    )


@pytest.mark.parametrize("method", ["rk2", "rk45"])
@pytest.mark.parametrize(
    "dtype,n,seed_x",
    [
        (np.float32, 17, np.sqrt(2.0) / 10),
        (np.float64, 31, np.sqrt(3.0) / 10),
    ],
)
@pytest.mark.parametrize("curves_out", [True, False])
def test_off_node_tangency_with_interpolation_cancellation(
    method, dtype, n, seed_x, curves_out
):
    x = np.linspace(-1, 1, n)
    z = np.linspace(-1, 1, 17)
    normal = (-2 if curves_out else 2) * (x - seed_x)
    r = trace_face_seed(x, z, seed_x, -1, normal, dtype, method)
    if curves_out:
        assert (r.exit_code_minus[0], r.exit_code_plus[0]) == (-1, 1)
        assert r.length_minus[0] == r.length_plus[0] == 0
        assert r.nsteps_minus[0] == r.nsteps_plus[0] == 0
    else:
        assert (r.exit_code_minus[0], r.exit_code_plus[0]) == (-11, 11)
        assert r.endpoints_minus[0, 2] > z[0]
        assert r.endpoints_plus[0, 2] > z[0]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("face", [0.0, 1.0])
@pytest.mark.parametrize("slope", [1e-9, 1e-5, 1e-4])
def test_rk45_shallow_return_matches_analytic_arc(dtype, side, face, slope):
    x = np.linspace(-1, 1, 129)
    z = face + np.linspace(0, 2, 129) - (2 if side > 0 else 0)
    normal = -side * (slope - 2 * x)
    r = trace_face_seed(x, z, 0.0, side, normal, dtype, "rk45")
    code = 1 if side < 0 else 2
    assert (r.exit_code_minus[0], r.exit_code_plus[0]) == (-code, code)
    assert r.length_minus[0] == 0
    exact_arc = 0.5 * (slope * np.hypot(1, slope) + np.arcsinh(slope))
    assert r.length_plus[0] == pytest.approx(exact_arc, rel=5e-5, abs=0)
    assert r.endpoints_plus[0, 0] == pytest.approx(slope, rel=5e-5, abs=0)
    assert r.endpoints_plus[0, 2] == face
    assert np.all((z[0] <= r.zfl) & (r.zfl <= z[-1]))


@pytest.mark.parametrize("method", ["rk2", "rk45"])
def test_shallow_straight_line_enters_instead_of_stopping_at_seed_face(method):
    g = np.linspace(-1, 1, 17)
    slope = 1e-9
    r = trace_face_seed(g, g, 0.0, -1, np.full_like(g, slope), np.float64, method)
    assert (r.exit_code_minus[0], r.exit_code_plus[0]) == (-1, 11)
    assert r.length_plus[0] == 0.06
    assert r.endpoints_plus[0, 2] > g[0]
