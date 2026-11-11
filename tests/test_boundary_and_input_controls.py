"""Regressions for boundary starts, step controls, and arc limits."""

import json

import numpy as np
import pytest

from fieldline_tracing_proto import trace_field_lines


def trace(fields, grid, seed, method="rk45", **kwargs):
    return trace_field_lines(
        *fields,
        grid,
        grid,
        grid,
        np.asarray(seed, dtype=float).reshape(-1, 3),
        kwargs.pop("ds", 0.013),
        1000,
        method=method,
        periodic_x=False,
        periodic_y=False,
        periodic_z=False,
        dtype=kwargs.pop("dtype", None),
        **kwargs,
    )


@pytest.mark.parametrize("method", ["rk2", "rk45"])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("storage", ["single_pass", "two_pass"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_exact_tangent_outward_seed_has_zero_arc(method, axis, side, storage, dtype):
    g = np.linspace(-1, 1, 9)
    xyz = np.meshgrid(g, g, g, indexing="ij")
    t = (axis + 1) % 3
    fields = [np.zeros_like(xyz[0]) for _ in range(3)]
    fields[t][:] = 1
    fields[axis] = side * 2 * xyz[t]
    seed = np.zeros(3)
    seed[axis] = side
    fields = [a.astype(dtype) for a in fields]
    r = trace(fields, g, seed, method, storage=storage, max_length=0.2)
    face = (5, 3, 1)[axis] + int(side > 0)
    assert r.exit_code_minus[0] == -face and r.exit_code_plus[0] == face
    assert r.length_minus[0] == r.length_plus[0] == 0
    assert r.nsteps_minus[0] == r.nsteps_plus[0] == 0
    assert r.line_count[0] == 1 and r.sfl[0] == 0
    np.testing.assert_array_equal(r.endpoints_minus[0], seed)
    np.testing.assert_array_equal(r.endpoints_plus[0], seed)


@pytest.mark.parametrize("method", ["rk2", "rk45"])
@pytest.mark.parametrize("kind", ["inward", "flat"])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("side", [-1, 1])
def test_tangent_inward_or_flat_seed_is_not_an_exit(method, kind, axis, side):
    g = np.linspace(-1, 1, 9)
    xyz = np.meshgrid(g, g, g, indexing="ij")
    t = (axis + 1) % 3
    fields = [np.zeros_like(xyz[0]) for _ in range(3)]
    fields[t][:] = 1
    if kind == "inward":
        fields[axis] = -side * 2 * xyz[t]
    seed = np.zeros(3)
    seed[axis] = side
    r = trace(fields, g, seed, method, max_length=0.2)
    assert r.exit_code_minus[0] == -11 and r.exit_code_plus[0] == 11
    assert r.length_minus[0] == r.length_plus[0] == 0.2
    if kind == "inward":
        assert (
            side * r.endpoints_plus[0, axis] < 1
            and side * r.endpoints_minus[0, axis] < 1
        )
    else:
        assert r.endpoints_plus[0, axis] == r.endpoints_minus[0, axis] == side


@pytest.mark.parametrize("method", ["rk2", "rk45"])
@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("nonuniform", [False, True])
def test_tangent_node_uses_the_cell_of_each_branch(method, side, dtype, nonuniform):
    g = (
        np.array([-1.0, -0.7, -0.2, 0.0, 0.3, 0.9, 1.0])
        if nonuniform
        else np.linspace(-1, 1, 9)
    )
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    fields = tuple(a.astype(dtype) for a in (X * 0 + 1, X * 0, side * np.abs(X)))
    r = trace(fields, g, [0, 0, side], method, max_length=0.2)
    assert r.exit_code_plus[0] == (2 if side > 0 else 1)
    assert r.length_plus[0] == 0
    assert r.exit_code_minus[0] == -11 and r.length_minus[0] == 0.2
    assert side * r.endpoints_minus[0, 2] < 1


@pytest.mark.parametrize(
    "attempts", [2**53 + 1, 2**62, int(np.iinfo(np.int64).max), 0, -1, True, 1.5]
)
def test_unrepresentable_or_invalid_attempt_budget_is_rejected(attempts):
    g = np.linspace(-1, 1, 3)
    a = np.ones((3, 3, 3))
    with pytest.raises(ValueError, match="max_attempts"):
        trace((a, a * 0, a * 0), g, [0, 0, 0], max_attempts=attempts)


def test_largest_exact_attempt_budget_is_preserved():
    g = np.linspace(-1, 1, 3)
    a = np.ones((3, 3, 3))
    r = trace(
        (a, a * 0, a * 0), g, [0, 0, 0], max_attempts=np.int64(2**53), max_length=0.2
    )
    assert r.exit_code_plus[0] == 11
    assert json.loads(r.settings_json)["max_attempts"] == 2**53


@pytest.mark.parametrize("method", ["rk2", "rk45"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_zero_dimensional_real_step_matches_scalar(method, dtype):
    g = np.linspace(-1, 1, 3)
    a = np.ones((3, 3, 3))
    ds = np.array(0.01, dtype=dtype)
    r = trace((a, a * 0, a * 0), g, [0, 0, 0], method, ds=ds, max_length=0.2)
    ref = trace((a, a * 0, a * 0), g, [0, 0, 0], method, ds=float(ds), max_length=0.2)
    for name, value in vars(r).items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, getattr(ref, name))


@pytest.mark.parametrize(
    "ds",
    [
        True,
        np.bool_(True),
        np.array(True),
        np.array([0.01]),
        np.array(np.nan),
        np.array(np.inf),
        np.array(0.0),
        np.array(1j),
        np.array("0.01"),
        None,
    ],
)
def test_invalid_step_has_clear_scalar_error(ds):
    g = np.linspace(-1, 1, 3)
    a = np.ones((3, 3, 3))
    with pytest.raises(ValueError, match="ds"):
        trace((a, a * 0, a * 0), g, [0, 0, 0], ds=ds)


@pytest.mark.parametrize("method", ["rk2", "rk45"])
@pytest.mark.parametrize("slope", [0.3, 1.0])
@pytest.mark.parametrize("ds", [0.01, 0.1, 0.13, 0.37])
def test_arc_limit_wins_roundoff_face_tie(method, slope, ds):
    g = np.linspace(-1, 1, 9)
    a = np.ones((9, 9, 9))
    limit = np.hypot(1.0, slope)
    r = trace((a, a * slope, a * 0), g, [0, 0, 0], method, ds=ds, max_length=limit)
    assert r.exit_code_minus[0] == -11 and r.exit_code_plus[0] == 11
    assert r.length_minus[0] == r.length_plus[0] == limit
    assert r.sfl[-1] == 2 * limit
    assert np.all(np.diff(r.sfl) > 0)


@pytest.mark.parametrize("method", ["rk2", "rk45"])
def test_genuinely_earlier_face_is_not_promoted_to_arc_limit(method):
    g = np.linspace(-1, 1, 9)
    a = np.ones((9, 9, 9))
    face_length = np.hypot(1.0, 0.3)
    r = trace(
        (a, a * 0.3, a * 0), g, [0, 0, 0], method, ds=0.1, max_length=face_length + 1e-9
    )
    assert r.exit_code_minus[0] == -5 and r.exit_code_plus[0] == 6
    assert r.length_minus[0] < face_length + 1e-10
    assert r.length_plus[0] < face_length + 1e-10


@pytest.mark.parametrize("method", ["rk2", "rk45"])
@pytest.mark.parametrize("side", [-1, 1])
def test_tangent_curvature_uses_the_periodic_seam_cell(method, side):
    g = np.linspace(-1, 1, 9)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    r = trace_field_lines(
        X * 0 + 1,
        X * 0,
        side * (1 - np.abs(X)),
        g,
        g,
        g,
        np.array([[-1.0, 0.0, float(side)]]),
        0.013,
        1000,
        method=method,
        max_length=0.2,
        dtype=np.float64,
        periodic_x=True,
        periodic_y=False,
        periodic_z=False,
        period=(2.0, 0.0, 0.0),
        periodic_convention="endpoint_included",
    )
    assert r.exit_code_plus[0] == (2 if side > 0 else 1) and r.length_plus[0] == 0
    assert r.exit_code_minus[0] == -11 and r.length_minus[0] == 0.2
    assert side * r.endpoints_minus[0, 2] < 1


@pytest.mark.parametrize("method", ["rk2", "rk45"])
@pytest.mark.parametrize("stride", [2**53 + 1, int(np.iinfo(np.int64).max), 10**400])
def test_huge_stride_keeps_terminal_only_semantics_without_float_overflow(
    method, stride
):
    g = np.linspace(-1, 1, 3)
    a = np.ones((3, 3, 3))
    r = trace(
        (a, a * 0, a * 0), g, [0, 0, 0], method, max_length=0.2, save_stride=stride
    )
    ref = trace(
        (a, a * 0, a * 0), g, [0, 0, 0], method, max_length=0.2, save_stride=1001
    )
    assert json.loads(r.settings_json)["save_stride"] == stride
    assert r.line_count[0] == 3
    for name, value in vars(r).items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, getattr(ref, name))
