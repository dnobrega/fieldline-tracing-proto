"""Adaptive solver checks, including an independent SciPy reference."""

import json

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator

from fieldline_tracing_proto.tracing import (
    load_trace_npz,
    save_trace_npz,
    trace_field_lines,
)


def run(field="circle", dtype=np.float64, **options):
    grid = np.linspace(-3.0, 3.0, 9)
    x, y, z = np.meshgrid(grid, grid, grid, indexing="ij")
    bx, by, bz = -y, x, np.zeros_like(x)
    if field == "straight":
        bx, by, bz = np.zeros_like(x), np.zeros_like(x), np.ones_like(x)
    if field == "helix":
        bz = np.ones_like(x) * 0.5
    opts = dict(
        ds=0.8,
        max_steps=1000,
        method="rk45",
        dtype=dtype,
        max_length=2.0,
        periodic_x=False,
        periodic_y=False,
        seeds=np.array([[1.0, 0.0, 0.0]]),
    )
    opts.update(options)
    return trace_field_lines(
        bx, by, bz, grid, grid, grid, **opts, periodic_convention="endpoint_included"
    )


@pytest.mark.parametrize("dtype,tol", [(np.float32, 2e-6), (np.float64, 2e-8)])
@pytest.mark.parametrize("stride", [1, 3, 2000])
def test_circle_both_branches_and_saving(dtype, tol, stride):
    r = run(dtype=dtype, save_stride=stride, rtol=1e-9, atol=1e-11)
    assert r.method == "rk45"
    assert r.xfl.dtype == dtype
    np.testing.assert_array_equal(r.exit_code_minus, [-11])
    np.testing.assert_array_equal(r.exit_code_plus, [11])
    np.testing.assert_allclose(r.length_minus, 2.0)
    np.testing.assert_allclose(r.length_plus, 2.0)
    signed_s = r.sfl - 2.0
    np.testing.assert_allclose(r.xfl, np.cos(signed_s), atol=tol, rtol=0)
    np.testing.assert_allclose(r.yfl, np.sin(signed_s), atol=tol, rtol=0)
    assert np.all(np.diff(r.sfl) > 0)
    assert r.sfl[0] == 0 and r.sfl[-1] == 4
    assert r.nrejected_plus[0] > 0
    assert r.nfev_plus[0] == 1 + 6 * (r.nsteps_plus[0] + r.nrejected_plus[0])
    np.testing.assert_array_equal(
        r.seeds[0],
        [r.xfl[r.seed_index[0]], r.yfl[r.seed_index[0]], r.zfl[r.seed_index[0]]],
    )


def test_tighter_tolerance_improves_accuracy():
    errors = []
    for tol in (1e-3, 1e-6, 1e-9):
        r = run(rtol=tol, atol=tol * 0.01)
        errors.append(np.hypot(r.xfl[-1] - np.cos(2), r.yfl[-1] - np.sin(2)))
    assert errors[2] < errors[1] / 10 < errors[0] / 100


def test_helix_and_step_bound():
    r = run(field="helix", rtol=1e-10, atol=1e-12, max_step=0.2)
    theta = (r.sfl - 2.0) / np.sqrt(1.25)
    np.testing.assert_allclose(r.xfl, np.cos(theta), atol=1e-9)
    np.testing.assert_allclose(r.yfl, np.sin(theta), atol=1e-9)
    np.testing.assert_allclose(r.zfl, 0.5 * theta, atol=1e-9)
    assert np.max(np.diff(r.sfl)) <= 0.2 + 1e-14


def test_periodic_seams_do_not_create_error_rejections():
    grid = np.linspace(0.0, 1.0, 5)
    zeros = np.zeros((5, 5, 5))
    r = trace_field_lines(
        zeros + 1,
        zeros,
        zeros,
        grid,
        grid,
        grid,
        np.array([[0.9, 0.5, 0.5]]),
        0.3,
        100,
        method="rk45",
        max_length=3.0,
        max_step=0.4,
        dtype=np.float64,
        rtol=1e-10,
        atol=1e-12,
        periodic_convention="endpoint_included",
    )
    signed_s = r.sfl - 3.0
    np.testing.assert_allclose(r.xfl, (0.9 + signed_s) % 1.0, atol=1e-14)
    np.testing.assert_array_equal(r.nrejected_plus, [0])
    np.testing.assert_array_equal(r.nrejected_minus, [0])


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("stride", [1, 1000])
def test_boundary_localization(axis, stride):
    grid = np.linspace(-1.0, 1.0, 5)
    fields = [np.zeros((5, 5, 5)) for _ in range(3)]
    fields[axis] += 1.0
    r = trace_field_lines(
        *fields,
        grid,
        grid,
        grid,
        np.zeros((1, 3)),
        0.7,
        1000,
        method="rk45",
        max_length=5.0,
        periodic_x=False,
        periodic_y=False,
        periodic_z=False,
        boundary_tol=1e-8,
        dtype=np.float64,
        save_stride=stride,
        periodic_convention="endpoint_included",
    )
    coordinates = (r.xfl, r.yfl, r.zfl)[axis]
    assert -1 <= coordinates[0] <= -1 + 1e-8
    assert 1 - 1e-8 <= coordinates[-1] <= 1
    np.testing.assert_allclose(r.length_plus, [1.0], atol=1e-8)
    code = (6, 4, 2)[axis]
    assert r.exit_code_plus[0] == code
    assert r.exit_code_minus[0] == -(code - 1)


@pytest.mark.parametrize(
    "options,code",
    [
        ({"max_steps": 1}, 7),
        ({"max_attempts": 1, "rtol": 1e-12, "atol": 1e-14}, 13),
        ({"min_step": 0.7, "rtol": 1e-12, "atol": 1e-14}, 12),
    ],
)
def test_bounded_failure_paths(options, code):
    r = run(**options)
    assert r.exit_code_plus[0] == code
    assert r.exit_code_minus[0] == -code
    assert np.all(np.isfinite(r.xfl))
    assert r.line_count[0] == r.nsteps_minus[0] + 1 + r.nsteps_plus[0]


@pytest.mark.parametrize("invalid", [0.0, np.nan, np.inf])
def test_weak_and_invalid_field_stops(invalid):
    grid = np.linspace(-1.0, 1.0, 3)
    zeros = np.zeros((3, 3, 3))
    r = trace_field_lines(
        zeros + invalid,
        zeros,
        zeros,
        grid,
        grid,
        grid,
        np.zeros((1, 3)),
        0.1,
        100,
        method="rk45",
        periodic_convention="endpoint_included",
    )
    code = 8 if invalid == 0 else 9
    assert r.exit_code_plus[0] == code
    assert r.line_count[0] == 1
    assert r.nfev_plus[0] == 1


def test_independent_reference_on_same_nonuniform_interpolated_field():
    grid = np.array([-3.0, -1.3, -0.4, 0.2, 0.9, 1.7, 3.0])
    x, y, z = np.meshgrid(grid, grid, grid, indexing="ij")
    fields = (-y + 0.08 * x * x, x + 0.03 * y * z, np.full_like(x, 0.4))
    seed = np.array([1.0, 0.1, -0.1])
    r = trace_field_lines(
        *fields,
        grid,
        grid,
        grid,
        seed[None],
        0.5,
        1000,
        method="rk45",
        dtype=np.float64,
        max_length=2.0,
        periodic_x=False,
        periodic_y=False,
        rtol=1e-9,
        atol=1e-11,
        periodic_convention="endpoint_included",
    )
    interp = RegularGridInterpolator((grid, grid, grid), np.stack(fields, axis=-1))
    for sign, endpoint in ((-1.0, 0), (1.0, -1)):

        def rhs(s, pos):
            b = interp(pos)[0]
            return sign * b / np.linalg.norm(b)

        ref = solve_ivp(rhs, (0.0, 2.0), seed, rtol=1e-11, atol=1e-13, max_step=0.02)
        assert ref.success
        np.testing.assert_allclose(
            [r.xfl[endpoint], r.yfl[endpoint], r.zfl[endpoint]],
            ref.y[:, -1],
            atol=3e-7,
            rtol=0,
        )


def test_npz_metadata_and_legacy_loading(tmp_path):
    r = run()
    path = tmp_path / "rk45.npz"
    save_trace_npz(path, r)
    restored = load_trace_npz(path)
    for name in vars(r):
        np.testing.assert_array_equal(getattr(r, name), getattr(restored, name))
    settings = json.loads(restored.settings_json)
    assert settings["max_length"] == 2
    legacy = {
        k: v
        for k, v in vars(r).items()
        if k
        not in (
            "method",
            "settings_json",
            "nrejected_minus",
            "nrejected_plus",
            "nfev_minus",
            "nfev_plus",
        )
    }
    np.savez(path, **legacy)
    restored = load_trace_npz(path)
    assert restored.method == "rk2"
    assert restored.nfev_plus is None
    np.testing.assert_array_equal(restored.xfl, r.xfl)


@pytest.mark.parametrize(
    "options",
    [
        {"method": "rk4"},
        {"rtol": 0},
        {"atol": np.nan},
        {"max_step": 0},
        {"max_length": -1},
        {"min_step": 2, "max_step": 1},
        {"max_attempts": 1.5},
        {"boundary_tol": 0},
    ],
)
def test_invalid_options(options):
    with pytest.raises(ValueError):
        run(**options)


def test_empty_seeds():
    r = run(seeds=np.empty((0, 3)))
    assert r.total_points == r.nlines == 0
    assert r.nrejected_plus.size == 0


def test_curved_boundary_crossing():
    horizontal = np.linspace(-2.0, 2.0, 9)
    vertical = np.linspace(-0.2, 0.2, 5)
    x, y, z = np.meshgrid(horizontal, horizontal, vertical, indexing="ij")
    r = trace_field_lines(
        -y,
        x,
        np.full_like(x, 0.3),
        horizontal,
        horizontal,
        vertical,
        np.array([[1.0, 0.0, 0.0]]),
        0.8,
        1000,
        method="rk45",
        dtype=np.float64,
        periodic_x=False,
        periodic_y=False,
        rtol=1e-10,
        atol=1e-12,
        boundary_tol=1e-9,
        periodic_convention="endpoint_included",
    )
    length = 0.2 * np.sqrt(1.09) / 0.3
    np.testing.assert_allclose(r.length_plus, [length], atol=2e-9)
    np.testing.assert_allclose(r.length_minus, [length], atol=2e-9)
    np.testing.assert_allclose(
        [r.xfl[-1], r.yfl[-1], r.zfl[-1]],
        [np.cos(2 / 3), np.sin(2 / 3), 0.2],
        atol=2e-9,
    )
    assert r.exit_code_plus[0] == 2 and r.exit_code_minus[0] == -1


def test_final_length_clip_can_be_smaller_than_min_step():
    r = run(field="straight", ds=0.5, max_step=0.5, min_step=0.2, max_length=0.51)
    np.testing.assert_allclose(r.length_plus, [0.51])
    assert r.nsteps_plus[0] == 2
    assert r.exit_code_plus[0] == 11


def test_no_arc_length_progress():
    # Tiny steps cannot change a coordinate near 1 in float64.
    r = run(
        field="straight", ds=1e-20, min_step=1e-25, seeds=np.array([[0.0, 0.0, 1.0]])
    )
    assert r.exit_code_plus[0] == 12
    assert r.line_count[0] == 1


def test_rk2_rejects_nondefault_adaptive_controls():
    with pytest.raises(ValueError, match="Adaptive controls"):
        run(method="rk2", rtol=1e-6)
