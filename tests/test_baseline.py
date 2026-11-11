"""Analytic checks for the original implementation, before solver changes."""

from types import SimpleNamespace

import numpy as np
import pytest

from fieldline_tracing_proto.diagnostics import compute_line_diagnostics
from fieldline_tracing_proto.sampling import (
    sample_ragged_to_padded,
    sample_scalar_on_lines,
)
from fieldline_tracing_proto.tracing import (
    load_trace_npz,
    ragged_to_padded,
    save_trace_npz,
    trace_field_lines,
)
from fieldline_tracing_proto.winding import compute_local_winding


def uniform_trace(dtype=np.float64, stride=1, seeds=None, **kwargs):
    grid = np.array([-2.0, -0.7, 0.0, 0.8, 2.0], dtype=dtype)
    shape = (5, 5, 5)
    if seeds is None:
        seeds = np.array([[0.0, 0.0, 0.0], [0.2, 0.1, 0.3]], dtype=dtype)
    return trace_field_lines(
        np.zeros(shape),
        np.zeros(shape),
        np.ones(shape),
        grid,
        grid,
        grid,
        seeds,
        ds=0.125,
        max_steps=5,
        save_stride=stride,
        dtype=dtype,
        periodic_x=False,
        periodic_y=False,
        **kwargs,
        periodic_convention="endpoint_included",
    ), grid


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("stride", [1, 2, 8])
def test_straight_lines_and_ragged_layout(dtype, stride):
    result, _ = uniform_trace(dtype, stride)
    distances = np.unique(np.r_[0, np.arange(stride, 6, stride), 5]) * 0.125
    offsets = np.r_[-distances[:0:-1], distances]
    assert result.xfl.dtype == dtype
    assert result.total_points == result.nlines * offsets.size
    np.testing.assert_array_equal(result.line_start, [0, offsets.size])
    np.testing.assert_array_equal(result.line_count, offsets.size)
    np.testing.assert_array_equal(result.exit_code_minus, -7)
    np.testing.assert_array_equal(result.exit_code_plus, 7)
    for i in range(result.nlines):
        x, y, z, s = result.get_line(i)
        np.testing.assert_allclose(x, result.seeds[i, 0], atol=1e-6)
        np.testing.assert_allclose(y, result.seeds[i, 1], atol=1e-6)
        np.testing.assert_allclose(z, result.seeds[i, 2] + offsets, atol=1e-6)
        np.testing.assert_allclose(s, offsets + 0.625, atol=1e-6)
        assert np.all(np.diff(s) > 0)
        seed = result.seed_index[i]
        np.testing.assert_allclose(
            [result.xfl[seed], result.yfl[seed], result.zfl[seed]], result.seeds[i]
        )
    np.testing.assert_allclose(result.length_minus, 0.625)
    np.testing.assert_allclose(result.length_plus, 0.625)
    np.testing.assert_array_equal(result.nsteps_plus, 5)


def test_persistence_and_padding(tmp_path):
    result, _ = uniform_trace()
    path = tmp_path / "trace.npz"
    save_trace_npz(path, result)
    restored = load_trace_npz(path)
    for name in vars(result):
        np.testing.assert_array_equal(getattr(restored, name), getattr(result, name))
    for padded, name in zip(ragged_to_padded(result), ("xfl", "yfl", "zfl", "sfl")):
        np.testing.assert_array_equal(padded.ravel(), getattr(result, name))


def test_affine_sampling_and_constant_diagnostics():
    result, grid = uniform_trace()
    x, y, z = np.meshgrid(grid, grid, grid, indexing="ij")
    sample = sample_scalar_on_lines(
        result, 2 * x - 3 * y + z + 4, grid, grid, grid, dtype=np.float64
    )
    np.testing.assert_array_equal(sample.codes_all, 0)
    np.testing.assert_allclose(
        sample.values_all, 2 * result.xfl - 3 * result.yfl + result.zfl + 4
    )
    padded = sample_ragged_to_padded(sample)
    np.testing.assert_array_equal(padded.ravel(), sample.values_all)
    const = sample_scalar_on_lines(
        result, np.full_like(x, 2.0), grid, grid, grid, dtype=np.float64
    )
    diag = compute_line_diagnostics(result, const, dtype=np.float64)
    np.testing.assert_allclose(diag.integral, 2.5)
    np.testing.assert_allclose(diag.integral_abs, 2.5)
    np.testing.assert_allclose(diag.mean, 2.0)
    np.testing.assert_allclose(diag.cancellation, 1.0)
    np.testing.assert_allclose(diag.s_centroid_abs, 0.625)


@pytest.mark.parametrize("ds", [0.13, 0.2])
def test_boundary_reports_exact_partial_endpoint(ds):
    grid = np.linspace(-1, 1, 5)
    zeros = np.zeros((5, 5, 5))
    result = trace_field_lines(
        zeros,
        zeros,
        zeros + 1,
        grid,
        grid,
        grid,
        np.zeros((1, 3)),
        ds,
        100,
        dtype=np.float64,
        periodic_convention="endpoint_included",
    )
    assert result.exit_code_minus[0] == -1
    assert result.exit_code_plus[0] == 2
    assert np.all(np.abs(result.zfl) <= 1.0)
    assert 1.0 - result.zfl[-1] <= ds + 1e-12
    assert result.length_plus[0] == pytest.approx(1.0)
    assert result.zfl[-1] == 1.0


def test_periodic_seam_and_arc_length():
    grid = np.linspace(0.0, 1.0, 5)
    zeros = np.zeros((5, 5, 5))
    result = trace_field_lines(
        zeros + 1,
        zeros,
        zeros,
        grid,
        grid,
        grid,
        np.array([[0.9, 0.5, 0.5]]),
        0.125,
        5,
        dtype=np.float64,
        periodic_convention="endpoint_included",
    )
    np.testing.assert_allclose(result.xfl, (0.9 + np.arange(-5, 6) * 0.125) % 1.0)
    np.testing.assert_allclose(np.diff(result.sfl), 0.125)


def test_empty_seeds():
    result, _ = uniform_trace(seeds=np.empty((0, 3)))
    assert result.nlines == result.total_points == 0
    assert ragged_to_padded(result)[0].shape == (0, 0)


def test_null_field():
    grid = np.linspace(-1.0, 1.0, 3)
    zeros = np.zeros((3, 3, 3))
    result = trace_field_lines(
        zeros,
        zeros,
        zeros,
        grid,
        grid,
        grid,
        np.zeros((1, 3)),
        0.1,
        10,
        periodic_convention="endpoint_included",
    )
    np.testing.assert_array_equal(result.line_count, [1])
    np.testing.assert_array_equal(result.exit_code_minus, [-8])
    np.testing.assert_array_equal(result.exit_code_plus, [8])


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_nonfinite_field_termination(invalid):
    grid = np.linspace(-1.0, 1.0, 3)
    zeros = np.zeros((3, 3, 3))
    result = trace_field_lines(
        zeros + invalid,
        zeros,
        zeros + 1,
        grid,
        grid,
        grid,
        np.zeros((1, 3)),
        0.1,
        10,
        periodic_x=False,
        periodic_y=False,
        dtype=np.float64,
        periodic_convention="endpoint_included",
    )
    np.testing.assert_array_equal(result.line_count, [1])
    np.testing.assert_array_equal(result.exit_code_minus, [-9])
    np.testing.assert_array_equal(result.exit_code_plus, [9])


@pytest.mark.parametrize(
    "option,value", [("ds", 0), ("max_steps", 0), ("save_stride", 0)]
)
def test_existing_parameter_guards(option, value):
    grid = np.linspace(-1.0, 1.0, 3)
    zeros = np.zeros((3, 3, 3))
    options = dict(ds=0.1, max_steps=10, save_stride=1)
    options[option] = value
    with pytest.raises(ValueError):
        trace_field_lines(
            zeros,
            zeros,
            zeros + 1,
            grid,
            grid,
            grid,
            np.zeros((1, 3)),
            **options,
            periodic_convention="endpoint_included",
        )


def test_singleton_axis_rejected():
    grid = np.array([0.0])
    zeros = np.zeros((1, 1, 1))
    with pytest.raises(ValueError):
        trace_field_lines(
            zeros,
            zeros,
            zeros,
            grid,
            grid,
            grid,
            np.zeros((1, 3)),
            0.1,
            2,
            periodic_convention="endpoint_included",
        )


def test_fractional_step_budget_rejected():
    grid = np.linspace(-1.0, 1.0, 3)
    zeros = np.zeros((3, 3, 3))
    with pytest.raises(ValueError):
        trace_field_lines(
            zeros,
            zeros,
            zeros + 1,
            grid,
            grid,
            grid,
            np.zeros((1, 3)),
            0.1,
            2.5,
            periodic_convention="endpoint_included",
        )


def test_midpoint_second_order_on_circle():
    grid = np.linspace(-2.0, 2.0, 9)
    x, y, z = np.meshgrid(grid, grid, grid, indexing="ij")
    errors = []
    for n in (10, 20, 40):
        result = trace_field_lines(
            -y,
            x,
            np.zeros_like(x),
            grid,
            grid,
            grid,
            np.array([[1.0, 0.0, 0.0]]),
            1.0 / n,
            n,
            periodic_x=False,
            periodic_y=False,
            dtype=np.float64,
            periodic_convention="endpoint_included",
        )
        errors.append(
            np.linalg.norm([result.xfl[-1] - np.cos(1.0), result.yfl[-1] - np.sin(1.0)])
        )
    assert 3.5 < errors[0] / errors[1] < 4.5
    assert 3.5 < errors[1] / errors[2] < 4.5


@pytest.mark.parametrize("turns", [0.0, 1.0, -1.0])
def test_winding_known_rotation(turns):
    u = np.linspace(0.0, 1.0, 401)
    theta = 2 * np.pi * turns * u
    result = SimpleNamespace(
        xfl=np.r_[np.zeros_like(u), np.cos(theta)],
        yfl=np.r_[np.zeros_like(u), np.sin(theta)],
        sfl=np.r_[u, u],
        line_start=np.array([0, u.size]),
        line_count=np.array([u.size, u.size]),
    )
    wind = compute_local_winding(
        result,
        (1, 2),
        neighbor_offsets=((0, 1),),
        nsample=401,
        dtype=np.float64,
        parameterization="normalized",
        plane="xy",
    )
    assert wind.mean_wind[0] == pytest.approx(turns, abs=1e-12)
    np.testing.assert_array_equal(wind.n_pairs, [1, 0])
    assert np.isnan(wind.mean_wind[1])
