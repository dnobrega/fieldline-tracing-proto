"""Regression checks for shared interpolation and two-pass branch storage."""

import numpy as np
import pytest

from fieldline_tracing_proto.grid import _wrap_coord
from fieldline_tracing_proto.sampling import sample_scalar_on_lines
from fieldline_tracing_proto.tracing import ragged_to_padded, trace_field_lines


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("stride", [1, 3, 20])
def test_unequal_branch_lengths_and_final_points(dtype, stride):
    grid = np.linspace(-1.0, 1.0, 5, dtype=dtype)
    zeros = np.zeros((5, 5, 5), dtype=dtype)
    seeds = np.array(
        [[0.0, 0.0, -0.875], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype
    )
    result = trace_field_lines(
        zeros,
        zeros,
        zeros + 1,
        grid,
        grid,
        grid,
        seeds,
        0.125,
        10,
        save_stride=stride,
        dtype=dtype,
        periodic_convention="endpoint_included",
    )
    np.testing.assert_array_equal(result.nsteps_minus, [1, 8, 10])
    np.testing.assert_array_equal(result.nsteps_plus, [10, 8, 0])
    for i in range(3):
        m, p = result.nsteps_minus[i], result.nsteps_plus[i]
        saved_m = np.unique(np.r_[0, np.arange(stride, m + 1, stride), m])
        saved_p = np.unique(np.r_[0, np.arange(stride, p + 1, stride), p])
        expected = np.r_[-saved_m[:0:-1], saved_p] * 0.125
        x, y, z, s = result.get_line(i)
        np.testing.assert_allclose(z, seeds[i, 2] + expected)
        np.testing.assert_allclose(s, expected + m * 0.125)
        assert result.seed_index[i] == result.line_start[i] + len(saved_m) - 1
    x, y, z, s = ragged_to_padded(result)
    for i, count in enumerate(result.line_count):
        np.testing.assert_array_equal(z[i, :count], result.get_line(i)[2])
        assert np.all(np.isnan(z[i, count:]))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_curved_trace_and_affine_sampling_across_shared_stencil(dtype):
    grid = np.array([-2.0, -0.8, -0.1, 0.4, 1.0, 2.0], dtype=dtype)
    x, y, z = np.meshgrid(grid, grid, grid, indexing="ij")
    seeds = np.array([[0.7, 0.0, 0.0], [1.0, 0.0, 0.2]], dtype=dtype)
    result = trace_field_lines(
        -y,
        x,
        np.full_like(x, 0.3),
        grid,
        grid,
        grid,
        seeds,
        0.025,
        37,
        save_stride=4,
        dtype=dtype,
        periodic_convention="endpoint_included",
    )
    sample = sample_scalar_on_lines(
        result, 2 * x - y + 3 * z, grid, grid, grid, dtype=dtype
    )
    np.testing.assert_array_equal(sample.codes_all, 0)
    np.testing.assert_allclose(
        sample.values_all,
        2 * result.xfl - result.yfl + 3 * result.zfl,
        atol=2e-6,
        rtol=2e-6,
    )
    assert np.all(np.isfinite(result.sfl))


@pytest.mark.parametrize(
    "q,expected",
    [(-3.0, -1.0), (3.0, 1.0), (-1.0, -1.0), (1.0, 1.0), (-1.25, 0.75), (1.25, -0.75)],
)
def test_periodic_endpoint_convention(q, expected):
    assert _wrap_coord(q, -1.0, 1.0) == expected
