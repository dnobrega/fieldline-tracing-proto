"""Safety contracts shared by the tracer and its downstream consumers."""

from types import SimpleNamespace
from typing import get_type_hints

import numpy as np
import pytest

from fieldline_tracing_proto.diagnostics import compute_line_diagnostics
from fieldline_tracing_proto.sampling import sample_scalar_on_lines
from fieldline_tracing_proto.tracing import trace_field_lines


def trace(**overrides):
    grid = np.array([-1.0, 0.0, 1.0])
    zeros = np.zeros((3, 3, 3))
    options = {
        "bx": zeros,
        "by": zeros,
        "bz": zeros + 1,
        "xgrid": grid,
        "ygrid": grid,
        "zgrid": grid,
        "seeds": np.zeros((1, 3)),
        "ds": 0.1,
        "max_steps": 3,
        "dtype": np.float64,
        "periodic_x": True,
        "periodic_y": True,
        "period": (2.0, 2.0, 0.0),
    }
    options.update(overrides)
    return trace_field_lines(**options, periodic_convention="endpoint_included")


@pytest.mark.parametrize(
    "options",
    [
        {"ds": np.nan},
        {"ds": np.inf},
        {"ds": -1},
        {"bmin": np.nan},
        {"bmin": np.inf},
        {"bmin": -1},
        {"max_steps": True},
        {"max_steps": 3.0},
        {"max_steps": 2**31},
        {"save_stride": 1.5},
        {"save_stride": False},
        {"dtype": np.int32},
        {"dtype": np.float16},
        {"seeds": np.array([[np.nan, 0.0, 0.0]])},
        {"seeds": np.array([[np.inf, 0.0, 0.0]])},
        {"seeds": np.zeros((2, 2))},
        {"xgrid": np.array([0.0, 0.0, 1.0])},
        {"xgrid": np.array([0.0, 2.0, 1.0])},
        {"xgrid": np.array([0.0, np.nan, 1.0])},
        {"xgrid": np.array([0.0, 1.0])},
        {"bx": np.zeros((2, 3, 3))},
    ],
)
def test_bad_trace_inputs(options):
    with pytest.raises(ValueError):
        trace(**options)


def test_numpy_integer_parameters_and_weak_nonzero_field():
    weak = np.full((3, 3, 3), 1e-14)
    result = trace(bz=weak, max_steps=np.int64(3), save_stride=np.int64(2))
    np.testing.assert_array_equal(result.exit_code_plus, [8])
    np.testing.assert_array_equal(result.line_count, [1])


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_nonfinite_scalar_and_sampling_coordinates(invalid):
    result = trace()
    grid = np.array([-1.0, 0.0, 1.0])
    sample = sample_scalar_on_lines(
        result, np.full((3, 3, 3), invalid), grid, grid, grid
    )
    np.testing.assert_array_equal(sample.codes_all, 9)
    assert np.all(np.isnan(sample.values_all))
    result.xfl[0] = invalid
    sample = sample_scalar_on_lines(result, np.ones((3, 3, 3)), grid, grid, grid)
    assert sample.codes_all[0] == 9
    assert np.isnan(sample.values_all[0])
    np.testing.assert_array_equal(sample.codes_all[1:], 0)


@pytest.mark.parametrize("dtype", [np.int32, np.float16])
def test_sampling_rejects_unsupported_dtype(dtype):
    grid = np.array([-1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="dtype"):
        sample_scalar_on_lines(
            trace(), np.ones((3, 3, 3)), grid, grid, grid, dtype=dtype
        )


def test_sampling_keeps_coordinate_precision_independent_of_output_dtype():
    grid = np.array([1.0, 1.0 + 1e-9, 2.0])
    sample = sample_scalar_on_lines(
        trace(),
        np.ones((3, 3, 3)),
        grid,
        grid,
        grid,
        dtype=np.float32,
        periodic_convention="endpoint_included",
    )
    assert sample.values_all.dtype == np.float32


@pytest.mark.parametrize("use_codes", [False, True])
def test_diagnostics_do_not_bridge_invalid_samples(use_codes):
    result = SimpleNamespace(
        sfl=np.arange(5, dtype=float),
        zfl=np.arange(5, dtype=float),
        line_start=np.array([0]),
        line_count=np.array([5]),
    )
    sample = SimpleNamespace(
        values_all=np.array([2.0, 2.0, np.nan, 2.0, 2.0]),
        codes_all=np.array([0, 0, 9, 0, 0]),
        line_start=result.line_start,
        line_count=result.line_count,
    )
    diag = compute_line_diagnostics(
        result, sample, use_codes=use_codes, dtype=np.float64
    )
    np.testing.assert_allclose(diag.integral, [4.0])
    np.testing.assert_allclose(diag.mean, [2.0])
    np.testing.assert_allclose(diag.s_centroid_abs, [2.0])


def test_sampling_type_hints_resolve():
    assert get_type_hints(sample_scalar_on_lines)["result"].__name__ == "TraceResult"


def test_distant_periodic_seed_wraps_without_iterative_loop():
    result = trace(seeds=np.array([[1e12 + 0.25, 0.0, 0.0]]))
    # Both the seed and subsequent public positions are now canonical.
    mask = np.ones(result.total_points, dtype=bool)
    mask[result.seed_index] = False
    np.testing.assert_allclose(result.xfl[mask], 0.25)
