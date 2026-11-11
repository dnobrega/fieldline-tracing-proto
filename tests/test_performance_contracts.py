"""Scientific/storage contracts for scalar kernels and optimistic buffers."""

import json
import warnings
from decimal import Decimal, localcontext
from types import SimpleNamespace

import numpy as np
import pytest

from fieldline_tracing_proto import (
    PeriodicInferenceWarning,
    compute_local_winding,
    trace_field_lines,
)
from fieldline_tracing_proto.events import _range_proves_inside


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("method", ["rk2", "rk45"])
@pytest.mark.parametrize("stride", [1, 3])
def test_overflow_recovery_keeps_exact_faces_and_all_branch_metadata(
    dtype, method, stride
):
    g = np.array([0.0, 0.2, 0.5, 1.0, 2.0])
    a = np.ones((5, 5, 5), dtype=dtype)
    seeds = np.array([[0.5, 0.6, 0.01], [0.5, 0.6, 0.5], [0.5, 0.6, 1.9]])
    kw = dict(
        method=method,
        max_length=3.0,
        save_stride=stride,
        periodic_x=False,
        periodic_y=False,
    )
    r = trace_field_lines(
        a * 0,
        a * 0,
        a,
        g,
        g,
        g,
        seeds,
        0.037,
        20000,
        buffer_points=1,
        storage="single_pass",
        memory_limit_mb=0.001,
        **kw,
    )
    ref = trace_field_lines(
        a * 0, a * 0, a, g, g, g, seeds, 0.037, 20000, storage="two_pass", **kw
    )
    for name in (
        "xfl",
        "yfl",
        "zfl",
        "sfl",
        "line_start",
        "line_count",
        "seed_index",
        "exit_code_minus",
        "exit_code_plus",
        "nsteps_minus",
        "nsteps_plus",
        "length_minus",
        "length_plus",
        "nfev_minus",
        "nfev_plus",
        "nrejected_minus",
        "nrejected_plus",
        "endpoints_minus",
        "endpoints_plus",
    ):
        np.testing.assert_array_equal(getattr(r, name), getattr(ref, name))
    np.testing.assert_array_equal(r.exit_code_minus, -1)
    np.testing.assert_array_equal(r.exit_code_plus, 2)
    np.testing.assert_allclose(r.length_minus, seeds[:, 2], atol=1e-12)
    np.testing.assert_allclose(r.length_plus, 2 - seeds[:, 2], atol=1e-12)
    meta = json.loads(r.settings_json)
    assert meta["overflow_retraced"] == 3
    assert meta["peak_branch_buffer_bytes"] <= 0.001 * 1024**2
    assert meta["work_nfev"] > int(r.nfev_minus.sum() + r.nfev_plus.sum())


def test_safety_budget_does_not_expand_optimistic_buffers():
    g = np.array([-1.0, 1.0])
    a = np.ones((2, 2, 2))
    seeds = np.zeros((32, 3))
    r = trace_field_lines(
        a * 0,
        a * 0,
        a,
        g,
        g,
        g,
        seeds,
        0.05,
        1_000_000,
        method="rk45",
        max_length=0.2,
        periodic_x=False,
        periodic_y=False,
        buffer_points=4,
        memory_limit_mb=1.0,
    )
    meta = json.loads(r.settings_json)
    assert meta["storage"] == "single_pass" and meta["overflow_retraced"] == 0
    assert meta["buffer_capacity_peak"] == 4 and meta["effective_block_size"] == 32
    assert meta["peak_branch_buffer_bytes"] == 32 * 9 * 32
    np.testing.assert_array_equal(r.line_count, 5)
    np.testing.assert_allclose(r.length_plus, 0.2)


def test_memory_floor_uses_two_pass_and_accounts_for_actual_work():
    g = np.array([-1.0, 1.0])
    a = np.ones((2, 2, 2))
    seeds = np.zeros((32, 3))
    r = trace_field_lines(
        a * 0,
        a * 0,
        a,
        g,
        g,
        g,
        seeds,
        0.05,
        10000,
        method="rk45",
        max_length=0.2,
        periodic_x=False,
        periodic_y=False,
        buffer_points=4,
        memory_limit_mb=0.001,
    )
    meta = json.loads(r.settings_json)
    assert meta["storage"] == "two_pass" and meta["peak_branch_buffer_bytes"] == 0
    assert meta["work_nfev"] == 2 * int(r.nfev_minus.sum() + r.nfev_plus.sum())
    np.testing.assert_allclose(r.sfl[[0, -1]], [0, 0.4])


def test_learning_switches_to_exact_storage_when_large_rows_cannot_keep_parallelism():
    g = np.array([-20.0, 20.0])
    a = np.ones((2, 2, 2))
    seeds = np.zeros((16, 3))
    kw = dict(max_length=2.0, periodic_x=False, periodic_y=False)
    r = trace_field_lines(
        a * 0,
        a * 0,
        a,
        g,
        g,
        g,
        seeds,
        0.05,
        5000,
        buffer_points=1,
        block_size=4,
        memory_limit_mb=0.001,
        **kw,
    )
    ref = trace_field_lines(
        a * 0, a * 0, a, g, g, g, seeds, 0.05, 5000, storage="two_pass", **kw
    )
    meta = json.loads(r.settings_json)
    assert meta["storage"] == "mixed" and meta["overflow_retraced"] == 4
    assert meta["peak_branch_buffer_bytes"] <= 0.001 * 1024**2
    np.testing.assert_array_equal(r.sfl, ref.sfl)
    np.testing.assert_array_equal(r.zfl, ref.zfl)


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_invalid_cell_cap_rejected(value):
    g = np.array([-1.0, 1.0])
    a = np.ones((2, 2, 2))
    with pytest.raises(ValueError, match="max_step_cells"):
        trace_field_lines(
            a * 0,
            a * 0,
            a,
            g,
            g,
            g,
            [[0, 0, 0]],
            0.1,
            5,
            method="rk45",
            max_step_cells=value,
        )


def test_cell_cap_and_absolute_cap_are_unambiguous():
    g = np.linspace(-2, 2, 33)
    a = np.ones((33, 33, 33))
    kw = dict(method="rk45", max_length=1.0, periodic_x=False, periodic_y=False)
    small = trace_field_lines(
        a * 0, a * 0, a, g, g, g, [[0, 0, 0]], 0.01, 1000, max_step_cells=0.5, **kw
    )
    large = trace_field_lines(
        a * 0, a * 0, a, g, g, g, [[0, 0, 0]], 0.01, 1000, max_step_cells=2.0, **kw
    )
    assert small.nsteps_plus[0] > large.nsteps_plus[0]
    assert np.max(np.diff(small.sfl)) <= 0.0625 + 1e-14
    assert np.max(np.diff(large.sfl)) <= 0.25 + 1e-14
    np.testing.assert_allclose(large.zfl[[0, -1]], [-1, 1])
    with pytest.raises(ValueError, match="only one"):
        trace_field_lines(
            a * 0,
            a * 0,
            a,
            g,
            g,
            g,
            [[0, 0, 0]],
            0.01,
            100,
            max_step=0.1,
            max_step_cells=1.0,
            **kw,
        )
    with pytest.raises(ValueError, match="Adaptive"):
        trace_field_lines(
            a * 0,
            a * 0,
            a,
            g,
            g,
            g,
            [[0, 0, 0]],
            0.01,
            100,
            max_step_cells=1.0,
            periodic_x=False,
            periodic_y=False,
        )


def test_periodic_assumptions_are_visible_and_explicit_domains_are_quiet():
    g = np.array([0.0, 0.5])
    a = np.ones((2, 2, 2))
    with pytest.warns(PeriodicInferenceWarning, match="inferred period"):
        r = trace_field_lines(a * 0, a * 0, a, g, g, g, [[0.1, 0.1, 0.1]], 0.01, 2)
    assert json.loads(r.settings_json)["period_inferred_axes"] == ["x", "y"]
    with warnings.catch_warnings(record=True) as found:
        warnings.simplefilter("always")
        trace_field_lines(
            a * 0,
            a * 0,
            a,
            g,
            g,
            g,
            [[0.1, 0.1, 0.1]],
            0.01,
            2,
            periodic_x=True,
            periodic_y=True,
            period=(1.0, 1.0, 0.0),
        )
    assert not found


def test_sparse_winding_mask_exposes_affected_lines_and_preserves_other_pairs():
    s = np.array([0.0, 1.0, 0.0, 0.2, 0.4, 0.0, 0.2, 0.4])
    r = SimpleNamespace(
        sfl=s,
        zfl=s.copy(),
        xfl=np.array([0.1, 0.1, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4]),
        yfl=s * 0,
        line_start=np.array([0, 2, 5]),
        line_count=np.array([2, 3, 3]),
        seed_index=np.array([0, 2, 5]),
        settings_json='{"period":[1,0,0]}',
    )
    with pytest.raises(ValueError, match="Sparse wrapped"):
        compute_local_winding(r, (1, 3), neighbor_offsets=((0, 1),))
    with pytest.warns(UserWarning, match="1 sparse wrapped"):
        w = compute_local_winding(
            r, (1, 3), neighbor_offsets=((0, 1),), sparse_policy="mask"
        )
    np.testing.assert_array_equal(w.valid_lines, [False, True, True])
    np.testing.assert_array_equal(w.invalid_reason, [2, 0, 0])
    np.testing.assert_array_equal(w.n_pairs, [0, 1, 0])
    assert w.mean_wind[1] == pytest.approx(0.0)


def test_overlapping_ragged_segments_rejected_before_parallel_writes():
    r = SimpleNamespace(
        sfl=np.array([0.0, 0.1, 0.2]),
        xfl=np.zeros(3),
        yfl=np.zeros(3),
        zfl=np.array([0.0, 0.1, 0.2]),
        line_start=np.array([0, 1]),
        line_count=np.array([2, 2]),
        seed_index=np.array([0, 1]),
    )
    with pytest.raises(ValueError, match="overlap"):
        compute_local_winding(r, (1, 2))


@pytest.mark.parametrize("scale", [1e-320, 1e-300, 1e-100, 1.0, 1e100, 1e300])
def test_fast_range_proof_against_exact_decimal_coefficient_enclosures(scale):
    rng = np.random.default_rng(291)
    certified = 0
    with localcontext() as ctx:
        ctx.prec = 1100
        bound = Decimal.from_float(scale)
        for _ in range(80):
            c = rng.normal(size=5) * scale * 0.2
            if _range_proves_inside(c, -scale, scale):
                certified += 1
                values = [Decimal.from_float(float(v)) for v in c]
                low = values[0] + sum((v for v in values[1:] if v < 0), Decimal(0))
                high = values[0] + sum((v for v in values[1:] if v > 0), Decimal(0))
                assert low > -bound and high < bound
        assert certified > 10
        edge = np.array(
            [
                np.nextafter(scale, 0.0),
                np.spacing(scale),
                -np.spacing(scale) / 2,
                0.0,
                0.0,
            ]
        )
        assert not _range_proves_inside(edge, -scale, scale)


@pytest.mark.parametrize("origin", [-1000.0, -2.0, 0.0, 1000.0])
def test_periodic_fast_path_preserves_canonical_rounding(origin):
    from fieldline_tracing_proto.grid import _wrap_coord

    period = 4.5
    values = np.r_[
        np.linspace(origin, origin + period, 257),
        np.nextafter(origin + period, origin),
        origin - 1e12,
        origin + 1e12,
    ]
    for value in values:
        expected = origin + np.remainder(value - origin, period)
        if expected >= origin + period:
            expected = origin
        assert _wrap_coord(value, origin, origin + 4.0, period) == expected


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_nonperiodic_closed_return_retains_terminal_arc(dtype):
    x = np.linspace(-2, 2, 9)
    X, Y, Z = np.meshgrid(x, x, [-1.0, 1.0], indexing="ij")
    r = trace_field_lines(
        -Y.astype(dtype),
        X.astype(dtype),
        Z.astype(dtype) * 0,
        x,
        x,
        [-1.0, 1.0],
        [[2**-0.5, 2**-0.5, 0.0]],
        0.05,
        20000,
        method="rk45",
        max_length=2 * np.pi,
        rtol=1e-10,
        atol=1e-12,
        save_stride=20000,
        periodic_x=False,
        periodic_y=False,
    )
    assert r.line_count[0] == 3
    np.testing.assert_allclose(r.sfl, [0.0, 2 * np.pi, 4 * np.pi], atol=1e-12)
    np.testing.assert_allclose(r.xfl, 2**-0.5, atol=1e-7)
    np.testing.assert_allclose(r.yfl, 2**-0.5, atol=1e-7)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_exact_trace_anchors_prevent_rounding_outside_scalar_domain(dtype):
    from fieldline_tracing_proto import compute_line_diagnostics, sample_scalar_on_lines

    g = np.array([0.0, 1.1])
    a = np.ones((2, 2, 2), dtype)
    r = trace_field_lines(
        a * 0,
        a * 0,
        a,
        g,
        g,
        g,
        [[0.3, 0.3, 0.55]],
        0.2,
        10,
        periodic_x=False,
        periodic_y=False,
    )
    before = r.zfl.copy()
    sample = sample_scalar_on_lines(r, a * 2, g, g, g)
    np.testing.assert_array_equal(sample.codes_all, 0)
    assert compute_line_diagnostics(r, sample).integral[0] == pytest.approx(
        2.2, abs=1e-12
    )
    np.testing.assert_array_equal(r.zfl, before)
    if dtype == np.float32:
        literal = sample_scalar_on_lines(r, a * 2, g, g, g, exact_anchors=False)
        assert literal.codes_all[-1] == 2
    r.zfl[-1] = 1.2
    edited = sample_scalar_on_lines(r, a * 2, g, g, g)
    assert edited.codes_all[-1] == 2
