"""Scientific regressions on native XYZ C/F/strided fields."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from fieldline_tracing_proto import (
    compute_line_diagnostics,
    compute_local_winding,
    sample_scalar_on_lines,
    trace_field_lines,
)
from fieldline_tracing_proto.grid import (
    _find_index_1d,
    _inv_spacing,
    is_uniform_grid,
    prepare_array,
)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("method", ["rk2", "rk45"])
def test_non_cubic_xyz_affine_field_and_scalar(order, method, dtype):
    x = np.linspace(-2, 2, 7)
    y = np.linspace(-3, 3, 5)
    z = np.linspace(-1, 1, 9)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    fields = [np.array(a, dtype=dtype, order=order) for a in (-Y, X, Z * 0 + 0.3)]
    for a in fields:
        assert np.shares_memory(prepare_array(a, a.dtype), a)
    kwargs = {"max_length": 0.5, "rtol": 1e-9} if method == "rk45" else {}
    r = trace_field_lines(
        *fields,
        x,
        y,
        z,
        np.array([[1.0, 0.0, 0.0]]),
        0.005,
        100,
        method=method,
        periodic_x=False,
        periodic_y=False,
        **kwargs,
    )
    scalar = np.array(2 * X - 3 * Y + 4 * Z, dtype=dtype, order=order)
    sample = sample_scalar_on_lines(r, scalar, x, y, z)
    np.testing.assert_allclose(
        sample.values_all,
        2 * r.xfl - 3 * r.yfl + 4 * r.zfl,
        atol=2e-6 if dtype == np.float32 else 1e-13,
    )
    assert json.loads(r.settings_json)["component_layouts"] == [order] * 3
    np.testing.assert_array_equal(sample.codes_all, 0)


@pytest.mark.parametrize("layout", ["preserve", "interleaved"])
def test_strided_and_packed_inputs_match_native_reference(layout):
    x = np.linspace(-2, 2, 9)
    y = np.linspace(-2, 2, 7)
    z = np.linspace(-2, 2, 5)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    fields = (-Y, X, Z * 0 + 0.2)
    r = trace_field_lines(
        *(a[::2] for a in fields),
        x[::2],
        y,
        z,
        np.array([[1.0, 0.0, 0.0]]),
        0.02,
        100,
        method="rk45",
        max_length=0.7,
        periodic_x=False,
        periodic_y=False,
        field_layout=layout,
    )
    np.testing.assert_allclose(r.xfl, np.cos((r.sfl - 0.7) / np.sqrt(1.04)), atol=3e-6)
    assert r.sfl.dtype == np.float64


@pytest.mark.parametrize("order", ["C", "F"])
def test_endpoint_excluded_periodic_seam_and_one_period(order):
    x = np.arange(32) / 32
    y = np.array([0.0, 0.4, 1.0])
    z = np.array([0.0, 1.0])
    X, _Y, _Z = np.meshgrid(x, y, z, indexing="ij")
    a = np.array(np.ones_like(X), order=order)
    r = trace_field_lines(
        a,
        a * 0,
        a * 0,
        x,
        y,
        z,
        np.array([[1.3, 0.5, 0.5]]),
        1 / 64,
        64,
        periodic_x=True,
        periodic_y=False,
        period=(1.0, 0.0, 0.0),
    )
    assert r.xfl[-1] == pytest.approx(0.3, abs=1e-14)
    assert r.seeds[0, 0] == pytest.approx(0.3)
    assert r.original_seeds[0, 0] == 1.3
    q = np.array([1 - 1 / 64, 1 + 1 / 128])
    line = SimpleNamespace(
        xfl=q,
        yfl=q * 0 + 0.5,
        zfl=q * 0 + 0.5,
        line_start=np.array([0]),
        line_count=np.array([2]),
    )
    sampled = sample_scalar_on_lines(
        line,
        np.array(np.sin(2 * np.pi * X), order=order),
        x,
        y,
        z,
        periodic_y=False,
        period=(1.0, 0.0, 0.0),
    )
    expected = np.interp(q % 1, np.r_[x, 1.0], np.r_[np.sin(2 * np.pi * x), 0.0])
    np.testing.assert_allclose(sampled.values_all, expected, atol=1e-14)


def test_periodic_sheared_field_has_no_secular_drift():
    x = np.arange(64) / 64
    g = np.array([0.0, 1.0])
    X, _Y, _Z = np.meshgrid(x, g, g, indexing="ij")
    r = trace_field_lines(
        X * 0 + 1,
        X * 0,
        0.2 * np.sin(2 * np.pi * X),
        x,
        g,
        g,
        np.array([[0.0, 0.5, 0.5]]),
        1 / 512,
        10000,
        periodic_x=True,
        periodic_y=False,
        period=(1.0, 0.0, 0.0),
    )
    assert r.zfl.min() >= 0.5 - 1e-5
    assert r.zfl.max() <= 0.564


def test_nonuniform_period_requires_explicit_physical_length():
    x = np.array([0.0, 0.1, 0.4])
    g = np.array([0.0, 1.0])
    a = np.ones((3, 2, 2))
    with pytest.raises(ValueError, match="explicit physical period"):
        trace_field_lines(
            a, a * 0, a * 0, x, g, g, np.zeros((1, 3)), 0.01, 2, periodic_y=False
        )
    r = trace_field_lines(
        a,
        a * 0,
        a * 0,
        x,
        g,
        g,
        np.array([[0.7, 0.5, 0.5]]),
        0.01,
        2,
        periodic_x=True,
        periodic_y=False,
        period=(1.0, 0.0, 0.0),
    )
    assert r.xfl[r.seed_index[0]] == pytest.approx(0.7)


def test_default_grid_step_cap_detects_a_one_cell_structure():
    x = np.linspace(0, 20, 401)
    y = np.array([0.0, 1.0])
    z = np.array([-1.0, 1.0])
    X, _Y, _Z = np.meshgrid(x, y, z, indexing="ij")
    bz = np.exp(-(((X - 15) / 0.05) ** 2))
    r = trace_field_lines(
        X * 0 + 1,
        X * 0,
        bz,
        x,
        y,
        z,
        np.array([[1.0, 0.5, 0.0]]),
        0.01,
        10000,
        method="rk45",
        rtol=1e-8,
        atol=1e-10,
        periodic_x=False,
        periodic_y=False,
    )
    assert r.zfl[-1] == pytest.approx(0.05 * np.sqrt(np.pi), abs=3e-5)
    assert json.loads(r.settings_json)["max_step"] == "local_grid"


def test_rk45_error_scale_is_translation_invariant():
    errors = []
    counts = []
    for offset in (0.0, 100.0, 1000.0):
        x = np.linspace(-3, 3, 17) + offset
        z = np.array([-1.0, 1.0])
        X, Y, Z = np.meshgrid(x, x, z, indexing="ij")
        r = trace_field_lines(
            -(Y - offset),
            X - offset,
            Z * 0,
            x,
            x,
            z,
            np.array([[offset + 2, offset, 0.0]]),
            0.1,
            10000,
            method="rk45",
            max_length=20 * np.pi,
            rtol=1e-6,
            atol=1e-9,
            periodic_x=False,
            periodic_y=False,
        )
        errors.append(abs(np.hypot(r.xfl[-1] - offset, r.yfl[-1] - offset) - 2))
        counts.append(r.nsteps_plus[0])
    assert max(errors) < 2e-5
    assert max(counts) - min(counts) <= 1


@pytest.mark.parametrize("method", ["rk2", "rk45"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_exact_face_endpoints_and_monotone_arc(method, dtype):
    x = np.linspace(0, 1, 5)
    a = np.ones((5, 5, 5), dtype=dtype)
    r = trace_field_lines(
        a * 0,
        a * 0,
        a,
        x,
        x,
        x,
        np.array([[0.5, 0.5, 0.3]]),
        0.13,
        1000,
        method=method,
        periodic_x=False,
        periodic_y=False,
        dtype=dtype,
    )
    assert r.zfl[0] == 0.0 and r.zfl[-1] == 1.0
    assert r.endpoints_minus[0, 2] == 0.0 and r.endpoints_plus[0, 2] == 1.0
    assert r.exit_faces_minus[0] == 1 and r.exit_faces_plus[0] == 2
    assert np.all(np.diff(r.sfl) > 0)
    assert r.length_plus[0] == pytest.approx(0.7, abs=1e-12)
    assert r.nsteps_plus[0] < 30


@pytest.mark.parametrize("method", ["rk2", "rk45"])
def test_seed_outside_nonperiodic_grid_rejected_before_stepping(method):
    x = np.array([0.0, 1.0])
    a = np.ones((2, 2, 2))
    with pytest.raises(ValueError, match="Seed outside"):
        trace_field_lines(
            a * 0,
            a * 0,
            a,
            x,
            x,
            x,
            np.array([[0.5, 0.5, 1.5]]),
            0.1,
            10,
            method=method,
        )


@pytest.mark.parametrize(
    "extent,n", [(24.0, 768), (1.0, 256), (6.0, 1024), (16.0, 512)]
)
def test_float32_lattice_lookup_uses_actual_nodes(extent, n):
    grid = np.linspace(0, extent, n, dtype=np.float32)
    assert is_uniform_grid(grid)
    g = grid.astype(float)
    inv = _inv_spacing(g)
    q = np.r_[g[:-1], 0.5 * (g[:-1] + g[1:]), np.nextafter(g[1:-1], -np.inf)]
    for p in q:
        actual = _find_index_1d(g, p, True, inv)
        assert g[actual] <= p <= g[actual + 1]


def test_sampling_inherits_flags_periods_and_double_precision():
    x = np.linspace(0, 1, 5)
    a = np.ones((5, 5, 5))
    r = trace_field_lines(
        a * 0,
        a * 0,
        a,
        x,
        x,
        x,
        np.array([[0.5, 0.5, 0.5]]),
        0.1,
        2,
        periodic_x=False,
        periodic_y=False,
    )
    r.yfl[-1] = 2.0
    sample = sample_scalar_on_lines(r, a, x, x, x)
    assert sample.values_all.dtype == np.float64 and sample.codes_all[-1] == 4
    with pytest.warns(UserWarning, match="differs"):
        other = sample_scalar_on_lines(r, a, x, x, x, periodic_y=True)
    assert other.codes_all[-1] == 0


def test_single_and_two_pass_and_seed_permutation_match():
    x = np.linspace(-2, 2, 9)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    seeds = np.array([[1.0, 0.0, 0.0], [0.7, 0.3, 0.1], [0.5, 0.8, 0.0]])
    kwargs = {
        "method": "rk45",
        "max_length": 0.7,
        "save_stride": 3,
        "periodic_x": False,
        "periodic_y": False,
    }
    results = [
        trace_field_lines(
            -Y,
            X,
            Z * 0 + 0.2,
            x,
            x,
            x,
            seeds,
            0.1,
            1000,
            storage=storage,
            seed_order=order,
            **kwargs,
        )
        for storage, order in [
            ("single_pass", "input"),
            ("single_pass", "balanced"),
            ("two_pass", "balanced"),
        ]
    ]
    for r in results[1:]:
        for name in (
            "xfl",
            "yfl",
            "zfl",
            "sfl",
            "exit_code_minus",
            "exit_code_plus",
            "nsteps_plus",
            "nrejected_plus",
            "nfev_plus",
        ):
            np.testing.assert_array_equal(getattr(r, name), getattr(results[0], name))
    perm = np.array([2, 0, 1])
    r = trace_field_lines(-Y, X, Z * 0 + 0.2, x, x, x, seeds[perm], 0.1, 1000, **kwargs)
    for i, j in enumerate(perm):
        for a, b in zip(r.get_line(i), results[0].get_line(j)):
            np.testing.assert_array_equal(a, b)


def test_winding_defaults_remove_parallel_line_endpoint_artifact_and_reject_inf():
    x = np.linspace(0, 1, 5)
    a = np.ones((5, 5, 5))
    r = trace_field_lines(
        a,
        a,
        0.1 * a,
        x,
        x,
        x,
        np.array([[0.6, 0.2, 0.5], [0.63, 0.2, 0.5]]),
        0.001,
        3000,
        periodic_x=False,
        periodic_y=False,
    )
    w = compute_local_winding(r, (1, 2), neighbor_offsets=((0, 1),))
    assert w.mean_wind[0] == pytest.approx(0.0, abs=1e-12) and w.n_pairs[0] == 1
    r.sfl[-1] = np.inf
    w = compute_local_winding(r, (1, 2), neighbor_offsets=((0, 1),))
    assert np.isnan(w.mean_wind[0]) and w.n_pairs[0] == 0


def test_continuous_density_percentile_is_not_vertex_quantized():
    r = SimpleNamespace(
        sfl=np.array([0.0, 1.0]),
        zfl=np.array([0.0, 1.0]),
        line_start=np.array([0]),
        line_count=np.array([2]),
    )
    sample = SimpleNamespace(
        values_all=np.array([0.0, 2.0]),
        codes_all=np.array([0, 0]),
        line_start=r.line_start,
        line_count=r.line_count,
    )
    d = compute_line_diagnostics(r, sample)
    assert d.s16_abs[0] == pytest.approx(np.sqrt(0.16))
    assert d.s84_abs[0] == pytest.approx(np.sqrt(0.84))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_work_coordinates_keep_narrow_double_grid(dtype):
    x = 1.0 + np.arange(3) * 1e-10
    a = np.ones((3, 3, 3), dtype=dtype)
    r = trace_field_lines(
        a * 0,
        a * 0,
        a,
        x,
        x,
        x,
        np.array([[x[1], x[1], x[1]]]),
        2e-11,
        100,
        method="rk45",
        periodic_x=False,
        periodic_y=False,
    )
    assert r.length_plus[0] == pytest.approx(x[-1] - x[1], rel=1e-4)
    assert r.endpoints_plus[0, 2] == x[-1]
    assert r.sfl.dtype == np.float64
    # Quantized float32 output cannot resolve this grid; work/endpoints can.
    sampled = sample_scalar_on_lines(r, a, x, x, x)
    np.testing.assert_array_equal(sampled.codes_all, 0)


def test_periodic_winding_on_continuous_cover_and_sparse_rejection():
    x = np.arange(16) / 16
    y = np.linspace(0, 1, 3)
    z = np.linspace(-1, 1, 5)
    a = np.ones((16, 3, 5))
    kwargs = {
        "method": "rk45",
        "max_length": 2.0,
        "periodic_x": True,
        "periodic_y": False,
        "period": (1.0, 0.0, 0.0),
    }
    seeds = np.array([[0.99, 0.3, 0.0], [0.01, 0.4, 0.0]])
    r = trace_field_lines(
        a, a * 0, a * 0, x, y, z, seeds, 0.05, 1000, wrap_output=False, **kwargs
    )
    w = compute_local_winding(r, (1, 2), neighbor_offsets=((0, 1),))
    assert w.mean_wind[0] == pytest.approx(0.0, abs=1e-12)
    sparse = trace_field_lines(
        a, a * 0, a * 0, x, y, z, seeds, 0.05, 1000, save_stride=20, **kwargs
    )
    with pytest.raises(ValueError, match="Sparse wrapped"):
        compute_local_winding(sparse, (1, 2), neighbor_offsets=((0, 1),))


def test_schema3_preserves_layout_period_endpoints_and_arc(tmp_path):
    from fieldline_tracing_proto import load_trace_npz, save_trace_npz

    x = np.arange(4) / 4
    a = np.ones((4, 4, 4), dtype=np.float32, order="F")
    r = trace_field_lines(
        a * 0,
        a * 0,
        a,
        x,
        x,
        x,
        [[0.2, 0.2, 0.4]],
        0.1,
        100,
        method="rk45",
        periodic_x=True,
        periodic_y=True,
        period=(1.0, 1.0, 0.0),
    )
    path = tmp_path / "trace.npz"
    save_trace_npz(path, r)
    loaded = load_trace_npz(path)
    for key in (
        "sfl",
        "original_seeds",
        "endpoints_minus",
        "endpoints_plus",
        "exit_faces_minus",
        "exit_faces_plus",
    ):
        np.testing.assert_array_equal(getattr(r, key), getattr(loaded, key))
    assert loaded.sfl.dtype == np.float64 and loaded.settings_json == r.settings_json


@pytest.mark.parametrize("method", ["rk2", "rk45"])
@pytest.mark.parametrize("storage", ["single_pass", "two_pass"])
def test_whole_period_steps_retain_arc_even_when_wrapped_positions_coincide(
    method, storage
):
    x = np.arange(4) / 4
    y = np.array([0.0, 1.0])
    a = np.ones((4, 2, 2))
    kwargs = {"max_step": np.inf} if method == "rk45" else {}
    r = trace_field_lines(
        a,
        a * 0,
        a * 0,
        x,
        y,
        y,
        [[0.25, 0.5, 0.5]],
        1.0,
        1,
        method=method,
        periodic_x=True,
        periodic_y=False,
        period=(1.0, 0.0, 0.0),
        max_length=1.0,
        storage=storage,
        **kwargs,
    )
    assert r.line_count[0] == 3
    np.testing.assert_allclose(r.sfl, [0.0, 1.0, 2.0], atol=1e-14)
    np.testing.assert_allclose(r.xfl, 0.25, atol=1e-14)
