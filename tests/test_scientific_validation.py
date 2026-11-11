"""Analytic boundary cases and independent integration/interpolation references."""

import numpy as np
import pytest

from fieldline_tracing_proto.tracing import trace_field_lines
from validation.run_validation import (
    cubic_reference,
    cubic_trace,
    parabola_arc,
    parabola_trace,
)


@pytest.mark.parametrize("clearance", [1e-3, 1e-6, 1e-9])
@pytest.mark.parametrize("rtol", [1e-6, 1e-9, 1e-12])
def test_grazing_and_interior_trajectory_stays_inside(clearance, rtol):
    result = parabola_trace(clearance, rtol)
    assert result.exit_code_plus[0] == 11
    assert result.length_plus[0] == pytest.approx(1.3)
    seed = result.seed_index[0]
    x, y = result.xfl[seed:], result.yfl[seed:]
    np.testing.assert_allclose(x + y * y, 1 - clearance, atol=5 * rtol, rtol=0)
    assert np.max(x) <= 1.0
    assert y[-1] > 0.5  # Passed the tangency region and returned inside.


@pytest.mark.parametrize("rtol", [1e-6, 1e-9, 1e-12])
def test_unrestricted_step_must_detect_first_exit(rtol):
    result = parabola_trace(-1e-6, rtol)
    assert result.exit_code_plus[0] == 6
    expected_length = parabola_arc(-0.001) - parabola_arc(-0.5)
    # At this shallow crossing |dx/ds| ~ 0.002: position error is amplified
    # when converted to event arc length. boundary_tol is not a global-error bound.
    assert result.length_plus[0] == pytest.approx(
        expected_length, abs=max(1e-7, 100 * rtol)
    )


@pytest.mark.parametrize("rtol", [1e-6, 1e-9, 1e-12])
def test_exact_contact_is_either_traversed_or_resolved_near_tangency(rtol):
    result = parabola_trace(0.0, rtol)
    # An exact mathematical tangency is ill-conditioned under trajectory error.
    # A numerically perturbed curve may cross. Do not label such a result as
    # completion; require an explicit face exit close to the exact contact.
    code = result.exit_code_plus[0]
    assert code in (6, 11)
    if code == 6:
        assert abs(result.yfl[-1]) < np.sqrt(5 * rtol)
        assert result.xfl[-1] == pytest.approx(1.0, abs=1e-9)
        length_to_contact = parabola_arc(0.0) - parabola_arc(-0.5)
        assert abs(result.length_plus[0] - length_to_contact) < np.sqrt(5 * rtol)
    else:
        assert result.length_plus[0] == pytest.approx(1.3)


@pytest.mark.parametrize("max_step", [0.01, 0.001])
def test_bounded_step_detects_this_specific_excursion(max_step):
    result = parabola_trace(-1e-6, max_step=max_step)
    assert result.exit_code_plus[0] == 6
    expected_length = parabola_arc(-0.001) - parabola_arc(-0.5)
    assert result.length_plus[0] == pytest.approx(expected_length, abs=1e-8)
    assert result.yfl[-1] == pytest.approx(-0.001, abs=1e-8)


def constant_field_trace(vector, seed=(0.0, 0.0, 0.0), **controls):
    grid = np.linspace(-1.0, 1.0, 5)
    shape = (5, 5, 5)
    opts = {
        "ds": 0.7,
        "max_steps": 1000,
        "method": "rk45",
        "max_length": 5.0,
        "periodic_x": False,
        "periodic_y": False,
        "periodic_z": False,
        "dtype": np.float64,
        "boundary_tol": 1e-9,
    }
    opts.update(controls)
    return trace_field_lines(
        *(np.full(shape, v) for v in vector),
        grid,
        grid,
        grid,
        np.array([seed]),
        **opts,
        periodic_convention="endpoint_included",
    )


def test_simultaneous_corner_exits():
    result = constant_field_trace((1.0, 1.0, 1.0))
    np.testing.assert_allclose(result.length_minus, np.sqrt(3), atol=2e-9)
    np.testing.assert_allclose(result.length_plus, np.sqrt(3), atol=2e-9)
    np.testing.assert_allclose(
        [result.xfl[-1], result.yfl[-1], result.zfl[-1]], 1.0, atol=1e-9
    )
    np.testing.assert_allclose(
        [result.xfl[0], result.yfl[0], result.zfl[0]], -1.0, atol=1e-9
    )
    # Interpolation's x-first precedence is the current documented policy.
    assert result.exit_code_plus[0] == 6
    assert result.exit_code_minus[0] == -5


def test_boundary_seed_has_outward_and_inward_branches():
    result = constant_field_trace((1.0, 0.0, 0.0), seed=(1.0, 0.0, 0.0))
    assert result.exit_code_plus[0] == 6
    assert result.nsteps_plus[0] == 0
    assert result.length_plus[0] == 0
    assert result.exit_code_minus[0] == -5
    np.testing.assert_allclose(result.length_minus, 2.0, atol=1e-9)


def test_continuous_travel_tangent_to_face():
    result = constant_field_trace((0.0, 1.0, 0.0), seed=(1.0, 0.0, 0.0))
    np.testing.assert_array_equal(result.xfl, 1.0)
    assert result.exit_code_plus[0] == 4
    assert result.exit_code_minus[0] == -3
    np.testing.assert_allclose(result.length_plus, 1.0, atol=1e-9)


def test_many_periodic_crossings_on_all_axes():
    vector = np.array([1.0, 2.0, -0.5])
    seed = np.array([0.13, 0.27, 0.39])
    result = constant_field_trace(
        vector,
        seed=seed,
        max_length=12.3,
        max_step=1.7,
        periodic_x=True,
        periodic_y=True,
        periodic_z=True,
    )
    signed_s = result.sfl - result.length_minus[0]
    expected = seed[:, None] + vector[:, None] / np.linalg.norm(vector) * signed_s
    expected = -1 + (expected + 1) % 2.0
    np.testing.assert_allclose(
        np.array([result.xfl, result.yfl, result.zfl]), expected, atol=1e-13
    )
    assert result.exit_code_plus[0] == 11 and result.exit_code_minus[0] == -11
    np.testing.assert_array_equal(result.nrejected_plus, 0)
    np.testing.assert_array_equal(result.nrejected_minus, 0)


def test_approach_to_divergence_free_magnetic_null():
    grid = np.linspace(-1.0, 1.0, 9)
    x, y, z = np.meshgrid(grid, grid, grid, indexing="ij")
    result = trace_field_lines(
        -x,
        -y,
        2 * z,
        grid,
        grid,
        grid,
        np.array([[0.5, 0.0, 0.0]]),
        0.15,
        1000,
        method="rk45",
        max_length=0.8,
        bmin=1e-6,
        periodic_x=False,
        periodic_y=False,
        dtype=np.float64,
        rtol=1e-9,
        atol=1e-11,
        periodic_convention="endpoint_included",
    )
    assert result.exit_code_plus[0] == 8
    assert 0 < result.xfl[-1] < 1e-4
    assert result.length_plus[0] == pytest.approx(0.5, abs=1e-4)
    # Weak-field termination is not a root-localized threshold event.
    assert np.all(np.isfinite(result.sfl))


@pytest.mark.parametrize("nodes", [9, 17, 33, 65])
def test_both_branches_match_independent_interpolant_reference(nodes):
    result = cubic_trace(nodes, 1e-11, max_step=0.05)
    reference = cubic_reference(np.linspace(-2.0, 2.0, nodes))
    position = np.array([result.xfl[-1], result.yfl[-1], result.zfl[-1]])
    opposite = np.array([result.xfl[0], result.yfl[0], result.zfl[0]])
    np.testing.assert_allclose(position, reference, atol=2e-8, rtol=0)
    np.testing.assert_allclose(opposite, -reference, atol=2e-8, rtol=0)
    assert result.exit_code_plus[0] == 11 and result.exit_code_minus[0] == -11


def test_field_interpolation_error_converges_under_mesh_refinement():
    exact = cubic_reference()
    errors = [
        np.linalg.norm(cubic_reference(np.linspace(-2.0, 2.0, n)) - exact)
        for n in (9, 17, 33, 65)
    ]
    ratios = np.array(errors[:-1]) / errors[1:]
    assert np.all((3.5 < ratios) & (ratios < 4.5))


def test_tight_tolerances_reduce_integration_but_not_interpolation_floor():
    reference = cubic_reference(np.linspace(-2.0, 2.0, 17))
    exact = cubic_reference()
    integration, total = [], []
    for tol in (1e-3, 1e-11):
        result = cubic_trace(17, tol)
        position = np.array([result.xfl[-1], result.yfl[-1], result.zfl[-1]])
        integration.append(np.linalg.norm(position - reference))
        total.append(np.linalg.norm(position - exact))
    assert integration[1] < integration[0] * 1e-4
    assert total[1] == pytest.approx(np.linalg.norm(reference - exact), abs=2e-8)
    assert total[1] > 1e-3
