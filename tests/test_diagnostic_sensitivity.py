"""Check analytic sampling limits without asserting machine-specific timings."""

import numpy as np
import pytest

from validation.benchmark_tracing import (
    calibrate,
    geometry,
    solve,
    trajectory_error,
    worker,
)
from validation.diagnostic_sensitivity import (
    pair_winding,
    scalar_case,
    scalar_reference,
    synthetic_pair,
)


def test_sparse_scalar_output_can_lose_an_integral_on_an_exact_line():
    reference = scalar_reference()
    fine, fine_diag = scalar_case("rk2", stride=1)
    coarse, coarse_diag = scalar_case("rk2", stride=512)
    for result in (fine, coarse):
        np.testing.assert_allclose(result.zfl, result.sfl - 1.0, atol=1e-15)
    assert fine_diag.integral[0] == pytest.approx(reference["integral"], rel=1e-8)
    assert abs(coarse_diag.integral[0] / reference["integral"] - 1) > 0.99
    assert coarse.total_points < fine.total_points


@pytest.mark.parametrize("method", ["rk2", "rk45"])
def test_refining_output_resolves_scalar_peak_and_discrete_width(method):
    reference = scalar_reference()
    if method == "rk2":
        fine, fine_diag = scalar_case(method, stride=1)
        coarse, coarse_diag = scalar_case(method, stride=128)
    else:
        fine, fine_diag = scalar_case(method, max_step=1 / 256)
        coarse, coarse_diag = scalar_case(method, max_step=1 / 16)
    assert fine_diag.integral[0] == pytest.approx(reference["integral"], rel=1e-8)
    assert fine_diag.max_abs[0] == pytest.approx(1.0, abs=1e-12)
    assert abs(coarse_diag.max_abs[0] - 1.0) > 0.3
    assert abs(fine_diag.W68_s_abs[0] - reference["width"]) <= np.max(np.diff(fine.sfl))
    # Percentile output is a saved-coordinate statistic, not an exact quantile.
    assert fine_diag.W68_s_abs[0] != pytest.approx(reference["width"], abs=1e-5)


def test_angular_resampling_cannot_recover_lost_saved_turns():
    sparse = synthetic_pair(5, turns=3.0)
    assert pair_winding(sparse, 257) == pytest.approx(-1.0, abs=1e-12)
    assert pair_winding(sparse, 1025) == pytest.approx(-1.0, abs=1e-12)
    assert pair_winding(synthetic_pair(129, turns=3.0), 257) == pytest.approx(
        3.0, abs=1e-12
    )


def test_coarse_angular_resampling_can_alias_fine_geometry():
    fine = synthetic_pair(129, turns=3.0)
    assert pair_winding(fine, 5) == pytest.approx(-1.0, abs=1e-12)
    assert pair_winding(fine, 257) == pytest.approx(3.0, abs=1e-12)


@pytest.mark.parametrize("field", ["circle", "helix"])
@pytest.mark.parametrize("method", ["rk2", "rk45"])
def test_calibrated_configurations_meet_budget_on_new_seed_phases(field, method):
    controls, calibration_error = calibrate(field, 1e-6, method)
    assert calibration_error <= 0.8e-6
    result = solve(geometry(field, 23), controls)
    assert trajectory_error(result, field) <= 1e-6
    np.testing.assert_allclose(result.length_minus, 2.0, atol=1e-12)
    np.testing.assert_allclose(result.length_plus, 2.0, atol=1e-12)


def test_benchmark_worker_reports_checked_accuracy_and_distinct_memory_measures():
    controls, _ = calibrate("circle", 1e-4, "rk45")
    result = worker(
        dict(
            field="circle",
            target=1e-4,
            method="rk45",
            nseeds=3,
            controls=controls,
            periodic=False,
            repeats=3,
        )
    )
    assert result["measured_max_error"] <= 1e-4
    assert result["saved_points"] > 3
    assert result["output_array_mib"] > 0
    assert (
        result["worker_peak_rss_through_solves_mib"]
        >= result["worker_peak_rss_before_first_call_mib"]
    )
    assert result["warm_min_seconds"] <= result["warm_median_seconds"]
    assert result["per_integration_nfev"] > 0


def test_benchmark_worker_rejects_a_configuration_outside_error_budget():
    with pytest.raises(RuntimeError, match="failed its error budget"):
        worker(
            dict(
                field="circle",
                target=1e-10,
                method="rk2",
                nseeds=2,
                controls=dict(ds=0.5, max_steps=4),
                periodic=False,
                repeats=3,
            )
        )
