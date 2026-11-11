"""Regression for the real 256³ Olshevsky near-face RK45 underflow."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation.olshevsky.field import create_grid, olshevsky2020_field
from fieldline_tracing_proto import trace_field_lines


@pytest.fixture(scope="module")
def original_field():
    grid = create_grid(256)
    return grid, olshevsky2020_field(grid, grid, grid, dtype=np.float32)


@pytest.mark.parametrize("storage", ["single_pass", "two_pass"])
def test_olshevsky_seed224_reaches_exact_ymin_instead_of_underflow(
    original_field, storage
):
    grid, fields = original_field
    seed = np.array([[3.04341161394743, -1.2780522669063608, 1.385424563510582]])
    r = trace_field_lines(
        *fields,
        grid,
        grid,
        grid,
        seed,
        ds=0.2 * (grid[1] - grid[0]),
        max_steps=20000,
        method="rk45",
        max_length=0.5,
        dtype=np.float32,
        save_stride=1,
        max_step=grid[1] - grid[0],
        rtol=1e-7,
        atol=1e-10,
        periodic_x=False,
        periodic_y=False,
        periodic_z=False,
        bmin=1e-12,
        storage=storage,
    )
    assert r.exit_code_minus[0] == -3
    assert r.exit_faces_minus[0] == 3
    assert r.endpoints_minus[0, 1] == grid[0]
    assert np.all(r.endpoints_minus[0] >= grid[0])
    assert np.all(r.endpoints_minus[0] <= grid[-1])
    assert r.length_minus[0] == pytest.approx(0.4403506568940704, abs=2e-11)
    # Output may round to float32; the scientific endpoint remains float64.
    np.testing.assert_array_equal(
        np.array(r.get_line(0)[:3])[:, 0], r.endpoints_minus[0].astype(np.float32)
    )
    assert np.all(np.diff(r.get_line(0)[3]) > 0)
