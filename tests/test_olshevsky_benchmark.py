"""Independent scientific checks for the repository-only benchmark fixture."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation.olshevsky.field import olshevsky2020_field
from validation.olshevsky.reference import field_at, make_seeds


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_grid_generator_matches_direct_points(order, dtype):
    axes = [
        np.array([-1.1, 0.012, 0.2, 2.9]),
        np.array([-0.1, 0.3, 3.1]),
        np.array([-0.2, 0.4]),
    ]
    arrays = tuple(np.empty((4, 3, 2), dtype=dtype, order=order) for _ in range(3))
    result = olshevsky2020_field(*axes, out=arrays)
    assert all(a is b for a, b in zip(arrays, result))
    for i, j, k in np.ndindex(4, 3, 2):
        expected = field_at(np.array([axes[0][i], axes[1][j], axes[2][k]]))
        np.testing.assert_allclose(
            [a[i, j, k] for a in result],
            expected,
            rtol=2e-6 if dtype == np.float32 else 2e-13,
            atol=2e-6 if dtype == np.float32 else 2e-13,
        )


def test_null_seeds_are_refined_distinct_and_offset_in_physical_units():
    pytest.importorskip("scipy")
    general, near, roots = make_seeds(12, 6, 0.02)
    assert roots.shape == (14, 3)
    assert max(np.linalg.norm(field_at(p)) for p in roots) < 1e-10
    np.testing.assert_allclose(
        np.linalg.norm(near.reshape(14, 6, 3) - roots[:, None, :], axis=2),
        0.02,
        atol=1e-15,
    )
    repeated = make_seeds(12, 6, 0.02)
    for a, b in zip((general, near, roots), repeated):
        np.testing.assert_array_equal(a, b)
