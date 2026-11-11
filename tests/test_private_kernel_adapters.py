"""Keep the tuple-based developer inspection adapters usable outside hot loops."""

import numpy as np
import pytest
from fieldline_tracing_proto import tracing as T


def field(vector=(0.0, 0.0, 1.0), bmin=0.0):
    g = np.array([-1.0, 1.0])
    shape = (2, 2, 2)
    return (
        *[np.full(shape, v) for v in vector],
        g,
        g,
        g,
        np.zeros(3, dtype=np.bool_),
        np.ones(3, dtype=np.bool_),
        np.full(3, 0.5),
        np.zeros(3),
        bmin,
        tuple(np.empty(0, dtype=np.int64) for _ in range(3)),
    )


@pytest.mark.parametrize(
    "point,bmin,code",
    [(np.zeros(3), 0.0, 0), (np.array([2.0, 0.0, 0.0]), 0.0, 6), (np.zeros(3), 6.0, 8)],
)
def test_old_rhs_adapters_have_the_original_signature_and_status(point, bmin, code):
    f = field((3.0, 4.0, 0.0), bmin)
    x, y, z, status = T._rhs_xyz(f, point[0], point[1], point[2])
    vector, other = T._rhs(f, point)
    assert status == other == code
    np.testing.assert_array_equal(vector, [x, y, z])
    if code == 0:
        np.testing.assert_allclose(vector, [0.6, 0.8, 0.0], atol=1e-15)


@pytest.mark.parametrize("method", ["rk2", "rk45"])
def test_old_trial_adapters_remain_callable(method):
    f = field()
    p = np.zeros(3)
    u = np.array([0.0, 0.0, 1.0])
    c = np.empty(5)
    split = np.empty(4)
    roots = np.empty(10)
    if method == "rk2":
        r = T._rk2_trial(f, p, 0.1, u, c, split, roots)
    else:
        r = T._rk45_trial(
            f,
            p,
            0.1,
            u,
            np.full(3, 1e-7),
            np.empty((7, 3)),
            np.empty(3),
            c,
            split,
            roots,
        )
    np.testing.assert_allclose(r[0], [0.0, 0.0, 0.1], atol=1e-15)
    assert r[3] == 0 and r[6] == 0


@pytest.mark.parametrize("method", ["rk2", "rk45"])
def test_old_branch_adapter_retains_count_only_behavior(method):
    options = np.array(
        [
            method == "rk45",
            0.1,
            10,
            1,
            1e-12,
            0.1,
            0.2,
            100,
            1e-10,
            1e-7,
            1e-7,
            1e-7,
            0,
            1,
            1.0,
        ],
        dtype=float,
    )
    empty = np.empty(0)
    result = T._branch(
        field(), np.zeros(3), options, 1.0, empty, empty, empty, empty, 0, 1, False, 0
    )
    assert result[0] == 2 and result[1] == 11 and result[2] == 2 and result[3] == 0.2
    np.testing.assert_allclose(result[6], [0.0, 0.0, 0.2], atol=1e-15)
