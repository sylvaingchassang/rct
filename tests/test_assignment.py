import numpy as np
from parameterized import parameterized
from numpy.testing import assert_array_almost_equal, assert_array_equal
from random import seed
from itertools import zip_longest

from assignment import clean_weights, draw_iid_assignment, \
    draw_shuffled_assignment, get_assignments_by_positions


def test_clean_weights():
    assert_array_almost_equal(clean_weights(.1), [.9, .1])
    assert_array_almost_equal(clean_weights([.1, .3]), [.6, .1, .3])
    assert_array_almost_equal(clean_weights([.1, .9]), [.1, .9])


@parameterized.expand([
    [.5, [1, 1, 1, 1, 0, 1, 0, 1, 1, 0]],
    [[.4, .6], [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]],
    [[.3, .2], [1, 1, 1, 1, 0, 1, 0, 2, 2, 0]]
])
def test_draw_iid(weights, expected):
    np.random.seed(0)
    assert_array_almost_equal(draw_iid_assignment(weights, 10), expected)


@parameterized.expand([
    [.5, [1, 1, 0, 1, 0, 0, 0, 0, 1, 1]],
    [[.4, .6], [1, 1, 0, 1, 0, 1, 0, 0, 1, 1]],
    [[.3, .2], [1, 2, 0, 1, 0, 0, 0, 0, 2, 1]]
])
def test_draw_shuffle(weights, expected):
    seed(0)
    assert_array_almost_equal(draw_shuffled_assignment(weights, 10), expected)


@parameterized.expand([
    [[0, 0, 0, 0, 1], [[0, 1, 2, 3], [4]]],
    [[1, 1, 0, 0, 0], [[2, 3, 4], [0, 1]]],
    [[0, 2, 0, 1], [[0, 2], [3], [1]]]
])
def test_get_assignments_by_position(assignment, expected):
    assert all(np.array_equal(a, b) for a, b in zip_longest(
        get_assignments_by_positions(assignment), expected))
