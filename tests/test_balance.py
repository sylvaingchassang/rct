from numpy.testing import TestCase, assert_array_almost_equal, \
    assert_almost_equal
import pandas as pd
import numpy as np
from balance import NumericFunction, BalanceFunction, MahalanobisBalance


class TestNumericFunction(TestCase):
    def setUp(self):
        @NumericFunction.numerize
        def f(x):
            return 2 * x
        self.f = f

        def g(x):
            return 3 * x
        self.g = g

    def test_type(self):
        assert isinstance(self.f, NumericFunction)
        assert not isinstance(self.g, NumericFunction)
        assert str(self.f) == 'NumericFunction: number valued function'

    def test_add(self):
        assert (self.f + self.g)(1) == 5 == (self.g + self.f)(1)

    def test_multiply(self):
        assert (self.f * self.g)(1) == (self.g * self.f)(1) == 6
        assert (.6 * self.f + self.g)(1) == (self.g + .6 * self.f)(1) == 4.2


class TestBalance(TestCase):
    def setUp(self) -> None:
        np.random.seed(0)
        self.df = pd.DataFrame(
            data=np.random.rand(10, 2),
            columns=['a', 'b'])
        self.assignment = set(
            np.random.choice(self.df.index, size=5, replace=False))

    def test_balance_function(self):
        assert_array_almost_equal(
            BalanceFunction._idxs_from_assignment(
                self.df, [self.assignment, [2, 3]]),
            [[0, 0, 0, 0, 1, 1, 1, 1, 0, 1], [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]]
        )
        assert_array_almost_equal(
            BalanceFunction.assignment_indices(
                self.df, [self.assignment, [2, 3]]),
            [[0, 0, 0, 0, 1, 1, 1, 1, 0, 1], [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
             [1, 1, 0, 0, 0, 0, 0, 0, 1, 0]]
        )
        assert_array_almost_equal(
            BalanceFunction.assignment_indices(
                self.df, [np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1])]),
            [[0, 0, 0, 0, 1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 0, 0, 0, 0, 1, 0]]
        )

    def test_mahalanobis(self):
        maha = MahalanobisBalance().balance_func
        assert isinstance(maha, NumericFunction)
        assert_array_almost_equal(
            maha(self.df, self.assignment), sum(self.assignment))

