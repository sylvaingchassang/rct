import pandas as pd
import numpy as np
from parameterized import parameterized
from numpy.testing import TestCase, assert_array_almost_equal, \
    assert_almost_equal, assert_array_equal

from balance import NumericFunction, BalanceObjective, MahalanobisBalance, \
    PValueBalance, BlockBalance, min_across_covariates, identity, \
    pvalues_report, max_absolute_value, max_across_covariates


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

    def test_neg(self):
        assert (-self.f + self.g)(1) == 1 == (self.g - self.f)(1)
        assert (self.f - self.g)(1) == -1
        assert (-self.f)(1) == -2

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
        df_cat = (0. * (self.df['a'] < .3) + 1. * self.df['a'].between(.3, .6)
                  + 2. * (self.df['a'] > .6))
        self.df_cat = df_cat.to_frame('cat1')
        self.df_cat['cat2'] = 1. * (self.df['b'] < .5)

    def test_balance_objective(self):
        assert_array_almost_equal(
            BalanceObjective._idxs_from_assignment(
                self.df, [self.assignment, [2, 3]]),
            [[0, 0, 0, 0, 1, 1, 1, 1, 0, 1], [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]])
        assert_array_almost_equal(
            BalanceObjective.assignment_indices(
                self.df, [self.assignment, [2, 3]]),
            [[0, 0, 0, 0, 1, 1, 1, 1, 0, 1], [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
             [1, 1, 0, 0, 0, 0, 0, 0, 1, 0]])
        assert_array_almost_equal(
            BalanceObjective.assignment_indices(
                self.df, [np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1])]),
            [[0, 0, 0, 0, 1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 0, 0, 0, 0, 1, 0]])

    def test_mahalanobis_id(self):
        maha = MahalanobisBalance().balance_func
        assert isinstance(maha, NumericFunction)
        assert_array_almost_equal(
            maha(self.df, [self.assignment, [2, 3]]).T,
            [[-1.1719512, -0.9992989, -0.0849675]])
        assert_array_almost_equal(
            maha(self.df, [self.assignment]), [[-1.047968]])

    def test_mahalanobis_max(self):
        maha_max = MahalanobisBalance(np.max).balance_func
        assert_almost_equal(
            maha_max(self.df, [self.assignment, [2, 3]]), [-1.1719512])

    def test_pvalues(self):
        pv_balance = PValueBalance().balance_func
        assert_array_almost_equal(
            pv_balance(self.df, [self.assignment, [2, 3]]),
            [[0.320395, 0.523023], [0.892326, 0.790063]])

    @parameterized.expand([
        [np.min, min_across_covariates, 0.320395],
        [identity, min_across_covariates, [0.320395, 0.790063]],
        [np.min, identity, [[0.320395, 0.523023]]],
    ])
    def test_pvalue_agg(self, t_agg, c_agg, expected):
        pv_balance = PValueBalance(
            treatment_aggreagtor=t_agg,
            covariate_aggregator=c_agg).balance_func
        assert_array_almost_equal(
            pv_balance(self.df, [self.assignment, [2, 3]]), expected)

    def test_pvalues_report(self):
        report = pvalues_report(self.df, [self.assignment, [2, 3]])
        assert_array_equal(report.columns, ['a', 'b'])
        assert_array_equal(report.index, ['t1', 't2'])
        assert_array_almost_equal(
            report, [[0.320395, 0.523023], [0.892326, 0.790063]])

    def test_count_by_col(self):
        assert_array_almost_equal(
            BlockBalance().count_by_col('cat1', self.df_cat,
                                        [self.assignment]),
            [[1, 1, 3], [1, 3, 1]]
        )

    def test_relative_count_by_col(self):
        assert_array_almost_equal(BlockBalance().relative_count_by_col(
            'cat1', self.df_cat, [self.assignment]),
            [0.5, 0.5]
        )

    def test_block_balance(self):
        block_res = BlockBalance().balance_func(self.df_cat, [self.assignment])
        assert_array_equal(block_res.columns, ['cat1', 'cat2'])
        assert_array_almost_equal(
            block_res, [[-.5, -1], [-.5, -1.]]
        )

    @parameterized.expand([
        [max_absolute_value, np.max, identity, [[-.5, -1]]],
        [np.abs, np.max, identity, [[-0, -.25], [-.5, -1], [-.5, np.NAN]]],
        [max_absolute_value, np.max, max_across_covariates, [-1]]
    ])
    def test_block_balance_agg(self, cat_agg, treat_agg, cov_agg, expected):
        block_res = BlockBalance(
            covariate_aggregator=cov_agg, category_aggregator=cat_agg,
            treatment_aggreagtor=treat_agg
        ).balance_func(self.df_cat, [self.assignment])
        assert_array_almost_equal(
            block_res, expected
        )
