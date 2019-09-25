import pandas as pd
from numpy.testing import TestCase, assert_array_almost_equal
import numpy as np
from parameterized import parameterized

from ..design import RCT, KRerandomizedRCT
from ..balance import pvalues_report, MahalanobisBalance, PValueBalance, \
    min_across_covariates
from ..assignment import get_assignments_as_positions

from os import path

DATA_PATH = path.join(path.dirname(__file__), 'example_covariates.csv')
df = pd.read_csv(DATA_PATH)

pvalue_balance = PValueBalance(
    treatment_aggreagtor=np.min, covariate_aggregator=min_across_covariates)


class TestIntegration(TestCase):

    @parameterized.expand([
        [[.5, .5], pvalue_balance, [[50, 50], [50, 50]],
         [[[0.663702, 0.866303, 0.551186]], [[0.963649, 0.971983, 0.551186]]]],
        [[.3, .3, .4], pvalue_balance, [[40, 34, 26], [40, 30, 30]],
         [[[0.903671, 0.528733, 0.477888], [0.966614, 0.895375, 0.970118]],
          [[0.516997, 0.765729, 0.733228], [0.571203, 0.890846, 0.733228]]]]
    ])
    def test_krct(self, weights, balance, expected_count, expected_report):
        krct = KRerandomizedRCT(
            balance, DATA_PATH, weights, k=20, seed=0)

        iid_assignment = krct.assignment_from_iid
        shuffled_assignment = krct.assignment_from_shuffled

        assert_array_almost_equal(
            iid_assignment['t'].value_counts(), expected_count[0])
        assert_array_almost_equal(
            shuffled_assignment['t'].value_counts(), expected_count[1])
        assert_array_almost_equal(
            pvalues_report(df, get_assignments_as_positions(
                iid_assignment)), expected_report[0])
        assert_array_almost_equal(
            pvalues_report(df, get_assignments_as_positions(
                shuffled_assignment)), expected_report[1])
