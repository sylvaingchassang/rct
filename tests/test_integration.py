import pandas as pd
from numpy.testing import TestCase, assert_array_almost_equal
import numpy as np
from parameterized import parameterized

from ..design import RCT, KRerandomizedRCT
from ..balance import pvalues_report, mahalanobis_balance, pvalue_balance, \
    block_balance
from ..assignment import get_assignments_as_positions

from os import path

DATA_PATH = path.join(path.dirname(__file__), 'test_data')
COVARIATES_PATH = path.join(DATA_PATH, 'example_covariates.csv')

cov_df = pd.read_csv(COVARIATES_PATH)


class TestIntegration(TestCase):

    @staticmethod
    def _save_assignment_details(df, assignment, name):
        test_data_path = path.join(DATA_PATH, name)
        report = pvalues_report(df, assignment).values.flatten()
        d = {'count': list(assignment['t'].value_counts()),
             'report': list(report)}
        with open(test_data_path, 'w+') as fh:
            fh.write(str(d))

    @staticmethod
    def _check_assignment_details(df, assignment, name):
        test_data_path = path.join(DATA_PATH, name)
        with open(test_data_path, 'r') as fh:
            details = eval(fh.read())
        report = pvalues_report(df, get_assignments_as_positions(
            assignment)).values.flatten()
        assert_array_almost_equal(
            assignment['t'].value_counts(), details['count'])
        assert_array_almost_equal(report, details['report'])

    def assert_assignment_matches(self, df, assignment, name, generate=False):
        if generate:
            self._save_assignment_details(df, assignment, name)
        else:
            self._check_assignment_details(df, assignment, name)

    @parameterized.expand([
        [[.5, .5], pvalue_balance(), 'krct_2_pvalue'],
        [[.3, .3, .4], pvalue_balance(), 'krct_3_pvalue'],
        [[.5, .5], mahalanobis_balance(), 'krct_2_mahalanobis'],
        [[.3, .3, .4], mahalanobis_balance(), 'krct_3_mahalanobis'],
        [[.5, .5], block_balance(['C']), 'krct_2_block'],
        [[.3, .3, .4], block_balance(['C']), 'krct_3_block'],
        [[.5, .5], mahalanobis_balance(['A', 'B']) + block_balance(['C']),
         'krct_2_mixed'],
        [[.3, .3, .4],  mahalanobis_balance(['A', 'B']) + block_balance(['C']),
         'krct_3_mixed']
    ])
    def test_krct(self, weights, balance, name):
        krct = KRerandomizedRCT(
            balance, COVARIATES_PATH, weights, k=20, seed=0)
        for i, assignment in enumerate([krct.assignment_from_iid,
                                        krct.assignment_from_shuffled]):
            self.assert_assignment_matches(
                cov_df, assignment, '{}_{}'.format(name, i))

    @parameterized.expand([
        [[.5, .5], 'rct_2'],
        [[.3, .3, .4],  'rct_3'],
    ])
    def test_rct(self, weights, name):
        rct = RCT(COVARIATES_PATH, weights, seed=0)
        for i, assignment in enumerate([rct.assignment_from_iid,
                                        rct.assignment_from_shuffled]):
            self.assert_assignment_matches(
                cov_df, assignment, '{}_{}'.format(name, i))
