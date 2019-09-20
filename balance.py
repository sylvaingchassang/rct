import abc
import pandas as pd
import numpy as np
from operator import eq
from numbers import Number
from itertools import combinations
from statsmodels.formula.api import ols
from functools import partial


class NumericFunction:
    @classmethod
    def numerize(cls, f):
        return NumericFunction(f)

    def __init__(self, f):
        self.func = f

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __add__(self, other):
        return self.numerize(lambda x: self(x) + other(x))

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, Number):
            return self.numerize(lambda x: self(x) * other)
        else:
            return self.numerize(lambda x: self(x) * other(x))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __str__(self):
        return 'NumericFunction: number valued function'


class BalanceObjective:

    @property
    def balance_func(self):
        return NumericFunction.numerize(self._balance_func)

    @abc.abstractmethod
    def _balance_func(self, df, assignments):
        """"""

    @classmethod
    def assignment_indices(cls, df, assignments):
        idxs = cls._idxs_from_assignment(df, assignments)
        return cls._append_complementary_assignment(idxs)

    @classmethod
    def _idxs_from_assignment(cls, df, assignments):
        if len(assignments[0]) == len(df.index):
            return assignments
        else:
            return [df.index.isin(a) for a in assignments]

    @classmethod
    def _append_complementary_assignment(cls, idxs):
        total_assignments = np.add(*idxs) if len(idxs) > 1 else idxs[0]
        if not min(total_assignments):
            idxs.append(np.logical_not(total_assignments))
        return idxs


identity = lambda x: x
min_across_covariates = partial(np.min, axis=1)


class MahalanobisBalance(BalanceObjective):
    def __init__(self, treatment_aggregator=identity):
        self.treatment_aggregator = treatment_aggregator

    def _balance_func(self, df, assignments):
        inverse_cov = np.linalg.inv(df.cov())
        means = [df.loc[idx].mean() for idx in
                 self.assignment_indices(df, assignments)]
        combs = list(combinations(range(len(means)), 2))
        mean_diffs = [means[a] - means[b] for a, b in combs]
        res = pd.DataFrame(data=[mean_diff @ inverse_cov @ mean_diff
                                 for mean_diff in mean_diffs],
                           index=['{}-{}'.format(a, b) for a, b in combs])
        return -self.treatment_aggregator(res)


class PValueBalance(BalanceObjective):
    def __init__(self, treatment_aggreagtor=identity,
                 covariate_aggregator=identity):
        self.treatment_aggregator = treatment_aggreagtor
        self.covariate_aggregator = covariate_aggregator

    def _balance_func(self, df, assignments):
        pvalues = dict((col, self.pvalues_by_col(
            col, df, assignments)) for col in df.columns)
        return self.covariate_aggregator(pd.DataFrame(pvalues))

    def pvalues_by_col(self, col, df, assignments):
        pv = self.treatment_aggregator(self.ols_col_on_treatment(
            col, df, assignments).pvalues.iloc[1:].values)
        if isinstance(pv, Number):
            pv = [pv]
        return pv

    @staticmethod
    def ols_col_on_treatment(col, df, assignments):
        t_dummies = pd.DataFrame(
            dict(('t{}'.format(i), df.index.isin(assignment))
                 for i, assignment in enumerate(assignments)))
        data = pd.concat((df, t_dummies), axis=1)
        formula = '{} ~ 1 + {}'.format(col, ' + '.join(t_dummies.columns))
        return ols(formula, data=data).fit()
