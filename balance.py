import abc
import pandas as pd
import numpy as np
from numbers import Number
from itertools import combinations
from statsmodels.formula.api import ols
from functools import partial

from .utils import NumericFunction
from .assignment import get_assignments_as_positions


def identity(x): return x


def max_absolute_value(x): return np.max(np.abs(x))


min_across_covariates = partial(np.min, axis=1)
max_across_covariates = partial(np.max, axis=1)
mean_across_covariates = partial(np.mean, axis=1)


class BalanceObjective:

    def __init__(self, cols=None):
        self._cols = cols

    def col_selection(self, df):
        return self._cols or df.columns

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


class MahalanobisBalance(BalanceObjective):
    def __init__(self, treatment_aggregator=identity, cols=None):
        self.treatment_aggregator = treatment_aggregator
        super().__init__(cols)

    def _balance_func(self, df, assignments):
        df_sel = df[self.col_selection(df)]
        inverse_cov = np.linalg.inv(df_sel.cov())
        means = [df_sel.loc[idx].mean() for idx in
                 self.assignment_indices(df_sel, assignments)]
        combs = list(combinations(range(len(means)), 2))
        mean_diffs = [means[a] - means[b] for a, b in combs]
        res = pd.DataFrame(data=[mean_diff @ inverse_cov @ mean_diff
                                 for mean_diff in mean_diffs],
                           index=['{}-{}'.format(a, b) for a, b in combs])
        return -self.treatment_aggregator(res)


def mahalanobis_balance(cols=None):
    return MahalanobisBalance(np.max, cols=cols).balance_func


class PValueBalance(BalanceObjective):
    def __init__(self, treatment_aggreagtor=identity,
                 covariate_aggregator=identity, cols=None):
        self.treatment_aggregator = treatment_aggreagtor
        self.covariate_aggregator = covariate_aggregator
        super().__init__(cols)

    def _balance_func(self, df, assignments):
        pvalues = dict((col, self.pvalues_by_col(
            col, df, assignments)) for col in self.col_selection(df))
        return self.covariate_aggregator(pd.DataFrame(pvalues))

    def pvalues_by_col(self, col, df, assignments):
        pv = self.treatment_aggregator(self.ols_col_on_treatment(
            col, df, assignments).pvalues.iloc[1:].values)
        if isinstance(pv, Number):
            pv = [pv]
        return pv

    def ols_col_on_treatment(self, col, df, assignments):
        t_dummies = pd.DataFrame(
            dict(('t{}'.format(i), df.index.isin(assignment))
                 for i, assignment in enumerate(assignments)))
        data = pd.concat((df, t_dummies), axis=1)
        sel_dummies = self._get_non_collinear_dummies(t_dummies)
        formula = '{} ~ 1 + {}'.format(col, ' + '.join(sel_dummies))
        return ols(formula, data=data).fit()

    def _get_non_collinear_dummies(self, t_dummies):
        is_collinear = int(t_dummies.sum(axis=1).mean()) == 1
        if is_collinear:
            return t_dummies.columns[:-1]
        else:
            return t_dummies.columns


def pvalues_report(df, assignments):
    if isinstance(assignments, pd.DataFrame):
        assignments = get_assignments_as_positions(assignments)
    report = PValueBalance().balance_func(df, assignments)
    idx = ['t{}'.format(i + 1) for i in range(len(report))]
    report.index = idx
    return report


def pvalue_balance(cols=None):
    return PValueBalance(
        treatment_aggreagtor=np.min,
        covariate_aggregator=min_across_covariates,
        cols=cols
    ).balance_func


class BlockBalance(BalanceObjective):

    def __init__(self, treatment_aggreagtor=identity,
                 covariate_aggregator=identity,
                 category_aggregator=max_absolute_value, cols=None):
        self.treatment_aggregator = treatment_aggreagtor
        self.covariate_aggregator = covariate_aggregator
        self.category_aggregator = category_aggregator
        super().__init__(cols)

    def _balance_func(self, df, assignments):
        relative_count_all = dict((col, self.relative_count_by_col(
            col, df, assignments)) for col in self.col_selection(df))
        return -self.covariate_aggregator(pd.DataFrame(relative_count_all))

    def relative_count_by_col(self, col, df, assignments):
        df_count = self.count_by_col(col, df, assignments)
        relative_dev = (df_count - df_count.median()) / df_count.median()
        res = self.treatment_aggregator(
            relative_dev.apply(self.category_aggregator, axis=1))
        return [res] if isinstance(res, Number) else res

    def count_by_col(self, col, df, assignments):
        cat = sorted(list(set(df[col])))
        idxs = self.assignment_indices(df, assignments)
        count = [[sum(df[col].loc[idx].eq(v)) for v in cat] for idx in idxs]
        return pd.DataFrame(data=count, columns=cat,
                            index=['t{}'.format(i) for i in range(len(idxs))])


def block_balance(cols=None):
    return BlockBalance(
        np.max, max_across_covariates, cols=cols).balance_func
