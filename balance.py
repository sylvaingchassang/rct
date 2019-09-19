import abc
import pandas as pd
import numpy as np
from operator import eq
from numbers import Number


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


class BalanceFunction:

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


class MahalanobisBalance(BalanceFunction):

    def _balance_func(self, df, assignments):
        return sum(assignments) #df.mean()
