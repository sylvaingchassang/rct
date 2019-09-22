from hashlib import md5
from functools import partial
import random
import abc

import lazy_property
import pandas as pd
import numpy as np

from assignment import draw_iid_assignment, draw_shuffled_assignment, \
    get_assignments_by_positions
from balance import BalanceObjective


class RCTBase:
    def __init__(self, file_path, weights, seed=0):
        self.file_path = file_path
        self.weights = weights
        self.shift_seed = seed
        self.df = pd.read_csv(file_path)

    @lazy_property.LazyProperty
    def file_hash_int(self):
        hasher = md5()
        with open(self.file_path, 'rb') as fh:
            buf = fh.read()
            hasher.update(buf)
        return int(hasher.hexdigest(), 16)

    @lazy_property.LazyProperty
    def seed(self):
        return (self.file_hash_int + self.shift_seed) % 2 ** 32

    @property
    @abc.abstractmethod
    def assignment_from_iid(self):
        """"""

    def _draw_iid_assignment(self):
        return self.as_frame(
            draw_iid_assignment(self.weights, self.sample_size))

    @property
    @abc.abstractmethod
    def assignment_from_shuffled(self):
        """"""

    def _draw_shuffled_assignment(self):
        return self.as_frame(
            draw_shuffled_assignment(self.weights, self.sample_size))

    @property
    def sample_size(self):
        return len(self.df)

    def as_frame(self, assignment):
        return pd.DataFrame(
            data=assignment, index=self.df.index)


class RCT(RCTBase):

    @property
    def assignment_from_iid(self):
        np.random.seed(self.seed)
        return self._draw_iid_assignment()

    @property
    def assignment_from_shuffled(self):
        random.seed(self.seed)
        return self._draw_shuffled_assignment()


class KRerandomizedRCT(RCTBase):
    def __init__(self, objective: BalanceObjective,
                 file_path, weights, k=None, seed=0):
        super().__init__(file_path, weights, seed)
        self._balance = objective.balance_func
        self._k = k

    def balance(self, assignment):
        return float(self._balance(
            self.df, get_assignments_by_positions(assignment)).values)

    @property
    def k(self):
        if self._k is None:
            self._k = self.sample_size
        return self._k

    @property
    def assignment_from_iid(self):
        np.random.seed(self.seed)
        assignments = (draw_iid_assignment(self.weights, self.sample_size)
                       for _ in range(self.sample_size))
        return self._get_best_assignment(assignments)

    @property
    def assignment_from_shuffled(self):
        random.seed(self.seed)
        assignments = (draw_shuffled_assignment(
            self.weights, self.sample_size) for _ in range(self.sample_size))
        return self._get_best_assignment(assignments)

    def _get_best_assignment(self, assignments):
        return self.as_frame(max(assignments, key=self.balance))
