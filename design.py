from hashlib import md5
import random
import abc

import lazy_property
import pandas as pd
import numpy as np

from assignment import draw_iid_assignment, draw_shuffled_assignment, \
    get_assignments_by_positions
from balance import BalanceObjective
from utils import QuantileTarget


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


class BalancedRCTBase(RCTBase):
    def __init__(self, objective: BalanceObjective,
                 file_path, weights, k=None, seed=0):
        super().__init__(file_path, weights, seed)
        self._balance = objective.balance_func
        self._k = k

    @property
    def k(self):
        if self._k is None:
            self._k = self.sample_size
        return self._k

    def balance(self, assignment):
        return float(self._balance(
            self.df, get_assignments_by_positions(assignment)).values)

    def assignment_generator(self, draw_fun):
        return (draw_fun(self.weights, self.sample_size) for _ in range(
            self.k))


class KRerandomizedRCT(BalancedRCTBase):
    @property
    def assignment_from_iid(self):
        np.random.seed(self.seed)
        assignments = self.assignment_generator(draw_iid_assignment)
        return self._get_best_assignment(assignments)

    @property
    def assignment_from_shuffled(self):
        random.seed(self.seed)
        assignments = self.assignment_generator(draw_shuffled_assignment)
        return self._get_best_assignment(assignments)

    def _get_best_assignment(self, assignments):
        return self.as_frame(max(assignments, key=self.balance))


class QuantileTargetingRCT(BalancedRCTBase):
    def __init__(self, objective: BalanceObjective, file_path, weights,
                 quantile_target=None, seed=0, num_monte_carlo=1000):
        super().__init__(file_path, weights, seed)
        self._balance = objective.balance_func
        self.quantile_target = quantile_target
        self._k = num_monte_carlo

    def balance(self, assignment):
        return float(self._balance(
            self.df, get_assignments_by_positions(assignment)).values)

    @property
    def assignment_from_iid(self):
        np.random.seed(self.seed)
        assignments = self.assignment_generator(draw_iid_assignment)
        return self.get_target_assignments(assignments)

    @property
    def assignment_from_shuffled(self):
        random.seed(self.seed)
        assignments = self.assignment_generator(draw_shuffled_assignment)
        return self.get_target_assignments(assignments)

    def get_target_assignments(self, assignments):
        qtargets = QuantileTarget(
            self.quantile_target, self.balance, assignments)
        qtargets.compute_best()
        _, selected_assignment = random.choice(qtargets.quantiles)
        return selected_assignment
