from hashlib import md5
import lazy_property
import pandas as pd
import numpy as np
import random
import abc

from assignment import draw_iid_assignment, draw_shuffled_assignment


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

    def as_frame(self, l):
        return pd.DataFrame(data=l, index=self.df.index, columns=['t'])


class RCT(RCTBase):

    @property
    def assignment_from_iid(self):
        np.random.seed(self.seed)
        return self._draw_iid_assignment()

    @property
    def assignment_from_shuffled(self):
        random.seed(self.seed)
        return self._draw_shuffled_assignment()


class KRerandomizedRCT:
    pass