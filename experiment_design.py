import abc
import pandas as pd
from operator import eq


def clean_str(s):
    return s.lower().rstrip()


class AbstractBalance(abc.ABC):

    @classmethod
    def from_name(cls, name):
        for sub in cls.__subclasses__():
            if eq(*map(clean_str, (name, sub.name))):
                return sub()

    @property
    @abc.abstractmethod
    def name(self):
        pass

    @abc.abstractmethod
    def __call__(self):
        pass


class MahalanobisBalance(AbstractBalance):

    @property
    def name(self):
        return 'mahalanobis'

    def __call__(self):
        pass


class DistributionBalance(AbstractBalance):
    pass


class BlockingBalance(AbstractBalance):
    pass


class StaticAssignment:
    def __init__(self, K):
        pass


class DynamicAssignment:
    pass
