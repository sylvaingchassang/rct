from numbers import Number
import abc
from collections import deque
import multiprocessing as mp
from bisect import insort


class NumericFunction:
    @classmethod
    def numerize(cls, f):
        return NumericFunction(f)

    def __init__(self, f):
        self.func = f

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __add__(self, other):
        def f(*args, **kwargs):
            return self(*args, **kwargs) + other(*args, **kwargs)
        return self.numerize(f)

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        def f(*args, **kwargs): return -self(*args, **kwargs)
        return self.numerize(f)

    def __sub__(self, other):
        def f(*args, **kwargs):
            return self(*args, **kwargs) - other(*args, **kwargs)
        return self.numerize(f)

    def __rsub__(self, other):
        def f(*args, **kwargs):
            return -self(*args, **kwargs) + other(*args, **kwargs)
        return self.numerize(f)

    def __mul__(self, other):
        if isinstance(other, Number):
            def f(*args, **kwargs): return self(*args, **kwargs) * other
            return self.numerize(f)
        else:
            def f(*args, **kwargs):
                return self(*args, **kwargs) * other(*args, **kwargs)
            return self.numerize(f)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __str__(self):
        return 'NumericFunction: number valued function'


class OrderedTupleBase:
    def __init__(self, *args):
        self.args = args

    @staticmethod
    @abc.abstractmethod
    def tuple_lt(args1, args2):
        """order over tuples"""

    def __gt__(self, other):
        return self.tuple_lt(self._args(other), self.args)

    def __lt__(self, other):
        return self.tuple_lt(self.args, self._args(other))

    def __eq__(self, other):
        return all(a == b for a, b in zip(self.args, self._args(other)))

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__,
                               ', '.join([str(a) for a in self.args]))

    def __iter__(self):
        return iter(self.args)

    @staticmethod
    def _args(other):
        return other.args if isinstance(other, OrderedTupleBase) else other


class LexTuple(OrderedTupleBase):
    @staticmethod
    def tuple_lt(args1, args2):
        for a, b in zip(args1, args2):
            try:
                if a < b:
                    return True
                elif a > b:
                    return False
            except ValueError:
                return False
            except TypeError:
                return False
        return False


class QuantileTarget:
    def __init__(self, q, objective_fun, generator_sample, len_generator):
        self.f = objective_fun
        self.generator = generator_sample
        self.num_q = q if q > 1 else int(q * len_generator)
        self.quantiles = deque([], maxlen=self.num_q)

    def compute_best(self):
        for s in self.generator:
            res = self.f(s)
            self._pop_insort(res, s)

    def _pop_insort(self, res, s):
        if len(self.quantiles) < self.num_q:
            insort(self.quantiles, LexTuple(res, s))
        elif (res, s) > self.quantiles[0]:
            self.quantiles.popleft()
            insort(self.quantiles, LexTuple(res, s))
