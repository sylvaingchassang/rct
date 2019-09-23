from numpy.testing import TestCase

from utils import NumericFunction, LexTuple, QuantileTarget


class TestNumericFunction(TestCase):
    def setUp(self):
        @NumericFunction.numerize
        def f(x):
            return 2 * x
        self.f = f

        def g(x):
            return 3 * x
        self.g = g

        @NumericFunction.numerize
        def f2(x, y):
            return x + y
        self.f2 = f2

        def g2(x, y):
            return x * y
        self.g2 = g2

    def test_type(self):
        assert isinstance(self.f, NumericFunction)
        assert not isinstance(self.g, NumericFunction)
        assert str(self.f) == 'NumericFunction: number valued function'

    def test_add(self):
        assert (self.f + self.g)(1) == 5 == (self.g + self.f)(1)

    def test_neg(self):
        assert (-self.f + self.g)(1) == 1 == (self.g - self.f)(1)
        assert (self.f - self.g)(1) == -1
        assert (-self.f)(1) == -2

    def test_multiply(self):
        assert (self.f * self.g)(1) == (self.g * self.f)(1) == 6
        assert (.6 * self.f + self.g)(1) == (self.g + .6 * self.f)(1) == 4.2

    def test_multi_args(self):
        assert (self.f2 + self.g2)(1, 2) == 5
        assert (self.f2 - self.g2)(1, 2) == 1
        assert (-self.f2)(1, 2) == -3
        assert (-self.f2 + self.g2)(1, 2) == -1


class TestLexOrderedTuple(TestCase):
    def test_repr(self):
        assert str(LexTuple(1, 2)) == 'LexTuple(1, 2)'

    def test_order_func(self):
        assert LexTuple.tuple_lt((0, 'b'), (1, 'a'))
        assert not LexTuple.tuple_lt((0, 'b'), (0, 'a'))
        assert not LexTuple.tuple_lt((4, 'b', 1), (1, 'c', 100))
        assert LexTuple.tuple_lt((-4, 'b', 1), (1, 'c', 100))
        assert not LexTuple.tuple_lt((1, 'c', 101), (1, 'c', 100))
        assert not LexTuple.tuple_lt((3, lambda x: x), (2, lambda y: y))

    def test_infix(self):
        assert not LexTuple(1, '1', 4) < LexTuple(1, '1', 3)
        assert LexTuple(1, 2) > LexTuple(0, 10)
        assert LexTuple(-1, 2) < LexTuple(0, 10)
        assert LexTuple(-1, 2) < (0, 10)
        assert not LexTuple(1, 2) < (1, 2)
        assert LexTuple(-1, 2) == LexTuple(-1, 2)
        assert LexTuple(-1, 2) == (-1, 2)


class TestQuantileTarget:

    def test_pop_insort(self):
        qs = QuantileTarget(.2, lambda x: x ** 2 - 1, range(10), 10)
        qs._pop_insort(0, 1)
        assert list(qs.quantiles) == [(0, 1)]
        qs._pop_insort(3, 2)
        assert list(qs.quantiles) == [(0, 1), (3, 2)]
        qs._pop_insort(2, 6)
        assert list(qs.quantiles) == [(2, 6), (3, 2)]
        qs._pop_insort(2, 'b')
        assert list(qs.quantiles) == [(2, 6), (3, 2)]

    def test_quantile_target(self):
        qs = QuantileTarget(.2, lambda x: -x ** 2 + 2 * x, range(10), 10)
        qs.compute_best()
        assert list(qs.quantiles) == [(0, 2), (1, 1)]
