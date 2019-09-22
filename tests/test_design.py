from numpy.testing import TestCase, assert_array_almost_equal
from design import RCT
from os import path


class TestRCT(TestCase):
    def setUp(self) -> None:
        self.file = path.join(path.dirname(__file__), 'example_covariates.csv')
        self.rct = RCT(self.file, [.5, .5], 1)

    def test_hash_int(self):
        assert (self.rct.file_hash_int ==
                151671729980354795404869707092356732292)

    def test_seed(self):
        assert (self.rct.seed == 2705298821)

    def test_sample_size(self):
        assert (self.rct.sample_size == 100)

    def test_draw_iid(self):
        assert_array_almost_equal(self.rct.assignment_from_iid.mean(), 0.57)
        assert_array_almost_equal(self.rct.assignment_from_iid[:10].T,
                                  [[1, 1, 0, 0, 0, 1, 0, 1, 0, 1]])
        assert_array_almost_equal(self.rct.assignment_from_iid.mean(), 0.57)

    def test_draw_shuffle(self):
        assert_array_almost_equal(
            self.rct.assignment_from_shuffled.mean(), [.5])
        assert_array_almost_equal(self.rct.assignment_from_shuffled[:10].T,
                                  [[0, 1, 1, 1, 0, 1, 0, 0, 0, 0]])
        assert_array_almost_equal(self.rct.assignment_from_shuffled[:10].T,
                                  [[0, 1, 1, 1, 0, 1, 0, 0, 0, 0]])
