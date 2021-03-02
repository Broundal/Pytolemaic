import unittest
from pprint import pprint

import numpy

from pytolemaic import FeatureTypes
from pytolemaic.analysis_logic.dataset_analysis.covriance_shift import CovarianceShift
from pytolemaic.utils.dmd import DMD


class TestCovarianceShift(unittest.TestCase):

    def _gen_data(self, seed, offset=0.):
        rs = numpy.random.RandomState(seed)
        x = rs.randn(1000, 10)

        x[:, 0:5] = numpy.round(x[:, 0:5], 0)
        x[0, 0] = 10
        x[0, 9] = 10
        x[1, 9] = -4

        x[:,7] += offset
        y = numpy.copy(x[:, 0])

        x[10, :] = numpy.nan
        x[:, 1] = numpy.nan
        x[:700, 2] = numpy.nan

        return DMD(x=x, y=y, columns_meta={
          DMD.FEATURE_TYPES: 5 * [FeatureTypes.categorical] + 5 * [FeatureTypes.numerical]})

    def setUp(self) -> None:
        self.train = self._gen_data(0)
        self.same_dist_test = self._gen_data(1)
        self.mid_dist_test = self._gen_data(1, offset=1)
        self.dif_dist_test = self._gen_data(1, offset=3)

    def test_same_dist(self):
        covariance = CovarianceShift()
        covariance.calc_covariance_shift(dmd_train=self.train, dmd_test=self.same_dist_test)

        print(covariance.covariance_shift)
        assert covariance.covariance_shift < 0.03

        pprint(covariance.covariance_shift_report().to_dict())
        pprint(covariance.covariance_shift_report().insights())
        pprint(covariance.covariance_shift_report().plot())

    def test_mid_dist(self):
        covariance = CovarianceShift()
        covariance.calc_covariance_shift(dmd_train=self.train, dmd_test=self.mid_dist_test)

        print(covariance.covariance_shift)
        assert covariance.covariance_shift > 0.3
        assert covariance.covariance_shift < 0.6

        pprint(covariance.covariance_shift_report().to_dict())
        pprint(covariance.covariance_shift_report().insights())
        pprint(covariance.covariance_shift_report().plot())

    def test_high_dist(self):

        covariance = CovarianceShift()
        covariance.calc_covariance_shift(dmd_train=self.train, dmd_test=self.dif_dist_test)

        print(covariance.covariance_shift)

        assert covariance.covariance_shift > 0.75

        pprint(covariance.covariance_shift_report().to_dict())
        pprint(covariance.covariance_shift_report().insights())
        pprint(covariance.covariance_shift_report().plot())



if __name__ == '__main__':
    a=TestCovarianceShift()
    a.setUp()
    a.test_high_dist()
    from matplotlib import pyplot as plt
    plt.show()