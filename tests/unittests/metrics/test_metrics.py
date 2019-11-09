import unittest

import numpy

from pytolemaic.utils.metrics import Metrics, CustomMetrics


class TestMetrics(unittest.TestCase):

    def test_confidence_interval(self):
        yt = numpy.random.rand(10)
        d = numpy.random.rand(10)

        yp = yt + 0.1 * d
        mae = Metrics.call('mae', yt, yp)
        ci_low, ci_high = \
            Metrics.confidence_interval('mae', y_true=yt, y_pred=yp)

        self.assertTrue(ci_low < mae < ci_high)

    def test_auc(self):

        yt = (numpy.random.rand(10000)+0.5).astype(int)
        yp = numpy.random.rand(10000,1)

        auc = CustomMetrics.auc(y_true=yt, y_pred=yp)
        print(auc)
        self.assertLess(abs(auc-0.5), 5e-2)

        yt = (numpy.random.rand(10000) + 0.5).astype(int)
        yp = numpy.random.rand(10000, 2)

        auc = CustomMetrics.auc(y_true=yt, y_pred=yp)
        print(auc)
        self.assertLess(abs(auc - 0.5), 5e-2)

        yt = (numpy.random.rand(10000)*3).astype(int)
        yp = numpy.random.rand(10000, 3)

        auc = CustomMetrics.auc(y_true=yt, y_pred=yp)
        print(auc)
        self.assertLess(abs(auc - 0.5), 5e-2)

        yt = (numpy.random.rand(10000)*3).astype(int)
        yp = numpy.zeros((10000, 3))
        for k in range(3):
            yp[yt==k, k]=1

        auc = CustomMetrics.auc(y_true=yt, y_pred=yp)
        print(auc)
        self.assertLess(abs(auc - 1.0), 5e-2)