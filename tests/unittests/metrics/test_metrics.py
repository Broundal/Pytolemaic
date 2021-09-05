import unittest

import numpy

from pytolemaic.utils.metrics import Metrics, CustomMetrics


class TestMetrics(unittest.TestCase):

    def test_confidence_interval(self):
        rs = numpy.random.RandomState(0)
        yt = rs.rand(100)
        d = rs.rand(100)

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

    def test_mape(self):
        y1 = numpy.random.rand(100)
        y2 = numpy.random.rand(100)

        ref = Metrics.mape.function(y_true=y1, y_pred=y2)
        for yy1 in [y1.ravel(), y1.reshape(-1,1)]:
            for yy2 in [y2.ravel(), y2.reshape(-1, 1)]:
                self.assertEqual(ref, Metrics.mape.function(y_true=yy1, y_pred=yy2))