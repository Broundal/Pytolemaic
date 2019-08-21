import unittest

import numpy

from pytolemaic.utils.metrics import Metrics


class TestMetrics(unittest.TestCase):

    def test_confidence_interval(self):
        yt = numpy.random.rand(10)
        d = numpy.random.rand(10)

        yp = yt + 0.1 * d
        mae = Metrics.call('mae', yt, yp)
        ci_low, ci_high = \
            Metrics.confidence_interval('mae', y_true=yt, y_pred=yp)

        self.assertTrue(ci_low < mae < ci_high)
