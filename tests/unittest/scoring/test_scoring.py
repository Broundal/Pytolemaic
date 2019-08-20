import unittest

import numpy

from pytolemaic.analysis_logic.model_analysis.scoring.scoring import \
    ScoringReport
from pytolemaic.utils.metrics import Metrics


class TestSensitivity(unittest.TestCase):

    def test_confidence_interval(self):
        sr = ScoringReport()
        yt = numpy.random.rand(10)
        d = numpy.random.rand(10)

        yp = yt + 0.1 * d
        mae = Metrics.call('mae', yt, yp)
        ci_low, ci_high = \
            sr._confidence_interval('mae', y_test=yt, y_pred=yp)

        self.assertTrue(ci_low < mae < ci_high)
