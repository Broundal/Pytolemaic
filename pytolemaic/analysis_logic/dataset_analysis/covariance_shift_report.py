import itertools
import numpy
from pytolemaic.utils.base_report import Report

class CovarianceShiftReport(Report):
    def __init__(self, covariance_shift:float, sensitivity_report:Report=None,
                 medium_lvl=0.7, high_lvl=0.95):
        self.medium_lvl = medium_lvl
        self.high_lvl = high_lvl
        self.covariance_shift = covariance_shift
        self.sensitivity = sensitivity_report

    def covariance_shift_insight(self):
        insights = []
        covariance_shift = numpy.round(self.covariance_shift, 2)
        if covariance_shift < self.medium_lvl:
            pass  # ok
        elif covariance_shift < self.high_lvl:
            insights.append(
                'Covariance shift ={:.2g} is not negligible, there may be some issue with train/test distribution'
                    .format(self.covariance_shift))
        else:
            insights.append("Covariance shift ={:.2g} is high! Double check the way you've defined the test set! "
                            "Check covariance_shift_sensitivity as it may indicate the source for the distribution "\
                            "differences.".format(self.covariance_shift))

        return insights

    def to_dict(self, printable=False):
        out = dict(covariance_shift=self.covariance_shift)
        if self.sensitivity is not None:
            out.update(dict(covariance_shift_sensitivity=self.sensitivity.to_dict(printable=printable)))
        return self._printable_dict(out, printable=printable)

    @classmethod
    def to_dict_meaning(cls):
        return dict(covariance_shift = "Measure whether the test and train comes from same distribution. "
                                       "High values (max 1) means the test and train come from different distribution. "
                                       "Low score (min 0) means they come from same distribution (=is ok).",
                    covariance_shift_sensitivity = "Sensitivity report for a model which classify samples to train/test sets. "
                                                   "This report is automatically generated only if covariance shift is high. "
                                                   "To generate it manually, see CovarianceShiftCalculator.")

    def plot(self):
        if self.sensitivity is not None:
            self.sensitivity.plot()

    def insights(self):

        return self._add_cls_name_prefix(
            itertools.chain(self.covariance_shift_insight(),
                            [self.sensitivity.most_important_feature_insight()] if self.sensitivity is not None else []))




