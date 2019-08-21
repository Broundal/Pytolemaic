import numpy

from pytolemaic.utils.constants import CLASSIFICATION, REGRESSION
from pytolemaic.utils.dmd import DMD
from pytolemaic.utils.general import GeneralUtils
from pytolemaic.utils.metrics import Metrics


class ScoringReport():
    def __init__(self, metrics: list = None):
        self.supported_metric = Metrics.supported_metrics()
        self.metrics = metrics or self.supported_metric
        self.metrics = [self.supported_metric[metric] for metric in
                        self.metrics
                        if metric in self.supported_metric]

    def score_report(self, model, dmd_test: DMD,
                     y_proba: numpy.ndarray = None,
                     y_pred: numpy.ndarray = None):

        score_report = {}

        model_support_dmd = GeneralUtils.dmd_supported(model, dmd_test)
        x_test = dmd_test if model_support_dmd else dmd_test.values
        y_true = dmd_test.target

        is_classification = GeneralUtils.is_classification(model)
        if is_classification:

            y_proba = y_proba or model.predict_proba(x_test)
            y_pred = y_pred or numpy.argmax(y_proba, axis=1)

            for metric in self.metrics:
                if not metric.ptype == CLASSIFICATION:
                    continue
                if metric.is_proba:
                    yp = y_proba
                else:
                    yp = y_pred

                score = metric.function(y_true, yp)
                ci_low, ci_high = Metrics.confidence_interval(metric,
                                                              y_true=y_true,
                                                              y_pred=y_pred,
                                                              y_proba=y_proba)
                score_report[metric.name] = dict(score=score,
                                                 ci_low=ci_low,
                                                 ci_high=ci_high)


        else:
            y_pred = y_pred or model.predict(x_test)
            for metric in self.metrics:
                if not metric.ptype == REGRESSION:
                    continue

                score = metric.function(y_true, y_pred)
                ci_low, ci_high = Metrics.confidence_interval(metric,
                                                              y_true=y_true,
                                                              y_pred=y_pred)
                score_report[metric.name] = dict(score=score,
                                                 ci_low=ci_low,
                                                 ci_high=ci_high)

        return score_report




if __name__ == '__main__':
    sr = ScoringReport()
    yt = numpy.random.rand(10)
    d = numpy.random.rand(10)
    print(Metrics.call('mae', yt, yt + 0.1 * d),
          sr._confidence_interval('mae', y_test=yt, y_pred=yt + 0.1 * d))
