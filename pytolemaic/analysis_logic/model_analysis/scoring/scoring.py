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
            yp = y_pred or numpy.argmax(y_proba, axis=1)

            for metric in self.metrics:
                if not metric.ptype == CLASSIFICATION:
                    continue
                if metric.is_proba:
                    score_report[metric.name] = metric.function(y_true,
                                                                y_proba)
                else:
                    score_report[metric.name] = metric.function(y_true, yp)

        else:
            yp = y_pred or model.predict(x_test)
            for metric in self.metrics:
                if not metric.ptype == REGRESSION:
                    continue
                score_report[metric.name] = metric.function(y_true, yp)

        return score_report

    def _confidence_interval(self, metric: str, y_test: numpy.ndarray,
                             y_proba: numpy.ndarray = None,
                             y_pred: numpy.ndarray = None,
                             low=0.25, high=0.75,
                             n_bags=20):
        metric = self.supported_metric[metric]
        if metric.is_proba:
            y_pred = y_proba

        rs = numpy.random.RandomState(0)

        scores = []
        for k in range(n_bags):
            bagging = rs.randint(0, len(y_test), len(y_test))
            scores.append(metric.function(y_test[bagging], y_pred[bagging]))

        return numpy.percentile(scores, q=[low * 100, high * 100])


if __name__ == '__main__':
    sr = ScoringReport()
    yt = numpy.random.rand(10)
    d = numpy.random.rand(10)
    print(Metrics.call('mae', yt, yt + 0.1 * d),
          sr._confidence_interval('mae', y_test=yt, y_pred=yt + 0.1 * d))
