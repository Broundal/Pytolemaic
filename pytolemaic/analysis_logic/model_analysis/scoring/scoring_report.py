from sklearn.metrics.classification import confusion_matrix

from pytolemaic.utils.metrics import Metrics

class ConfusionMatrixReport():
    def __init__(self, y_true, y_pred, labels=None):
        self._confusion_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels).tolist()
        self._labels = labels

    @property
    def labels(self):
        return self._labels

    @property
    def confusion_matrix(self):
        return self._confusion_matrix

    def to_dict(self):
        return dict(confusion_matrix=self.confusion_matrix,
                    labels=self.labels)


class ScatterReport():
    def __init__(self, y_true, y_pred):
        self._y_true = y_true
        self._y_pred = y_pred

    @property
    def y_true(self):
        return self._y_true

    @property
    def y_pred(self):
        return self._y_pred

    def to_dict(self):
        return dict(y_true=self.y_true,
                    y_pred=self.y_pred)


class ScoringMetricReport():
    def __init__(self, metric, value, ci_low, ci_high):
        self._metric = metric
        self._value = value
        self._ci_low = ci_low
        self._ci_high = ci_high

    def to_dict(self):
        return dict(
            metric=self.metric,
            value=self.value,
            ci_low=self.ci_low,
            ci_high=self.ci_high,
            ci_ratio=self.ci_ratio,
        )

    @property
    def metric(self):
        return self._metric

    @property
    def value(self):
        return self._value

    @property
    def ci_low(self):
        return self._ci_low

    @property
    def ci_high(self):
        return self._ci_high

    @property
    def ci_ratio(self):
        # large ci difference is more of a concern if score is high
        ci_low = Metrics.metric_as_loss(value=self.ci_low, metric=self.metric)
        ci_high = Metrics.metric_as_loss(value=self.ci_high, metric=self.metric)

        ci_ratio = abs(ci_high - ci_low) / (ci_high + ci_low) * 2
        return ci_ratio



class ScoringFullReport():
    def __init__(self, metric_reports: [ScoringMetricReport], separation_quality: float, confusion_matrix=None, scatter=None):
        self._separation_quality = separation_quality
        self._metric_scores_dict = {r.metric: r for r in metric_reports}
        self._confusion_matrix = confusion_matrix
        self._scatter = scatter


    def to_dict(self):
        metric_scores = {k: v.to_dict() for k, v in self.metric_scores.items()}
        for v in metric_scores.values():
            v.pop('metric', None)

        return dict(
            metric_scores=metric_scores,
            separation_quality=self.separation_quality,
            scatter=None if self.scatter is None else self.scatter.to_dict(),
            confusion_matrix=None if self.confusion_matrix is None else self.confusion_matrix.to_dict()
        )

    @property
    def separation_quality(self):
        return self._separation_quality

    @property
    def metric_scores(self) -> dict:
        return self._metric_scores_dict

    @property
    def confusion_matrix(self)->ConfusionMatrixReport:
        return self._confusion_matrix

    @property
    def scatter(self)->ScatterReport:
        return self._scatter
