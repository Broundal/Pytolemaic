from pytolemaic.utils.metrics import Metrics


class ScoringMetricReport():
    def __init__(self, metric, value, ci_low, ci_high):
        self._metric = metric
        self._value = value
        self._ci_low = ci_low
        self._ci_high = ci_high

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
        score = Metrics.metric_as_loss(value=self.value, metric=self.metric)
        ci_ratio = (self.ci_high - self.ci_low) / score
        return ci_ratio



class ScoringFullReport():
    def __init__(self, metric_reports: [ScoringMetricReport], separation_quality: float):
        self._separation_quality = separation_quality
        self._metric_scores = metric_reports
        self._metric_scores_dict = {r.metric: r for r in self._metric_scores}

    @property
    def separation_quality(self):
        return self._separation_quality

    @property
    def metric_scores(self) -> dict:
        return self._metric_scores_dict
