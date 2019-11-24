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


class ScoringFullReport():
    def __init__(self, metric_reports: [ScoringMetricReport], quality: float):
        self._quality = quality
        self._metric_scores = metric_reports
        self._metric_scores_dict = {r.metric: r for r in self._metric_scores}

    @property
    def quality(self):
        return self._quality

    @property
    def metric_scores(self) -> dict:
        return self._metric_scores_dict
