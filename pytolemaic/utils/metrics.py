import functools

import sklearn.metrics

from pytolemaic.utils.constants import REGRESSION, CLASSIFICATION


class Metric():
    def __init__(self, name, function, ptype, is_proba=False, is_loss=False):
        self.name = name
        self.function = function
        self.ptype = ptype
        self.is_proba = is_proba
        self.is_loss = is_loss


class Metrics():
    r2 = Metric(name='r2',
                function=sklearn.metrics.r2_score,
                ptype=REGRESSION)

    mae = Metric(name='mae',
                 function=sklearn.metrics.mean_absolute_error,
                 ptype=REGRESSION,
                 is_loss=True)

    # sklearn's auc does not support multiclass
    # auc = Metric(name='auc',
    #                   function=functools.partial(sklearn.metrics.roc_auc_score, average='macro'),
    #                   ptype=CLASSIFICATION,
    #                   is_proba=True)

    recall = Metric(name='recall',
                    function=functools.partial(sklearn.metrics.recall_score,
                                               average='macro'),

                    ptype=CLASSIFICATION)

    @classmethod
    def call(cls, metric, y_true, y_pred, **kwargs):
        metric = cls.supported_metrics()[metric]
        return metric.function(y_true, y_pred, **kwargs)

    @classmethod
    def supported_metrics(cls):
        return {m.name: m for m in
                [Metrics.r2, Metrics.recall, Metrics.mae]}

    @classmethod
    def metric_as_loss(cls, value, metric):
        if cls.supported_metrics()[metric].is_loss:
            return value
        else:
            return 1 - value

    @classmethod
    def metric_as_score(cls, value, metric):
        if not cls.supported_metrics()[metric].is_loss:
            return value
        else:
            return 1 - value
