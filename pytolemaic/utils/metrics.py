import functools

import numpy
from sklearn import metrics as sklearn_metrics

from pytolemaic.utils.constants import REGRESSION, CLASSIFICATION


class Metric():
    def __init__(self, name, function, ptype, full_name=None, is_proba=False, is_loss=False):
        self.name = name
        self.full_name = full_name or self.name
        self.function = function
        self.ptype = ptype
        self.is_proba = is_proba
        self.is_loss = is_loss

class CustomMetrics:
    @classmethod
    def auc(cls, y_true, y_pred):
        if y_pred.shape[1] == 1:
            return sklearn_metrics.roc_auc_score(y_true=y_true, y_score=y_pred)
        else:
            auc_list = []
            for i in range(y_pred.shape[1]):
                y_true_i = y_true == i
                y_pred_i = y_pred[:, i]
                if all(y_true_i) or all(~y_true_i):
                    auc_list.append(0)
                else:
                    auc_list.append(sklearn_metrics.roc_auc_score(y_true=y_true_i,
                                                                  y_score=y_pred_i))

            return float(numpy.mean(auc_list))

    @classmethod
    def rmse(cls, y_true, y_pred):
        return numpy.sqrt(sklearn_metrics.mean_squared_error(y_true=y_true, y_pred=y_pred))

    @classmethod
    def normalized_rmse(cls, y_true, y_pred):
        return numpy.sqrt(sklearn_metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)) / (
                numpy.std(y_true) + 1e-10)

    @classmethod
    def mape(cls, y_true, y_pred, eps=1e-100):
        # same as sklearn's mean_absolute_percentage_error.
        # Re-writing it as custom function for backwards compatibility

        # explicit ravel to avoid bug when shapes are not the same (1D vs 2D)
        y_true = numpy.asarray(y_true).ravel()
        y_pred = numpy.asarray(y_pred).ravel()

        delta_ratio = numpy.abs(y_true-y_pred) / numpy.clip(numpy.abs(y_true), eps, 1e100)
        return numpy.mean(delta_ratio)


class Metrics():
    r2 = Metric(name='r2',
                full_name="Coefficient of Determination (R^2)",
                function=sklearn_metrics.r2_score,
                ptype=REGRESSION)

    mae = Metric(name='mae',
                 full_name="Mean Absolute Error",
                 function=sklearn_metrics.mean_absolute_error,
                 ptype=REGRESSION,
                 is_loss=True)

    mse = Metric(name='mse',
                 full_name="Mean Squared Error",
                 function=sklearn_metrics.mean_squared_error,
                 ptype=REGRESSION,
                 is_loss=True)

    rmse = Metric(name='rmse',
                  full_name="Root Mean Squared Error",
                  function=CustomMetrics.rmse,
                  ptype=REGRESSION,
                  is_loss=True)

    mape = Metric(name='mape',
                  full_name="Mean Absolute Percentage Error",
                  function=CustomMetrics.mape, # == sklearn_metrics.mean_absolute_percentage_error
                  ptype=REGRESSION,
                  is_loss=True)

    normalized_rmse = Metric(name='normalized_rmse',
                             full_name="Normalized Root Mean Squared Error",
                             function=CustomMetrics.normalized_rmse,
                             ptype=REGRESSION,
                             is_loss=True)

    # sklearn's auc does not support multiclass
    auc = Metric(name='auc',
                 full_name="Area Under ROC curve",
                 function=CustomMetrics.auc,
                 ptype=CLASSIFICATION,
                 is_proba=True)

    recall = Metric(name='recall',
                    full_name="Mean Recall Score",
                    function=functools.partial(sklearn_metrics.recall_score,
                                               average='macro'),

                    ptype=CLASSIFICATION)

    @classmethod
    def call(cls, metric, y_true, y_pred, **kwargs):
        metric = cls.supported_metrics()[metric]
        return metric.function(y_true, y_pred, **kwargs)

    @classmethod
    def supported_metrics(cls):
        return {m.name: m for m in
                [Metrics.r2, Metrics.recall, Metrics.mae, Metrics.auc,
                 Metrics.mse, Metrics.rmse, Metrics.normalized_rmse,
                 Metrics.mape]}

    @classmethod
    def metric_as_loss(cls, value, metric):
        if isinstance(metric, str):
            metric = cls.supported_metrics()[metric]

        if metric.is_loss:
            return value
        else:
            return 1 - value

    @classmethod
    def metric_as_score(cls, value, metric):
        if isinstance(metric, str):
            metric = cls.supported_metrics()[metric]

        if not metric.is_loss:
            return value
        else:
            return 1 - value

    @classmethod
    def confidence_interval(cls, metric: str, y_true: numpy.ndarray,
                            y_proba: numpy.ndarray = None,
                            y_pred: numpy.ndarray = None,
                            low=0.25, high=0.75,
                            n_bags=20):
        if isinstance(metric, str):
            metric = cls.supported_metrics()[metric]

        if metric.is_proba:
            y_pred = y_proba

        rs = numpy.random.RandomState(0)

        scores = []
        for k in range(n_bags):
            bagging = rs.randint(0, len(y_true), len(y_true))
            scores.append(metric.function(y_true[bagging], y_pred[bagging]))

        return numpy.percentile(scores, q=[low * 100, high * 100])
