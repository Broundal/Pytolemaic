import multiprocessing

import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.multiclass import unique_labels

from pytolemaic.analysis_logic.model_analysis.scoring.scoring_report import ScoringMetricReport, ConfusionMatrixReport, \
    ScatterReport, SklearnClassificationReport
from pytolemaic.prediction_uncertainty.uncertainty_model import UncertaintyModelRegressor
from pytolemaic.utils.constants import CLASSIFICATION, REGRESSION
from pytolemaic.utils.dmd import DMD
from pytolemaic.utils.general import GeneralUtils
from pytolemaic.utils.metrics import Metrics


class Scoring():
    def __init__(self, metrics: list = None):
        self.supported_metric = Metrics.supported_metrics()
        self.metrics = metrics or self.supported_metric
        self.metrics = [self.supported_metric[metric] for metric in
                        self.metrics
                        if metric in self.supported_metric]

    def score_value_report(self, model, dmd_test: DMD,
                           labels=None,
                           y_proba: numpy.ndarray = None,
                           y_pred: numpy.ndarray = None) -> [ScoringMetricReport]:
        '''

        :param model: model of interest
        :param dmd_test: test set
        :param y_proba: pre-calculated predicted probabilities for test set, if available
        :param y_pred: pre-calculated models' predictions for test set, if available
        :return: scoring report
        '''
        score_report = []

        model_support_dmd = GeneralUtils.dmd_supported(model, dmd_test)
        x_test = dmd_test if model_support_dmd else dmd_test.values
        y_true = dmd_test.target

        is_classification = GeneralUtils.is_classification(model)

        confusion_matrix, scatter, classification_report = None, None, None
        if is_classification:

            y_proba = y_proba if y_proba is not None else model.predict_proba(x_test)
            y_pred = y_pred if y_pred is not None else numpy.argmax(y_proba, axis=1)

            confusion_matrix = ConfusionMatrixReport(y_true=y_true, y_pred=y_pred,
                                                     labels=labels if labels is not None else unique_labels(y_true,
                                                                                                            y_pred))

            classification_report = SklearnClassificationReport(y_true=y_true, y_pred=y_pred, y_proba=y_proba, labels=labels)

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
                score_report.append(ScoringMetricReport(
                    metric=metric.name,
                    value=score,
                    ci_low=ci_low,
                    ci_high=ci_high))


        else:
            y_pred = y_pred if y_pred is not None else model.predict(x_test)

            error_bars = self._calc_error_bars(dmd_test, model)

            scatter = ScatterReport(y_true=y_true, y_pred=y_pred, error_bars=error_bars)
            for metric in self.metrics:
                if not metric.ptype == REGRESSION:
                    continue

                score = metric.function(y_true, y_pred)
                ci_low, ci_high = Metrics.confidence_interval(metric,
                                                              y_true=y_true,
                                                              y_pred=y_pred)

                ci_low = GeneralUtils.f5(ci_low)
                ci_high = GeneralUtils.f5(ci_high)
                score = GeneralUtils.f5(score)
                score_report.append(ScoringMetricReport(
                    metric=metric.name,
                    value=score,
                    ci_low=ci_low,
                    ci_high=ci_high))

        return score_report, confusion_matrix, scatter, classification_report

    def _calc_error_bars(self, dmd_test, model):
        left, right = dmd_test.split(ratio=0.5, return_indices=True)
        error_bars = numpy.zeros(dmd_test.n_samples)
        for ind1, ind2 in [(left, right), (right, left)]:
            dmd_1 = dmd_test.split_by_indices(ind1)
            dmd_2 = dmd_test.split_by_indices(ind2)
            uncertainty_model = UncertaintyModelRegressor(model=model, uncertainty_method='mae')
            uncertainty_model.fit(dmd_test=dmd_1)
            error_bars[ind2] = uncertainty_model.uncertainty(dmd_2).ravel()
        return error_bars

    def _prepare_dataset_for_score_quality(self, dmd_train: DMD,
                                           dmd_test: DMD):
        '''

        :param dmd_train: train set
        :param dmd_test: test set
        :return: dataset with target of test/train
        '''

        dmd = DMD.concat([dmd_train, dmd_test])
        new_label = [0] * dmd_train.n_samples + [1] * dmd_test.n_samples
        dmd.set_target(new_label)

        train, test = dmd.split(ratio=dmd_test.n_samples / (dmd_train.n_samples + dmd_test.n_samples))
        return train, test

    def separation_quality(self, dmd_train: DMD, dmd_test: DMD):
        '''

        :param dmd_train: train set
        :param dmd_test: test set
        :return: estimation of score quality based on similarity between train and test sets
        '''

        train, test = self._prepare_dataset_for_score_quality(dmd_train=dmd_train,
                                                              dmd_test=dmd_test)

        classifier = GeneralUtils.simple_imputation_pipeline(
            estimator=RandomForestClassifier(n_estimators=100, n_jobs=multiprocessing.cpu_count() - 1))
        classifier.fit(train.values, train.target.ravel())

        yp = classifier.predict_proba(test.values)

        auc = Metrics.auc.function(y_true=test.target, y_pred=yp)
        auc = numpy.clip(auc, 0.5, 1)  # auc<0.5 --> 0.5

        return numpy.round(1 - (2 * auc - 1), 5)