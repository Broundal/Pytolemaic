import itertools

import numpy

from pytolemaic.analysis_logic.dataset_analysis.dataset_analysis import DatasetAnalysis
from pytolemaic.analysis_logic.dataset_analysis.dataset_analysis_report import DatasetAnalysisReport
from pytolemaic.analysis_logic.model_analysis.scoring.scoring import \
    Scoring
from pytolemaic.analysis_logic.model_analysis.scoring.scoring_report import ScoringFullReport
from pytolemaic.analysis_logic.model_analysis.sensitivity.sensitivity import \
    SensitivityAnalysis
from pytolemaic.analysis_logic.prediction_analysis.lime_report import LimeExplainer
from pytolemaic.dataset_quality_report import TestSetQualityReport, TrainSetQualityReport, QualityReport, \
    ModelQualityReport
from pytolemaic.prediction_uncertainty.uncertainty_model import \
    UncertaintyModelClassifier, UncertaintyModelRegressor
from pytolemaic.utils.constants import CLASSIFICATION, REGRESSION
from pytolemaic.utils.dmd import DMD, ShuffleSplitter, StratifiedSplitter
from pytolemaic.utils.general import GeneralUtils
from pytolemaic.utils.metrics import Metrics, Metric


def cache(func):
    def cache_wrapper(self, *args, **kwargs):
        if not hasattr(self, '_cache'):
            self._cache = {}

        if func.__name__ not in self._cache:
            self._cache[func.__name__] = func(self, *args, **kwargs)

        return self._cache[func.__name__]

    return cache_wrapper


class PyTrust():

    def __init__(self, model,
                 xtrain=None, ytrain=None,
                 sample_meta_train=None,

                 xtest=None, ytest=None,
                 sample_meta_test=None,

                 columns_meta=None,
                 metric: [str, Metric] = None,
                 splitter='shuffled',
                 labels=None):
        self.model = model

        if isinstance(splitter, str):
            if splitter == 'shuffled':
                splitter = ShuffleSplitter
            elif splitter == 'stratified':
                splitter = StratifiedSplitter
            else:
                raise NotImplementedError("splitter='{}' is not supported".format(splitter))
        else:
            if not hasattr(splitter, 'split'):
                raise ValueError("splitter='{}' does not supported split() operation".format(splitter))
            else:
                raise NotImplementedError("splitter='{}' is not supported".format(splitter))


        self.train = xtrain
        if self.train is not None and not isinstance(self.train, DMD):
            self.train = DMD(x=xtrain, y=ytrain,
                             samples_meta=sample_meta_train,
                             columns_meta=columns_meta,
                             splitter=splitter,
                             labels=labels)

        self.test = xtest
        if self.test is not None and not isinstance(self.test, DMD):
            self.test = DMD(x=xtest, y=ytest,
                            samples_meta=sample_meta_test,
                            columns_meta=columns_meta,
                            splitter=splitter,
                            labels=labels)

        if metric is None:
            if GeneralUtils.is_classification(model):
                metric = Metrics.recall
            else:
                metric = Metrics.mae

        self.metric = metric.name if isinstance(metric, Metric) else metric

        # todo
        self._validate_input()

        self.sensitivity = SensitivityAnalysis()

        self._uncertainty_models = {}

        self._cache = {}

    def _validate_input(self):
        if not hasattr(self.model, 'predict'):
            raise ValueError("Model must support predict() function")

    @cache
    def sensitivity_report(self):
        self.sensitivity.calculate_sensitivity(
            model=self.model, dmd_test=self.test, metric=self.metric)

        return self.sensitivity.sensitivity_report()

    @property
    @cache
    def is_classification(self):
        return GeneralUtils.is_classification(self.model)

    @property
    @cache
    def model_support_dmd(self):
        return GeneralUtils.dmd_supported(self.model, self.test)

    @property
    @cache
    def y_pred_test(self):

        test = self.test if self.model_support_dmd else self.test.values
        if self.y_proba_test is not None:  # save some time
            y_pred_test = numpy.argmax(self.y_proba_test, axis=1)
        else:
            y_pred_test = self.model.predict(test)
        return y_pred_test

    @property
    @cache
    def y_proba_test(self):
        if not self.is_classification:
            return None

        test = self.test if self.model_support_dmd else self.test.values
        y_proba_test = self.model.predict_proba(test)
        return y_proba_test

    @cache
    def scoring_report(self) -> ScoringFullReport:
        metrics = Metrics.supported_metrics()

        self.scoring = Scoring(metrics=metrics)

        score_values_report, confusion_matrix, scatter, classification_report = \
            self.scoring.score_value_report(model=self.model,
                                            dmd_test=self.test,
                                            labels=self.test.labels,
                                            y_pred=self.y_pred_test,
                                            y_proba=self.y_proba_test)

        if self.train is not None and self.test is not None:
            separation_quality = self.scoring.separation_quality(dmd_train=self.train, dmd_test=self.test)
        else:
            separation_quality = numpy.nan

        return ScoringFullReport(target_metric=self.metric,
                                 metric_reports=score_values_report,
                                 separation_quality=separation_quality,
                                 confusion_matrix=confusion_matrix,
                                 scatter=scatter,
                                 classification_report=classification_report)

    @cache
    def quality_report(self) -> QualityReport:
        scoring_report = self.scoring_report()

        test_set_report = TestSetQualityReport(scoring_report=scoring_report)

        sensitivity_report = self.sensitivity_report()
        train_set_report = TrainSetQualityReport(vulnerability_report=sensitivity_report.vulnerability_report)
        model_quality_report = ModelQualityReport(vulnerability_report=sensitivity_report.vulnerability_report,
                                                  scoring_report=scoring_report)

        return QualityReport(train_quality_report=train_set_report, test_quality_report=test_set_report,
                             model_quality_report=model_quality_report)

    def create_uncertainty_model(self, method='default'):
        if method not in self._uncertainty_models:

            if self.is_classification:
                method = 'confidence' if method == 'default' else method
                uncertainty_model = UncertaintyModelClassifier(
                    model=self.model,
                    uncertainty_method=method)
            else:
                method = 'mae' if method == 'default' else method
                uncertainty_model = UncertaintyModelRegressor(
                    model=self.model,
                    uncertainty_method=method)

            uncertainty_model.fit(dmd_test=self.test)
            self._uncertainty_models[method] = uncertainty_model

        return self._uncertainty_models[method]

    @cache
    def create_lime_explainer(self, **kwargs):
        lime_explainer = LimeExplainer(n_features_to_plot=20, **kwargs)
        lime_explainer.fit(self.train, model=self.model)
        return lime_explainer

    @cache
    def dataset_analysis_report(self, **kwargs) -> DatasetAnalysisReport:
        da = DatasetAnalysis(problem_Type=CLASSIFICATION if self.is_classification else REGRESSION)
        report = da.dataset_analysis_report(dataset=self.train)
        return report

    @cache
    def insights_summary(self) -> list:
        return list(itertools.chain(self.dataset_analysis_report().insights_summary(),
                                    self.scoring_report().insights_summary(),
                                    self.sensitivity_report().insights_summary()))
