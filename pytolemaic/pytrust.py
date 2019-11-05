from pytolemaic.analysis_logic.model_analysis.scoring.scoring import \
    ScoringReport
from pytolemaic.analysis_logic.model_analysis.sensitivity.sensitivity import \
    SensitivityAnalysis
from pytolemaic.analysis_logic.prediction_analysis.prediction_uncertainty.uncertainty_model import \
    UncertaintyModelClassifier, UncertaintyModelRegressor
from pytolemaic.utils.dmd import DMD, ShuffleSplitter
from pytolemaic.utils.general import GeneralUtils
from pytolemaic.utils.metrics import Metrics, Metric


class SklearnTrustBase():

    def __init__(self, model,
                 xtrain=None, ytrain=None,
                 sample_meta_train=None,

                 xtest=None, ytest=None,
                 sample_meta_test=None,

                 columns_meta=None,
                 metric: [str, Metric] = None,
                 splitter='shuffled'):
        self.model = model

        if splitter == 'shuffled':
            splitter = ShuffleSplitter  # todo support stratified
        else:
            raise NotImplementedError

        if xtrain is not None:
            if isinstance(xtrain, DMD):
                self.train = xtrain
            else:
                self.train = DMD(x=xtrain, y=ytrain,
                                 samples_meta=sample_meta_train,
                                 columns_meta=columns_meta,
                                 splitter=splitter)

        if xtest is not None:
            if isinstance(xtest, DMD):
                self.test = xtest
            else:
                self.test = DMD(x=xtest, y=ytest,
                                samples_meta=sample_meta_test,
                                columns_meta=columns_meta,
                                splitter=splitter)

        self.metric = metric.name if isinstance(metric, Metric) else metric

        # todo
        self._validate_input()

        self.sensitivity = SensitivityAnalysis()

        self._is_classification = None
        self._model_support_dmd = None
        self._y_pred_test = None
        self._y_proba_test = None
        self._uncertainty_models = {}


    def _validate_input(self):
        if not hasattr(self.model, 'predict'):
            raise ValueError("Model must support predict() function")

    def sensitivity_report(self):
        self.sensitivity.calculate_sensitivity(
            model=self.model, dmd_test=self.test, metric=self.metric)

        return self.sensitivity.sensitivity_report()

    @property
    def is_classification(self):
        if self._is_classification is None:
            self._is_classification = GeneralUtils.is_classification(
                self.model)

        return self._is_classification

    @property
    def model_support_dmd(self):
        if self._model_support_dmd is None:
            self._model_support_dmd = GeneralUtils.dmd_supported(self.model,
                                                                 self.test)

        return self._model_support_dmd

    @property
    def y_pred_test(self):
        if self._y_pred_test is None:
            test = self.test if self.model_support_dmd else self.test.values
            self._y_pred_test = self.model.predict(test)

        return self._y_pred_test

    @property
    def y_proba_test(self):
        if self._y_proba_test is None and self.is_classification:
            test = self.test if self.model_support_dmd else self.test.values
            self._y_proba_test = self.model.predict_proba(test)

        return self._y_proba_test


    def scoring_report(self):
        metrics = Metrics.supported_metrics()

        self.scoring = ScoringReport(metrics=metrics)

        score_values = self.scoring.score_report(model=self.model, dmd_test=self.test)
        score_quality = self.scoring.score_quality(dmd_train=self.train, dmd_test=self.test)
        return {'Score': score_values,
                'Quality':score_quality}

    def create_uncertainty_model(self, method='auto'):
        if method not in self._uncertainty_models:

            if self.is_classification:
                method = 'confidence' if method == 'auto' else method
                uncertainty_model = UncertaintyModelClassifier(
                    model=self.model,
                    uncertainty_method=method)
            else:
                method = 'rmse' if method == 'auto' else method
                uncertainty_model = UncertaintyModelRegressor(
                    model=self.model,
                    uncertainty_method=method)

            uncertainty_model.fit(dmd_test=self.test)
            self._uncertainty_models[method] = uncertainty_model

        return self._uncertainty_models[method]



