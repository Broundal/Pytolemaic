from pytolemaic.analysis_logic.model_analysis.scoring.scoring import \
    ScoringReport
from pytolemaic.analysis_logic.model_analysis.sensitivity.sensitivity import \
    SensitivityAnalysis
from pytolemaic.utils.dmd import DMD, ShuffleSplitter
from pytolemaic.utils.general import GeneralUtils
from pytolemaic.utils.metrics import Metrics


class SklearnTrustBase():

    def __init__(self, model,
                 Xtrain=None, Ytrain=None,
                 sample_meta_train=None,

                 Xtest=None, Ytest=None,
                 sample_meta_test=None,

                 columns_meta=None,
                 metric: str = None):
        self.model = model

        splitter = ShuffleSplitter  # todo support stratified

        if Xtrain is not None:
            if isinstance(Xtrain, DMD):
                self.train = Xtrain
            else:
                self.train = DMD(x=Xtrain, y=Ytrain,
                                 samples_meta=sample_meta_train,
                                 columns_meta=columns_meta,
                                 splitter=splitter)

        if Xtest is not None:
            if isinstance(Xtest, DMD):
                self.test = Xtest
            else:
                self.test = DMD(x=Xtest, y=Ytest,
                                samples_meta=sample_meta_test,
                                columns_meta=columns_meta,
                                splitter=splitter)

        self.metric = metric

        # todo
        self._validate_input()

        self.sensitivity = SensitivityAnalysis()

        self._is_classification = None
        self._model_support_dmd = None
        self._y_pred_test = None
        self._y_proba_test = None


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

        return self.scoring.score_report(model=self.model, dmd_test=self.test)
