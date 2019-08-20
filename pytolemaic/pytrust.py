from pytolemaic.analysis_logic.model_analysis.sensitivity.sensitivity import \
    SensitivityAnalysis
from pytolemaic.utils.dmd import DMD, ShuffleSplitter


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

    @classmethod
    def is_classification(cls, model):
        return hasattr(model, 'predict_proba')

    def _validate_input(self):
        if not hasattr(self.model, 'predict'):
            raise ValueError("Model must support predict() function")

    def sensitivity_report(self):
        self.sensitivity.calculate_sensitivity(
            model=self.model, dmd_test=self.test, metric=self.metric)

        return self.sensitivity.sensitivity_report()

    def scoring_report(self):

        pass
