import numpy
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from pytolemaic.utils.constants import REGRESSION, CLASSIFICATION
from pytolemaic.utils.dmd import DMD
from pytolemaic.utils.general import GeneralUtils


class UncertaintyModelBase():

    def __init__(self, model, uncertainty_method: str,
                 ptype=None,
                 supported_methods: list = None):
        self.model = model
        self.uncertainty_method = uncertainty_method
        self.dmd_supported = None
        self.is_classification = GeneralUtils.is_classification(model)

        if (self.is_classification and ptype != CLASSIFICATION) or \
                (not self.is_classification and ptype == CLASSIFICATION):
            raise ValueError(
                "{} does not support {}".format(type(self), ptype))

        if self.uncertainty_method not in supported_methods:
            raise NotImplementedError(
                "Uncertainty method {} is not in supported methods={}".format(
                    self.uncertainty_method, supported_methods))

    def uncertainty(self, dmd: DMD):
        raise NotImplementedError("")

    def fit_uncertainty_model(self, dmd_test, **kwargs):
        raise NotImplementedError("")

    def fit(self, dmd_test: DMD, **kwargs):
        self.dmd_supported = GeneralUtils.dmd_supported(model=self.model,
                                                        dmd=dmd_test)

        self.fit_uncertainty_model(dmd_test, **kwargs)
        return self

    def predict(self, dmd: DMD):
        if not self.dmd_supported:
            x = dmd.values
            return self.model.predict(x)
        else:
            return self.model.predict(dmd)

    def predict_proba(self, dmd: DMD):
        if not self.dmd_supported:
            x = dmd.values
            return self.model.predict_proba(x)
        else:
            return self.model.predict_proba(dmd)


class UncertaintyModelRegressor(UncertaintyModelBase):

    def __init__(self, model, uncertainty_method='mae'):
        super(UncertaintyModelRegressor, self).__init__(model=model,
                                                        uncertainty_method=uncertainty_method,
                                                        ptype=REGRESSION,
                                                        supported_methods=[
                                                            'mae'])

    def fit_uncertainty_model(self, dmd_test, n_jobs=-1, **kwargs):

        if self.uncertainty_method in ['mae']:
            estimator = RandomForestRegressor(
                random_state=0, n_jobs=n_jobs, n_estimators=100)

            self.uncertainty_model = GeneralUtils.simple_imputation_pipeline(
                estimator)

            yp = self.predict(dmd_test)
            self.uncertainty_model.fit(dmd_test.values,
                                       numpy.abs(
                                           dmd_test.target.ravel() - yp.ravel()))
        else:
            raise NotImplementedError("Method {} is not implemented"
                                      .format(self.uncertainty_method))

    def uncertainty(self, dmd: DMD):
        if self.uncertainty_method in ['mae']:
            out = self.uncertainty_model.predict(dmd.values)
            return out.reshape(-1, 1)
        else:
            raise NotImplementedError("Method {} is not implemented"
                                      .format(self.uncertainty_method))


class UncertaintyModelClassifier(UncertaintyModelBase):

    def __init__(self, model, uncertainty_method='confidence'):
        super(UncertaintyModelClassifier, self).__init__(model=model,
                                                         uncertainty_method=uncertainty_method,
                                                         ptype=CLASSIFICATION,
                                                         supported_methods=[
                                                             'probability',
                                                             'confidence']
                                                         )

    @classmethod
    def _get_confidence_estimator(cls, n_jobs=-1):

        estimator = RandomForestClassifier(
            random_state=0, n_jobs=n_jobs, n_estimators=100)

        return GeneralUtils.simple_imputation_pipeline(estimator)

    @classmethod
    def _get_probability_estimator(cls, n_jobs=-1):

        estimator = RandomForestClassifier(
            random_state=0, n_jobs=n_jobs, n_estimators=100)

        return GeneralUtils.simple_imputation_pipeline(estimator)

    def fit_uncertainty_model(self, dmd_test, estimator=None, n_jobs=-1,
                              **kwargs):

        if self.uncertainty_method in ['probability']:
            pass  # no fit logic required
        elif self.uncertainty_method in ['confidence']:
            estimator = RandomForestClassifier(
                random_state=0, n_jobs=n_jobs, n_estimators=5)

            self.uncertainty_model = GeneralUtils.simple_imputation_pipeline(
                estimator)

            yp = self.predict(dmd_test)
            is_correct = numpy.array(yp.ravel() == dmd_test.target.ravel(),
                                     dtype=int)

            # bug here
            self.uncertainty_model.fit(dmd_test.values, is_correct)

        else:
            raise NotImplementedError("Method {} is not implemented"
                                      .format(self.uncertainty_method))

    def uncertainty(self, dmd: DMD):
        if self.uncertainty_method in ['probability']:
            yproba = self.predict_proba(dmd)
            yproba += 1e-10 * numpy.random.rand(*yproba.shape)
            max_probability = numpy.max(yproba, axis=1).reshape(-1, 1)
            delta = max_probability - yproba
            yproba[delta == 0] = 0

            # delta[numpy.sum(delta, axis=1)>=20,:] = 0 # i
            out = numpy.max(yproba, axis=1).reshape(-1, 1) / max_probability
            return out.reshape(-1, 1)
        elif self.uncertainty_method in ['confidence']:
            # return the probability it's a mistake
            out = self.uncertainty_model.predict_proba(dmd.values)[:, 0]
            return out.reshape(-1, 1)
        else:
            raise NotImplementedError("Method {} is not implemented"
                                      .format(self.uncertainty_method))
