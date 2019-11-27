import numpy
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


class GeneralUtils():

    @classmethod
    def is_classification(cls, model):
        return hasattr(model, 'predict_proba')

    @classmethod
    def dmd_supported(cls, model, dmd):
        try:
            model.predict(dmd.split_by_indices(indices=[0, 1, 2]))
            return True
        except:
            return False

    @classmethod
    def simple_imputation_pipeline(cls, estimator):
        estimators = []
        estimators.append(('Imputer', SimpleImputer()))
        estimators.append(('Estimator', estimator))
        return Pipeline(steps=estimators)

    @classmethod
    def round_values(cls, d: dict, digits=5):
        for k, v in d.items():
            if isinstance(v, dict):
                cls.round_values(v, digits=digits)
            else:
                try:
                    d[k] = numpy.round(v, digits)
                except:
                    pass

        return d

    @classmethod
    def f5(cls, x):
        return numpy.round(x, 5)

    @classmethod
    def f3(cls, x):
        return numpy.round(x, 3)

    @classmethod
    def add_nans(cls, x, ratio=0.1):
        rs = numpy.random.RandomState(0)
        # let's add some missing values
        nan_locs = numpy.ones(numpy.prod(x.shape))
        nan_locs[rs.permutation(len(nan_locs))[:int(ratio * len(nan_locs))]] = numpy.nan
        nan_locs = nan_locs.reshape(x.shape)
        x = x * nan_locs
        return x
