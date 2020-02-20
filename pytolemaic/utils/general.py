import copy

import numpy
import pandas
from matplotlib._color_data import XKCD_COLORS
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

    @classmethod
    def shuffled_colors(cls):
        l = list(XKCD_COLORS.values())
        rs = numpy.random.RandomState(0)
        return rs.permutation(l)

    @classmethod
    def make_dict_json_compatible(cls, dictionary: dict):
        
        def retype(obj):
            if isinstance(obj, (numpy.int_, numpy.intc, numpy.intp, numpy.int8,
                                numpy.int16, numpy.int32, numpy.int64, numpy.uint8,
                                numpy.uint16, numpy.uint32, numpy.uint64)):
                return int(obj)
            elif isinstance(obj, (numpy.float_, numpy.float16, numpy.float32,
                                  numpy.float64)):
                return float(obj)
            elif isinstance(obj, (numpy.ndarray,)): #### This is the fix
                return obj.tolist()
            return  obj

        dictionary = copy.deepcopy(dictionary)
        for k,v in dictionary.items():
            if isinstance(v, dict):
                dictionary[k] = cls.make_dict_json_compatible(v)
            else:
                dictionary[k] = retype(v)

        return dictionary

    @classmethod
    def make_dict_printable(cls, dictionary: dict):
        dictionary = cls.round_values(dictionary, digits=5)
        dictionary = cls.make_dict_json_compatible(dictionary)

        for k, v in dictionary.items():
            if isinstance(v, dict):
                dictionary[k] = cls.make_dict_printable(v)
                if len(dictionary[k]) > 20:
                    keys = list(dictionary[k].keys())[:20]
                    dictionary[k] = {key: dictionary[k][key] for key in keys}
                    dictionary[k]['...'] = '...'
            elif isinstance(v, (list, tuple)):
                if len(v) > 10:
                    dictionary[k] = [v[0], v[1], '...', v[-2], v[-1]]
            else:
                pass

        return dictionary

    @classmethod
    def nan_mask(cls, x):
        return pandas.DataFrame(x).isnull().values
