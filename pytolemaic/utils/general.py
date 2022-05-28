import copy
import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler

from sklearn.base import RegressorMixin

def get_logger(name, level="DEBUG", handlers=(), log_path='default'):
    split = name.split('.')
    if len(split)>2 and split[0]=='pytolemaic':
        name = split[0] + '.' + split[-1]
    logger = logging.getLogger(name)
    logger.setLevel(level=os.environ.get("LOGLEVEL", level))

    if handlers is not None:
        if len(handlers)==0:
            stream_handler = logging.StreamHandler(sys.stdout)
            fmt = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s]: %(message)s",
                                    datefmt='%Y-%m-%d %H:%M:%S')
            stream_handler.setFormatter(fmt)
            stream_handler.setLevel(max(logger.level, logging.INFO))

            if log_path == 'default':
                abs_path = os.path.abspath(__file__)
                log_path = os.path.join(abs_path, '../../../logs/log.log')
            file_handler = TimedRotatingFileHandler(log_path, when='midnight')
            file_handler.setFormatter(fmt)
            file_handler.setLevel(logging.DEBUG)

            handlers = [stream_handler, file_handler]

        for h in handlers:
            logger.addHandler(h)

    return logger

logger = get_logger(__name__)

matplotlib_logger = logging.getLogger("matplotlib")
matplotlib_logger.setLevel(os.environ.get("LOGLEVEL", "WARNING"))




import numpy
import pandas
from matplotlib._color_data import XKCD_COLORS
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

global tictoc
from time import time


def tic(name=None):
    global tictoc

    tictoc = time()
    if name is not None:
        logger.info("Calculating {}...".format(name))


def toc(name=None):
    global tictoc
    if name is not None:
        logger.info("Calculating {}... Done {:.2g} seconds".format(name, time() - tictoc))
    return time() - tictoc


class GeneralUtils():

    @classmethod
    def is_classification(cls, model):
        """
        Use the existence of predict_proba method to differentiate between classifier and regressor
        In order to use a regressor with a predict_proba method introduce an _estimator_type property with value "regressor"
        """
        is_regression = getattr(model, "_estimator_type", None) in ["regressor", RegressorMixin._estimator_type]
        return hasattr(model, 'predict_proba') and callable(model.predict_proba) and not is_regression

    @classmethod
    def dmd_supported(cls, model, dmd):
        try:
            model.predict(dmd.split_by_indices(indices=[0, 1, 2]))
            return True
        except:
            return False

    @classmethod
    def simple_imputation_pipeline(cls, estimator):
        # note - this works only for enncoded data (e.g. no text in caategoricl features)
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
        for k, v in dictionary.items():
            if isinstance(v, dict):
                dictionary[k] = cls.make_dict_json_compatible(v)
            else:
                dictionary[k] = retype(v)

        return dictionary

    @classmethod
    def shorten_long_dict(cls, dictionary: dict, dict_limit=20, list_limit=10):
        if len(dictionary) > dict_limit:
            keys = list(dictionary.keys())[:dict_limit]
            dictionary = {key: dictionary[key] for key in keys}
            dictionary['...'] = '...'

        for k, v in dictionary.items():
            if isinstance(v, dict):
                dictionary[k] = cls.shorten_long_dict(v)
            elif isinstance(v, (list, tuple)):
                if len(v) > list_limit:
                    dictionary[k] = [v[0], v[1], '...', v[-2], v[-1]]
            else:
                pass

        return dictionary

    @classmethod
    def make_dict_printable(cls, dictionary: dict):

        dictionary = cls.shorten_long_dict(dictionary)
        dictionary = cls.round_values(dictionary, digits=5)
        dictionary = cls.make_dict_json_compatible(dictionary)

        return dictionary

    @classmethod
    def nan_mask(cls, x):
        return pandas.DataFrame(x).isnull().values
