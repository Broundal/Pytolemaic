import copy

import numpy
from pytolemaic import FeatureTypes
from sklearn.preprocessing import LabelEncoder
from pytolemaic.utils.general import get_logger
import pandas as pd
logger = get_logger(__name__)

class LabelEncoderProtected():
    def __init__(self, nan_list=()):
        self.encoders = []
        self.categorical_encoding_dict = {}
        self.nan_list = nan_list
        self.jibrish_value = '!@#$%^&*()'

    def _replace_nan_with_jibbrish(self, x):

        for k in self.nan_list:
            x[x==k] = numpy.nan

        nan_mask = pd.DataFrame(x).isnull().values

        x[nan_mask] = self.jibrish_value

        return x, nan_mask

    def _replace_jibbrish_with_nan(self, x):
        x[x==self.jibrish_value] = numpy.nan
        return x


    def fit(self, x, feature_types, feature_names=None):
        self.encoders = []
        self.categorical_encoding_dict = {}

        if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
            x = x.values

        if len(x.shape) == 1:
            x = x.reshape(-1,1)

        if feature_types is None:
            raise ValueError("Couldn't encode dmd w/o feature_types")

        if isinstance(feature_types, str):
            feature_types = [feature_types]
        if len(feature_types) != x.shape[1]:
            raise ValueError("Inconsistent # of features and # of feature_types")
        if feature_names is None:
            feature_names = numpy.arange(x.shape[1])

        self._feature_names = feature_names
        self._feature_types = feature_types

        x = copy.deepcopy(x).astype(object)
        x, nan_mask = self._replace_nan_with_jibbrish(x)

        for i, (name, feature_type) in enumerate(zip(self._feature_names, self._feature_types)):
            logger.info("Fit encoding for feature #{}:'{}'. Feature type is '{}'"
                         .format(i, name, feature_type))

            if feature_type == FeatureTypes.categorical:
                le = LabelEncoder().fit(x[:,i].astype(str))
                self.encoders.append(le)
                self.categorical_encoding_dict[name] = {i: cls_ for i, cls_ in enumerate(le.classes_)}
            else:
                self.encoders.append(None)

        return self

    def _encode_protected(self, le, series):
        try:
            return le.transform(series), None
        except ValueError as e:
            classes = le.classes_
            mask = numpy.array([x in classes for x in series])

            new_values = numpy.zeros(mask.shape)
            new_values[mask] = le.transform(series[mask])
            new_values[~mask] = -1  # never seen these values before - new class
            return new_values, e

    def transform(self, x):
        if not self.encoders:
            raise ValueError("Call fit() before transform()")

        if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
            x = x.values

        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        x = copy.deepcopy(x).astype(object)
        x, nan_mask = self._replace_nan_with_jibbrish(x)

        for i, (name, feature_type) in enumerate(zip(self._feature_names, self._feature_types)):
            if feature_type == FeatureTypes.categorical:
                logger.info("Transform feature #{}:'{}'".format(i, name))

                x[:,i], e = self._encode_protected(le=self.encoders[i],
                                                   series=x[:,i].astype(str))
                if e is not None:
                    logger.warning(e)

        x[nan_mask] = numpy.nan

        return x.astype(float)

if __name__ == '__main__':

    dmd = pd.DataFrame([list('fff'),
                 list('abc'),
                 list('ddd'),
                [numpy.nan]*3])
    lew = LabelEncoderProtected().fit(dmd, feature_types=[FeatureTypes.categorical] * 3)
    dmd_out = lew.transform(dmd)

    print(dmd_out.values)

    dmd = pd.DataFrame([list('fff'),
                        list('abc'),
                        list('hhh'),
                        [numpy.nan] * 3])
    dmd_out = lew.transform(dmd)



    print(dmd_out.values)

    dmd = pd.DataFrame([list('fff'),
                 list('abc'),
                 list('ddd'),
                [numpy.nan]*3])[0]
    lew = LabelEncoderProtected().fit(dmd, feature_types=[FeatureTypes.categorical] * 3)
    dmd_out = lew.transform(dmd)

    print(dmd_out.values)
