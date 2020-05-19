import copy
import logging

import numpy
import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from pytolemaic.utils.constants import FeatureTypes


class ShuffleSplitter():
    @classmethod
    def split(cls, dmdy, ratio=0.1, random_state=0):
        train, test = train_test_split(numpy.arange(dmdy.n_samples),
                                       test_size=ratio, random_state=random_state,
                                       shuffle=True, stratify=None)

        return train, test


class StratifiedSplitter():
    @classmethod
    def split(cls, dmdy, ratio=0.1, random_state=0):
        train, test = train_test_split(numpy.arange(dmdy.n_samples),
                                       test_size=ratio, random_state=random_state,
                                       shuffle=True, stratify=dmdy.target)

        if len(test) < ratio * 0.95 * dmdy.n_samples:
            logging.warning("Issue encountered with sklearn's stratified split! Reveting to shuffle split")
            train, test = ShuffleSplitter.split(dmdy=dmdy, ratio=ratio, random_state=random_state)

        return train, test



class DMD():
    FEATURE_NAMES = '__FEATURE_NAMES__'
    FEATURE_TYPES = '__FEATURE_TYPES__'
    INDEX = '__INDEX__'
    CATEGOICAL_ENCODING = '__CATEGOICAL_ENCODING__'

    # SAMPLE_WEIGHTS = '__SAMPLE_WEIGHTS__'

    def __init__(self, x, y=None, columns_meta: dict = None, samples_meta: dict = None,
                 splitter=ShuffleSplitter, target_labels: dict = None, categorical_encoding: dict = None,
                 feature_names: list = None, feature_types: list = None):

        if (feature_names or feature_types) and not columns_meta:
            columns_meta = {}

        if feature_names:
            columns_meta[self.FEATURE_NAMES] = feature_names
        if feature_types:
            columns_meta[self.FEATURE_TYPES] = feature_types

        self._x = pandas.DataFrame(x)
        if y is not None:
            self._y = pandas.DataFrame(y)
        else:
            self._y = None

        self._columns_meta = self._create_columns_meta(columns_meta, self._x)
        self._samples_meta = self._create_samples_meta(samples_meta, self._x)
        self._splitter = splitter

        # meta data
        if target_labels is not None:
            if self._y.values.max() >= len(target_labels):
                raise ValueError("Labels should be given to all classes")

        if isinstance(target_labels, list):
            target_labels = {i: cls_ for i, cls_ in enumerate(target_labels)}

        self._target_encoding = target_labels

        self._categorical_encoding_by_name = categorical_encoding or {}
        self._categorical_encoding_by_ind = self._validate_categorical_encoding()

    def _validate_categorical_encoding(self):
        categorical_encoding = self.categorical_encoding_by_feature_name

        if len(categorical_encoding) == 0:
            return {}

        categorical_features = self.categorical_features
        if categorical_features is None or len(categorical_features) == 0:
            raise ValueError(
                "When setting categorical_encoding you must also specify which feature is categorical through columns_meta")

        feature_names = self.feature_names
        values = self.values.astype(float, copy=False)
        for icol in categorical_features:
            if feature_names[icol] not in categorical_encoding:
                raise ValueError("No categorical names are set for feature #{}:{}".format(icol, feature_names[icol]))

            vec = values[:, icol]
            vec = vec[numpy.isfinite(vec)]
            delta = set(vec) - set(categorical_encoding[feature_names[icol]].keys())
            if delta:
                raise ValueError("No categorical name is set for categories {} (feature #{}:{})".format(delta, icol,
                                                                                                        feature_names[
                                                                                                            icol]))

        return {icol: self._categorical_encoding_by_name[feature_names[icol]] for icol in categorical_features}


    def __deepcopy__(self, memodict={}):
        return type(self)(x=copy.deepcopy(self._x),
                          y=copy.deepcopy(self._y),
                          columns_meta=copy.deepcopy(self._columns_meta),
                          samples_meta=copy.deepcopy(self._samples_meta),
                          splitter=copy.deepcopy(self._splitter),
                          )

    @classmethod
    def _create_columns_meta(cls, columns_meta, df):
        if columns_meta is None:
            columns_meta = pandas.DataFrame({DMD.FEATURE_NAMES: df.columns})
        else:
            columns_meta = pandas.DataFrame(columns_meta)

        if DMD.FEATURE_NAMES not in columns_meta:
            columns_meta[DMD.FEATURE_NAMES] = df.columns

        if df.shape[1] != columns_meta.shape[0]:
            raise ValueError("Given data has {} features but columns metadata was given for {} features"
                             .format(df.shape[1], columns_meta.shape[0]))
        return columns_meta

    @classmethod
    def _create_samples_meta(cls, samples_meta, df):
        if samples_meta is None:
            samples_meta = pandas.DataFrame(
                {DMD.INDEX: np.arange(df.shape[0])})
        else:
            samples_meta = pandas.DataFrame(samples_meta)

        if DMD.INDEX not in samples_meta:
            samples_meta[DMD.INDEX] = np.arange(df.shape[0])
        return samples_meta

    def split_by_indices(self, indices):
        return DMD(x=copy.deepcopy(self._x.iloc[indices, :]),
                   y=copy.deepcopy(self._y.iloc[indices, :]),
                   columns_meta=copy.deepcopy(self._columns_meta),
                   samples_meta=copy.deepcopy(
                       self._samples_meta.iloc[indices, :]),
                   splitter=self.splitter,
                   target_labels=self.target_encoding,
                   categorical_encoding=self.categorical_encoding_by_feature_name)

    def split(self, ratio, return_indices=False):

        dmd_y = DMD(x=self._y.values if self._y is not None else np.arange(
            self.n_samples), samples_meta=self._samples_meta)
        left, right = self.splitter.split(dmd_y, ratio=ratio)
        left = list(sorted(left))
        right = list(sorted(right))
        if return_indices:
            return left, right
        else:
            return self.split_by_indices(left), self.split_by_indices(right),

    def append(self, other, axis=0):
        if axis == 0:
            if other.feature_names != self.feature_names:
                raise ValueError("Cannot concat DMDs due to feature names difference")
            # noinspection PyTypeChecker
            if any(self._samples_meta.columns != other._samples_meta.columns):
                raise ValueError("Cannot concat DMDs due to samples meta difference")

            self._x = pandas.concat([self._x, other._x], axis=axis, ignore_index=True, copy=True)
            self._y = pandas.concat([self._y, other._y], axis=axis, ignore_index=True, copy=True)
            self._samples_meta = pandas.concat([self._samples_meta, other._samples_meta], axis=axis, ignore_index=True, copy=True)
            self._samples_meta[self.INDEX] = np.arange(self.n_samples)

        else:
            raise NotImplementedError("not implemented yet")

    def set_target(self, new_y):
        if new_y is None:
            self._y = None
        elif len(new_y) != self.n_samples:
            raise ValueError("Mismatch in number of samples")

        self._y = pandas.DataFrame(new_y)

    @classmethod
    def concat(cls, dmds, axis=0):
        dmd = None
        for dmd_item in dmds:
            if dmd is None:
                dmd = copy.deepcopy(dmd_item)
            else:
                dmd.append(dmd_item, axis=axis)

        return dmd

    @property
    def feature_names(self):
        return list(self._columns_meta[DMD.FEATURE_NAMES])

    @property
    def shape(self):
        return self._x.shape

    @property
    def n_samples(self):
        return self.shape[0]

    @property
    def n_features(self):
        return self.shape[1]

    @property
    def index(self):
        return self._samples_meta[DMD.INDEX]

    @property
    def values(self):
        return self._x.values

    @property
    def target(self):
        if self._y is None:
            return None
        else:
            return self._y.values

    @property
    def splitter(self):
        return self._splitter

    @property
    def target_encoding(self) -> dict:
        return self._target_encoding  # {index: cls_}

    @property
    def labels(self) -> list:
        """
        :return: class names ordered by encoding.
        """
        if self.target_encoding is None:
            return None

        encoding = self._target_encoding  # {index: cls_}
        inv_encoding = {cls_: index for index, cls_ in encoding.items()}
        labels = sorted(inv_encoding.keys(), key=lambda cls_: inv_encoding[cls_])
        return labels

    @property
    def categorical_features(self):
        if self.FEATURE_TYPES not in self._columns_meta.columns:
            return None
        else:
            return numpy.arange(self.n_features)[
                self._columns_meta[self.FEATURE_TYPES].values.ravel() == FeatureTypes.categorical]

    @property
    def categorical_encoding_by_feature_name(self):
        return self._categorical_encoding_by_name

    @property
    def categorical_encoding_by_icols(self):
        return self._categorical_encoding_by_ind

    @property
    def nan_mask(self):
        return self._x.isnull().values

    @classmethod
    def from_df(cls, df_train: pandas.DataFrame, target_name: str, is_classification, feature_types: list = None,
                df_test: pandas.DataFrame = None,
                categorical_encoding=False, split_ratio: float = None, nan_list: list = (), splitter=ShuffleSplitter):
        feature_names = list(df_train.columns)
        feature_names.remove(target_name)

        categorical_encoding_dict = None
        target_labels = None
        if categorical_encoding:
            if feature_types is None:
                raise ValueError("categorical_encoding requires feature types")
            logging.info("Categorical Encoding")
            # fit
            jibrish_value = '!@#$%^&*()'
            df_train = copy.deepcopy(df_train)
            for k in nan_list:
                df_train.replace(k, numpy.nan, inplace=True)

            nan_mask = df_train.isnull()
            # df_train[nan_mask] = jibrish_value
            df_train.replace(numpy.nan, jibrish_value, inplace=True)

            xencoders = []
            categorical_encoding_dict = {}
            for i, feature_name in enumerate(feature_names):
                logging.info("Fit encoding for feature {}".format(feature_name))

                if feature_types[i] == FeatureTypes.categorical:
                    le = LabelEncoder().fit(df_train[feature_name].astype(str))
                    xencoders.append(le)
                    categorical_encoding_dict[feature_name] = {i: cls_ for i, cls_ in enumerate(le.classes_)}
                else:
                    xencoders.append(None)

            if is_classification:
                yencoder = LabelEncoder().fit(df_train[target_name].astype(str))
                target_labels = {i: cls_ for i, cls_ in enumerate(yencoder.classes_)}
            else:
                yencoder = None
                target_labels = None

            # transform train
            for i, feature_name in enumerate(feature_names):
                if feature_types[i] == FeatureTypes.categorical:
                    logging.info("Encode train: feature {}".format(feature_name))
                    df_train[feature_name] = xencoders[i].transform(df_train[feature_name].astype(str))

            if yencoder:
                df_train[target_name] = yencoder.transform(df_train[target_name].astype(str))

            df_train[nan_mask] = numpy.nan

            if df_test is not None:
                df_test = copy.deepcopy(df_test)
                for k in nan_list:
                    df_test.replace(k, numpy.nan, inplace=True)

                nan_mask = df_test.isnull()
                df_test.replace(numpy.nan, jibrish_value, inplace=True)

                for i, feature_name in enumerate(feature_names):
                    if feature_types[i] == FeatureTypes.categorical:
                        logging.info("Encode test: feature {}".format(feature_name))

                        try:
                            df_test[feature_name] = xencoders[i].transform(df_test[feature_name].astype(str))
                        except ValueError as e:
                            logging.warning(e)
                            classes = xencoders[i].classes_
                            mask = numpy.array([x in classes for x in df_test[feature_name].astype(str)])

                            new_values = numpy.zeros(mask.shape)
                            new_values[mask] = xencoders[i].transform(df_test[feature_name].astype(str)[mask])
                            new_values[~mask] = -1
                            df_test[feature_name] = new_values

                if yencoder:
                    df_test[target_name] = yencoder.transform(df_test[target_name].astype(str))

                df_test[nan_mask] = numpy.nan

        logging.info("Creating DMDs")

        dmd_train = DMD(x=df_train[feature_names], y=df_train[target_name], splitter=splitter,
                        target_labels=target_labels, categorical_encoding=categorical_encoding_dict,
                        feature_names=feature_names, feature_types=feature_types)
        if df_test is None:
            if split_ratio is None:
                dmd_test = None
            else:
                dmd_train, dmd_test = dmd_train.split(ratio=split_ratio)
        else:

            dmd_test = DMD(x=df_test[feature_names], y=df_test[target_name], splitter=splitter,
                           target_labels=target_labels, categorical_encoding=categorical_encoding_dict,
                           feature_names=feature_names, feature_types=feature_types)

        return dmd_train, dmd_test
