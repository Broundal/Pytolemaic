import copy

import numpy as np
import pandas


class ShuffleSplitter():
    @classmethod
    def split(cls, dmdy, ratio=0.1, random_state=0):
        n_right = int(np.round(dmdy.n_samples * ratio, 0))
        rs = np.random.RandomState(random_state)
        shuffled = rs.permutation(dmdy.n_samples)
        return shuffled[:-n_right], shuffled[-n_right:]


class DMD():
    FEATURE_NAMES = '__FEATURE_NAMES__'
    FEATURE_TYPES = '__FEATURE_TYPES__'
    INDEX = '__INDEX__'

    # SAMPLE_WEIGHTS = '__SAMPLE_WEIGHTS__'

    def __init__(self, x, y=None, columns_meta=None, samples_meta=None,
                 splitter=ShuffleSplitter, labels=None):

        self._x = pandas.DataFrame(x)
        if y is not None:
            self._y = pandas.DataFrame(y)
        else:
            self._y = None

        self._columns_meta = self._create_columns_meta(columns_meta, self._x)
        self._samples_meta = self._create_samples_meta(samples_meta, self._x)
        self._splitter = splitter

        # meta data
        if labels is not None:
            if self._y.values.max() >= len(labels):
                raise ValueError("Labels should be given to all classes")

        self._labels = labels


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
                   splitter=self.splitter)

    def split(self, ratio):

        dmd_y = DMD(x=self._y.values if self._y is not None else np.arange(
            self.n_samples), samples_meta=self._samples_meta)
        left, right = self.splitter.split(dmd_y, ratio=ratio)
        left = list(sorted(left))
        right = list(sorted(right))

        return self.split_by_indices(left), self.split_by_indices(right),

    def append(self, other, axis=0):
        if axis == 0:
            if other.feature_names != self.feature_names:
                raise ValueError("Cannot concat DMDs due to feature names difference")
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
            return self._y.values.reshape(-1, 1)

    @property
    def splitter(self):
        return self._splitter

    @property
    def labels(self):
        return self._labels
