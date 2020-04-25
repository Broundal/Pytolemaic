import copy
import logging

import numpy
import pandas
import sklearn
import sklearn.datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from pytolemaic.utils.constants import FeatureTypes
from pytolemaic.utils.dmd import DMD
from pytolemaic.utils.general import GeneralUtils


class KDDCup99():
    def __init__(self, subset=False):

        self._d = sklearn.datasets.fetch_kddcup99(return_X_y=False, percent10=True)

        if subset:
            n = 10000
            inds = numpy.random.permutation(len(self._d['target']))[:n]
            self._d['data'] = self._d['data'][inds, :]
            self._d['target'] = self._d['target'][inds]

        self._xtrain, self._xtest, self._ytrain, self._ytest = None, None, None, None
        self.model = GeneralUtils.simple_imputation_pipeline(
            RandomForestClassifier(random_state=0, n_estimators=10, n_jobs=3))

        self.other_label_size = 1000

    @property
    def labels(self):
        return ['normal', 'abnormal']

    def column_names(self):
        feature_names = [
            'duration',
            'protocol_type',
            'service',
            'flag',
            'src_bytes',
            'dst_bytes',
            'land',
            'wrong_fragment',
            'urgent',
            'hot',
            'num_failed_logins',
            'logged_in',
            'num_compromised',
            'root_shell',
            'su_attempted',
            'num_root',
            'num_file_creations',
            'num_shells',
            'num_access_files',
            'num_outbound_cmds',
            'is_host_login',
            'is_guest_login',
            'count',
            'srv_count',
            'serror_rate',
            'srv_serror_rate',
            'rerror_rate',
            'srv_rerror_rate',
            'same_srv_rate',
            'diff_srv_rate',
            'srv_diff_host_rate',
            'dst_host_count',
            'dst_host_srv_count',
            'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate',
            'dst_host_srv_serror_rate',
            'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate',
        ]
        return feature_names

    #     ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

    def feature_types(self):
        n_features = len(self.column_names())
        num, cat = FeatureTypes.numerical, FeatureTypes.categorical
        types = n_features * [num]

        for k in range(n_features):
            if k in [1, 2, 3]:
                types[k] = cat
            if len(set(self._d['data'][:, k])) <= 3:
                types[k] = cat

        return types

    def _encode_data(self, df, fit):
        df = copy.deepcopy(df)
        feature_types = self.feature_types()

        nan_mask = df.replace('?', numpy.nan).isnull()
        df[nan_mask] = 0
        if fit:
            self.xencoders = []
            for i, col in enumerate(self.column_names()):
                if feature_types[i] == FeatureTypes.categorical:
                    print('fit', i, col)
                    self.xencoders.append(LabelEncoder().fit(list(set(df[col]))))
                else:
                    self.xencoders.append(None)

            self.yencoder = LabelEncoder().fit(list(set(df['target'])))
            self._labels = self.yencoder.classes_

        for i, col in enumerate(self.column_names()):
            if feature_types[i] == FeatureTypes.categorical:
                print('transform', i, col)
                df[col] = self.xencoders[i].transform(df[col])

        df['target'] = self.yencoder.transform(df['target'])

        df[nan_mask] = numpy.nan

        return df

    def _split_train_test(self):
        x = self._d['data']
        y = self._d['target']
        y = numpy.array([k.decode('utf-8') for k in y])

        threshold = self.other_label_size
        other_label = 'other'
        labels, counts = numpy.unique(y, return_counts=True)

        for label, count in zip(labels, counts):
            if count < threshold:
                y[y == label] = other_label

            if label.endswith('.'):
                y[y == label] = label[:-1]

        # 2 classes:
        # y[~(y == 'normal.')] = 'abnormal'
        # y[y == 'normal.'] = 'normal'

        for i in [1, 2, 3]:
            x[:, i] = [k.decode('utf-8') for k in x[:, i]]

        train_inds, test_inds = sklearn.model_selection.train_test_split(numpy.arange(len(y)),
                                                                         test_size=0.3, random_state=0,
                                                                         stratify=y)

        if len(test_inds) < 0.29 * len(y):
            logging.warning("Issue with sklearn's stratified split. reveting to shuffle split")
            train_inds, test_inds = sklearn.model_selection.train_test_split(numpy.arange(len(y)),
                                                                             test_size=0.3, random_state=0)

        df = pandas.DataFrame(x, columns=self.column_names())
        df['target'] = y
        df = self._encode_data(df, fit=True)
        x = df[self.column_names()].values
        y = df['target'].values

        self._xtrain = x[train_inds]
        self._ytrain = y[train_inds]

        self._xtest = x[test_inds]
        self._ytest = y[test_inds]

    @property
    def training_data(self):
        if self._xtrain is None:
            self._split_train_test()

        return self._xtrain, self._ytrain

    @property
    def testing_data(self):
        if self._xtest is None:
            self._split_train_test()

        return self._xtest, self._ytest

    def get_model(self):
        x, y = self.training_data
        print("fitting model")
        self.model.fit(x, y)
        return self.model

    def as_dmd(self):
        train = DMD(x=self.training_data[0], y=self.training_data[1],
                    samples_meta=None, columns_meta={DMD.FEATURE_NAMES: self.column_names(),
                                                     DMD.FEATURE_TYPES: self.feature_types()})

        test = DMD(x=self.testing_data[0], y=self.testing_data[1],
                   samples_meta=None, columns_meta={DMD.FEATURE_NAMES: self.column_names(),
                                                    DMD.FEATURE_TYPES: self.feature_types()})
        return train, test
