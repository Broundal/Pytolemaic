import copy
import os

import numpy
import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from pytolemaic.utils.constants import FeatureTypes
from pytolemaic.utils.general import GeneralUtils

this_file = os.path.dirname(__file__)
adult_train_path = os.path.join(this_file, 'uci_adult', 'adult.data')
adult_test_path = os.path.join(this_file, 'uci_adult', 'adult.test')


class UCIAdult():
    def __init__(self):
        self._xtrain, self._ytrain = None, None
        self._xtest, self._ytest = None, None
        self.model = GeneralUtils.simple_imputation_pipeline(
            RandomForestClassifier(random_state=0, n_estimators=100, n_jobs=3)),

    def column_names(self):
        # 1st has no importance, while 3rd has double importance

        return ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

    def feature_types(self):
        num, cat = FeatureTypes.numerical, FeatureTypes.categorical
        types = [num, cat, num, cat, num, cat, cat, cat, cat, cat, num, num, num, cat]
        assert len(types) == len(self.column_names())
        return types

    def _encode_data(self, df, fit):
        df = copy.deepcopy(df)
        feature_types = self.feature_types()

        nan_mask = df.replace('?', numpy.nan).isnull()
        if fit:
            self.encoders = []
            for i, col in enumerate(self.column_names()):
                if feature_types[i] == FeatureTypes.categorical:
                    self.encoders.append(LabelEncoder().fit(df[col]))
                else:
                    self.encoders.append(None)

        for i, col in enumerate(self.column_names()):
            if feature_types[i] == FeatureTypes.categorical:
                df[col] = self.encoders[i].transform(df[col])

        df[nan_mask] = numpy.nan

        return df

    @property
    def training_data(self):
        if self._xtrain is None:
            df = pandas.read_csv(adult_train_path)
            ytrain = df[['target']]
            xtrain = self._encode_data(df.drop(['target'], axis=1), fit=True)
            self._xtrain, self._ytrain = xtrain, ytrain

        return self._xtrain, self._ytrain

    @property
    def testing_data(self):
        if self._xtest is None:
            df = pandas.read_csv(adult_test_path)
            ytest = df[['target']]
            xtest = self._encode_data(df.drop(['target'], axis=1), fit=False)
            self._xtest, self._ytest = xtest, ytest

        return self._xtest, self._ytest

    def get_model(self):
        x, y = self.training_data
        self.model.fit(x, y)
        return self.model


if __name__ == '__main__':
    adult = UCIAdult()
    df1, df2 = adult.training_data
    print(df1.describe())
    print(df2.describe())

    df1, df2 = adult.testing_data
    print(df1.describe())
    print(df2.describe())
