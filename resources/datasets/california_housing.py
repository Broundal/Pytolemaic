import numpy
import sklearn
import sklearn.datasets
from sklearn.ensemble import RandomForestRegressor

from pytolemaic.utils.constants import FeatureTypes
from pytolemaic.utils.dmd import DMD


class CaliforniaHousing():
    def __init__(self):

        self._d = sklearn.datasets.fetch_california_housing(return_X_y=False)
        self._xtrain, self._xtest, self._ytrain, self._ytest = None, None, None, None
        self.model = RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=3)

    def column_names(self):
        return self._d['feature_names']

    #     ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

    def feature_types(self):
        num, cat = FeatureTypes.numerical, FeatureTypes.categorical
        types = len(self.column_names()) * [num]
        return types

    def _split_train_test(self):
        train_inds, test_inds = sklearn.model_selection.train_test_split(numpy.arange(self._d['data'].shape[0]),
                                                                         test_size=0.3, random_state=0)
        x = self._d['data']
        y = self._d['target']
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
