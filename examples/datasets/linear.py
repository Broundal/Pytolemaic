import numpy
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from pytolemaic.utils.general import GeneralUtils


class LinearBase():
    def __init__(self, model, n_samples=500, n_features=5):
        self.n_samples = n_samples
        self.n_features = n_features

        self._xtrain, self._ytrain = None, None
        self.model = model
        self.random_state = numpy.random.RandomState(0)
        self.y_type = float


    def _function(self, x):
        y = 0 * x[:, 0] + 3 * x[:, 1] + numpy.sum(x[:, 2:], axis=1)
        if self.y_type == float:
            return y
        else:
            return y.astype(int)

    def column_names(self):
        # 1st has no importance, while 3rd has double importance

        return ['zero importance', 'triple importance'] + ['regular importance #%d' % (k + 1) for k in
                                                                    range(self.n_features - 2)]

    @property
    def training_data(self):
        if self._xtrain is None:
            self._xtrain, self._ytrain = self.get_samples()

        return self._xtrain, self._ytrain

    def get_samples(self):
        # Let's define dataset and label

        x = self.random_state.rand(self.n_samples, self.n_features)

        y = self._function(x)

        x = GeneralUtils.add_nans(x)

        return x, y

    def get_model(self):
        x, y = self.training_data
        self.model.fit(x, y)
        return self.model

class LinearClassificationDataset(LinearBase):
    def __init__(self, **kwargs):
        super(LinearClassificationDataset, self).__init__(
            model=GeneralUtils.simple_imputation_pipeline(
                RandomForestClassifier(random_state=0, n_estimators=3, n_jobs=1)),
            **kwargs)
        self.y_type = int

class LinearRegressionDataset(LinearBase):
    def __init__(self, **kwargs):
        super(LinearRegressionDataset, self).__init__(
            model=GeneralUtils.simple_imputation_pipeline(
                RandomForestRegressor(random_state=0, n_estimators=3, n_jobs=1)),
            **kwargs)
        self.y_type = float



