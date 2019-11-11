
from pprint import pprint

import numpy
import pandas
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from pytolemaic.pytrust import PyTrust


## For this example we create train/test data representing a linear function
# both numpy and pandas.DataFrame is ok.
from pytolemaic.utils.dmd import DMD
from pytolemaic.utils.general import GeneralUtils
from pytolemaic.utils.metrics import Metrics

def run():

    xtrain = numpy.random.rand(10000, 3)
    columns_names = ['no importance feature', 'regular feature', 'triple importance feature']

    # 1st has no importance, while 3rd has double importance
    ytrain = 0 * xtrain[:, 0] + 1 * xtrain[:, 1] + 3 * xtrain[:, 2]


    xtest = numpy.random.rand(10000, 3)
    # 1st has no importance, while 3rd has double importance
    ytest = 0 * xtest[:, 0] + 1 * xtest[:, 1] + 3 * xtest[:, 2]

    xtrain = GeneralUtils.add_nans(xtrain)
    xtest = GeneralUtils.add_nans(xtest)

     ## Let's train a regressor
    regressor = GeneralUtils.simple_imputation_pipeline(
        RandomForestRegressor(random_state=0, n_estimators=3))
    regressor.fit(xtrain, ytrain.ravel())

    ## set metric
    metric = Metrics.mae.name

    ## set splitting strategy
    splitter = 'shuffled'

    ## sample meta data (e.g. sample weight) - empty in this example
    sample_meta_train = None
    sample_meta_test = None

    # set the feature names names
    columns_meta = {DMD.FEATURE_NAMES: [name for name in columns_names]}



    pytrust = PyTrust(
        model=regressor,
        xtrain=xtrain, ytrain=ytrain,
        xtest=xtest, ytest=ytest,
        sample_meta_train=sample_meta_train, sample_meta_test=sample_meta_test,
        columns_meta=columns_meta,
        metric=metric,
        splitter=splitter)

    sensitivity_report = pytrust.sensitivity_report()
    pprint(sensitivity_report.simplified_keys())

if __name__ == '__main__':
    run()
