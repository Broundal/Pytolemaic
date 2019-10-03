
from pprint import pprint

import numpy
import pandas
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from pytolemaic.pytrust import SklearnTrustBase


## For this example we create train/test data representing a linear function
# both numpy and pandas.DataFrame is ok.
from pytolemaic.utils.dmd import DMD
from pytolemaic.utils.general import GeneralUtils
from pytolemaic.utils.metrics import Metrics

def run():

    xtrain = numpy.random.rand(10000, 3)
    columns_names = ['zero importance', 'regular importance', 'triple importance']

    # 1st has no importance, while 3rd has double importance
    ytrain = 0 * xtrain[:, 0] + 1 * xtrain[:, 1] + 3 * xtrain[:, 2]


    xtest = numpy.random.rand(10000, 3)
    # 1st has no importance, while 3rd has double importance
    ytest = 0 * xtest[:, 0] + 1 * xtest[:, 1] + 3 * xtest[:, 2]


     ## Let's train a regressor
    regressor = GeneralUtils.simple_imputation_pipeline(
        RandomForestRegressor(random_state=0, n_estimators=3))
    regressor.fit(xtrain, ytrain)

    ## set metric
    metric = Metrics.mae.name

    ## set splitting strategy
    splitter = 'shuffled'

    ## sample meta data (e.g. sample weight) - empty in this example
    sample_meta_train = None
    sample_meta_test = None

    # set the feature names names
    columns_meta = {DMD.FEATURE_NAMES: [name for name in columns_names]}



    pytrust = SklearnTrustBase(
        model=regressor,
        Xtrain=xtrain, Ytrain=ytrain,
        Xtest=xtest, Ytest=ytest,
        sample_meta_train=sample_meta_train, sample_meta_test=sample_meta_test,
        columns_meta=columns_meta,
        metric=metric,
        splitter=splitter)

    sensitivity_report = pytrust.sensitivity_report()
    print(pprint(sensitivity_report))

if __name__ == '__main__':
    run()
