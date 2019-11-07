from pprint import pprint

import numpy
import pandas
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from pytolemaic.pytrust import SklearnTrustBase


## For this example we create train/test data representing a linear function
# both numpy and pandas.DataFrame is ok.
from pytolemaic.utils.dmd import DMD
from pytolemaic.utils.metrics import Metrics

def run():

    xtrain = numpy.random.rand(10000, 3)
    columns_names = ['zero importance', 'regular importance', 'triple importance']

    # 1st has no importance, while 3rd has double importance
    ytrain = 0 * xtrain[:, 0] + 1 * xtrain[:, 1] + 3 * xtrain[:, 2]
    ytrain = ytrain.astype(int)

    xtest = numpy.random.rand(10000, 3)
    # 1st has no importance, while 3rd has double importance
    ytest = 0 * xtest[:, 0] + 1 * xtest[:, 1] + 3 * xtest[:, 2]
    ytest = ytest.astype(int)

     ## Let's train a regressor
    classifier = RandomForestClassifier(random_state=0, n_estimators=3)
    classifier.fit(xtrain, ytrain.ravel())

    ## set metric
    metric = Metrics.recall.name

    ## set splitting strategy
    splitter = 'shuffled' # todo: support stratified

    ## sample meta data (e.g. sample weight) - empty in this example
    sample_meta_train = None
    sample_meta_test = None

    # set the feature names names
    columns_meta = {DMD.FEATURE_NAMES: [name for name in columns_names]}



    pytrust = SklearnTrustBase(
        model=classifier,
        xtrain=xtrain, ytrain=ytrain,
        xtest=xtest, ytest=ytest,
        sample_meta_train=sample_meta_train, sample_meta_test=sample_meta_test,
        columns_meta=columns_meta,
        metric=metric,
        splitter=splitter)

    scoring_report = pytrust.scoring_report()
    print('{} score is {:0.3f}'.format(metric, scoring_report['Score'][metric]['value']))
    print('Score quality is {:0.3f}'.format(scoring_report['Quality']))
    print('Confidence interval is [{:0.3f}, {:0.3f}]'.format(scoring_report['Score'][metric]['ci_low'], scoring_report['Score'][metric]['ci_high']))

    print(pprint(scoring_report))



if __name__ == '__main__':
    run()
