from pprint import pprint

import numpy
import pandas
from pytolemaic.utils.general import GeneralUtils
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from pytolemaic.pytrust import PyTrust


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

    xtrain = GeneralUtils.add_nans(xtrain)
    xtest = GeneralUtils.add_nans(xtest)

    ## Let's train a classifier
    classifier = GeneralUtils.simple_imputation_pipeline(
        RandomForestClassifier(random_state=0, n_estimators=3))


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



    pytrust = PyTrust(
        model=classifier,
        xtrain=xtrain, ytrain=ytrain,
        xtest=xtest, ytest=ytest,
        sample_meta_train=sample_meta_train, sample_meta_test=sample_meta_test,
        columns_meta=columns_meta,
        metric=metric,
        splitter=splitter)

    scoring_report = pytrust.scoring_report()

    score_value = scoring_report.metric_scores[metric].value
    ci_low = scoring_report.metric_scores[metric].ci_low
    ci_high = scoring_report.metric_scores[metric].ci_high
    quality = scoring_report.quality

    print('{} score is {:0.3f}'.format(metric, score_value))
    print('Score quality is {:0.3f}'.format(quality))
    print('Confidence interval is [{:0.3f}, {:0.3f}]'.format(ci_low, ci_high))

    pprint(scoring_report)



if __name__ == '__main__':
    run()
