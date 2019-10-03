import numpy
from sklearn.ensemble import RandomForestClassifier

from pytolemaic.pytrust import SklearnTrustBase
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
    classifier.fit(xtrain, ytrain)

    ## set metric
    metric = Metrics.recall

    ## set splitting strategy
    splitter = 'shuffled' # todo: support stratified

    ## sample meta data (e.g. sample weight) - empty in this example
    sample_meta_train = None
    sample_meta_test = None

    # set the feature names names
    columns_meta = {DMD.FEATURE_NAMES: [name for name in columns_names]}



    pytrust = SklearnTrustBase(
        model=classifier,
        Xtrain=xtrain, Ytrain=ytrain,
        Xtest=xtest, Ytest=ytest,
        sample_meta_train=sample_meta_train, sample_meta_test=sample_meta_test,
        columns_meta=columns_meta,
        metric=metric,
        splitter=splitter)

    uncertainty_model = pytrust.create_uncertainty_model(method='confidence')


    # create another test set, this time to test uncertainty
    x_new_test = numpy.random.rand(10000, 3)
    y_new_test = 0 * x_new_test[:, 0] + 1 * x_new_test[:, 1] + 3 * x_new_test[:, 2]
    y_new_test = y_new_test.astype(int)


    new_test = DMD(x=x_new_test, y=y_new_test)

    yp = uncertainty_model.predict(new_test) # this is same as model.predict
    base_score = metric.function(new_test.target, yp)


    uncertainty = uncertainty_model.uncertainty(new_test)
    p25, p50, p75 = numpy.percentile(numpy.unique(uncertainty), [25, 50, 75])

    # samples with low uncertainty
    good = (uncertainty < p25).ravel()
    subset_good_score = metric.function(
        y_true=y_new_test[good], y_pred=yp[good])

    # samples with high uncertainty
    bad = (uncertainty > p75).ravel()
    subset_bad_score = metric.function(
        y_true=y_new_test[bad], y_pred=yp[bad])

    print('general score is {:0.3f}'.format(base_score))
    print('score for samples with high confidence is {:0.3f}'.format(subset_good_score))
    print('score for samples with low confidence is {:0.3f}'.format(subset_bad_score))
    print('{:0.3f} < {:0.3f} < {:0.3f} = {}'.format(subset_bad_score, base_score, subset_good_score, subset_bad_score<base_score<subset_good_score))

if __name__ == '__main__':
    run()