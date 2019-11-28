import numpy
from sklearn.ensemble import RandomForestClassifier

from examples.datasets.linear import LinearClassificationDataset
from pytolemaic.pytrust import PyTrust
from pytolemaic.utils.dmd import DMD
from pytolemaic.utils.general import GeneralUtils
from pytolemaic.utils.metrics import Metrics


def run():

    # Obtain simple classification dataset. Use LinearRegressionDataset for regression
    dataset = LinearClassificationDataset()

    # uncertainty model requires test data and trained model
    xtest, ytest = dataset.get_samples()
    classifier = dataset.get_model()

    ## set metric
    metric = Metrics.recall

    pytrust = PyTrust(
        model=classifier,
        xtest=xtest, ytest=ytest,
        metric=metric)

    # uncertainty model may be based on 'confidence' or 'probability' for classification, and 'mae' or 'rmse' for regression
    for method in ['confidence', 'probability']:
        uncertainty_model = pytrust.create_uncertainty_model(method=method)

        # create another test set, this time to test uncertainty
        x_new_test, y_new_test = dataset.get_samples()

        new_test = DMD(x=x_new_test, y=y_new_test)

        yp = uncertainty_model.predict(new_test)  # this is same as model.predict
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

        print('\n\n\n#########################################333\n')
        print("performance for method *{}*".format(method))
        print('{} score is {:0.3f}'.format(metric.name, base_score))
        print('{} score for samples with high confidence is {:0.3f}'.format(metric.name, subset_good_score))
        print('{} score for samples with low confidence is {:0.3f}'.format(metric.name, subset_bad_score))
        print('{:0.3f} < {:0.3f} < {:0.3f} = {}'.format(subset_bad_score, base_score, subset_good_score,
                                                        subset_bad_score < base_score < subset_good_score))


if __name__ == '__main__':
    run()
