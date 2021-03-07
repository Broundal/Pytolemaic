from pprint import pprint

import numpy

from pytolemaic import Metrics
from pytolemaic import PyTrust
from resources.datasets.linear import LinearClassificationDataset


def run():
    ## For this example we create train/test data representing a linear function
    # PyTrust supports both numpy and pandas.DataFrame.

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

    # obtain more samples, never seen before, for which we want to measure uncertainty
    x_new_test, y_new_test = dataset.get_samples()


    # uncertainty model may be based on 'confidence' or 'probability' for classification, and 'mae' or 'rmse' for regression
    for method in ['confidence', 'probability']:
        # train uncertainty model
        uncertainty_model = pytrust.create_uncertainty_model(method=method)

        pprint(uncertainty_model.uncertainty_analysis_output)

        yp = uncertainty_model.predict(x_new_test)  # this is same as model.predict

        # and now it's possible to calculate uncertainty on new samples!
        uncertainty = uncertainty_model.uncertainty(x_new_test)

        # let's see whether we can use this value to separate good samples and bad samples:

        base_score = metric.function(y_new_test, yp)
        p25, p50, p75 = numpy.percentile(numpy.unique(uncertainty), [25, 50, 75])

        # samples with low uncertainty
        good = (uncertainty < p25).ravel()
        subset_good_score = metric.function(
            y_true=y_new_test[good], y_pred=yp[good])

        # samples with high uncertainty
        bad = (uncertainty > p75).ravel()
        subset_bad_score = metric.function(
            y_true=y_new_test[bad], y_pred=yp[bad])

        print('\n\n\n#########################################\n')
        print("performance for method *{}*".format(method))
        print('{} score is {:0.3f}'.format(metric.name, base_score))
        print('{} score for samples with high confidence is {:0.3f}'.format(metric.name, subset_good_score))
        print('{} score for samples with low confidence is {:0.3f}'.format(metric.name, subset_bad_score))
        print('{:0.3f} < {:0.3f} < {:0.3f} = {}'.format(subset_bad_score, base_score, subset_good_score,
                                                        subset_bad_score < base_score < subset_good_score))


if __name__ == '__main__':
    run()
