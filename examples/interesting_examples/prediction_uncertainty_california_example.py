import numpy
from matplotlib import pyplot as plt

from pytolemaic import Metrics
from pytolemaic import PyTrust
from pytolemaic.utils.general import GeneralUtils
from resources.datasets.california_housing import CaliforniaHousing


def run():
    # init
    dataset = CaliforniaHousing()
    regressor = dataset.get_model()
    train, test = dataset.as_dmd()

    test_1st_half, test_2nd_half = test.split(ratio=0.2)

    metric = Metrics.mae

    pytrust = PyTrust(
        model=regressor,
        xtest=test_1st_half,
        metric=metric)

    xtest2, ytest2 = test_2nd_half.values, test_2nd_half.target

    method = 'mae'  # or 'probability'
    uncertainty_model = pytrust.create_uncertainty_model(method=method)
    yp = uncertainty_model.predict(xtest2)  # same as model.predict
    uncertainty = uncertainty_model.uncertainty(xtest2)  # uncertainty value
    print('y_true, y_pred, uncertainty')
    print(numpy.concatenate([ytest2.reshape(-1, 1), yp.reshape(-1, 1), uncertainty.reshape(-1, 1)], axis=1)[:10])

    # example
    plt.figure()
    uncertainty_levels = numpy.array([0, 0.2, 0.4, 0.6, 0.8, 1.0001])
    mn, mx = 1, 0

    # uncertainty model may be based on 'confidence' or 'probability' for classification, and 'mae' or 'rmse' for regression
    for method in ['quantile', 'mae', 'rmse']:
        print('working on method %s' % method)
        # train uncertainty model
        uncertainty_model = pytrust.create_uncertainty_model(method=method)
        yp = uncertainty_model.predict(xtest2)  # same as model.predict
        uncertainty = uncertainty_model.uncertainty(xtest2)  # uncertainty value

        level_inds = numpy.digitize(uncertainty.ravel(), uncertainty_levels)

        performance = []
        for ibin in range(len(uncertainty_levels) - 1):
            inds = level_inds == ibin + 1
            if not any(inds):
                performance.append(0)
            else:
                subset_score = metric.function(y_true=ytest2[inds], y_pred=yp[inds])
                performance.append(subset_score)

        uncertainty_levels_middle = (uncertainty_levels[1:] + uncertainty_levels[:-1]) / 2

        plt.figure(1)
        plt.plot(uncertainty_levels_middle, performance, '*-b' if method == 'mae' else '*-r')

        plt.xlabel("Uncertainty level")
        plt.ylabel("{} Score".format(metric.name))
        plt.title("{} score vs uncertainty level".format(metric.name))
        plt.legend(['method=mae', 'method=rmse', 'method=quantile'], loc='upper right')

        print(uncertainty_levels_middle)
        print(GeneralUtils.f3(performance))
        mn = min(min(performance), mn)
        mx = max(max(performance), mx)
        uncertainty_model.plot_calibration_curve()

    plt.figure(1)
    # emphasize bins
    for level in uncertainty_levels:
        plt.plot([level, level], [mn, mx], '-k')


if __name__ == '__main__':
    run()
    plt.show()
