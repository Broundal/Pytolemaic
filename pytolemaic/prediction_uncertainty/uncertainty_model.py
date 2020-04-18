import multiprocessing

import numpy
import sklearn
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import brier_score_loss

from pytolemaic.utils.constants import REGRESSION, CLASSIFICATION
from pytolemaic.utils.dmd import DMD
from pytolemaic.utils.general import GeneralUtils
from pytolemaic.utils.metrics import Metrics


class UncertaintyModelBase():

    def __init__(self, model, uncertainty_method: str,
                 ptype=None,
                 supported_methods: list = None):
        self.model = model
        self.uncertainty_method = uncertainty_method
        self.dmd_supported = None
        self.is_classification = GeneralUtils.is_classification(model)

        if (self.is_classification and ptype != CLASSIFICATION) or \
                (not self.is_classification and ptype == CLASSIFICATION):
            raise ValueError(
                "{} does not support {}".format(type(self), ptype))

        if self.uncertainty_method not in supported_methods:
            raise NotImplementedError(
                "Uncertainty method {} is not in supported methods={}".format(
                    self.uncertainty_method, supported_methods))

    def uncertainty(self, dmd: DMD):
        raise NotImplementedError("")

    def fit_uncertainty_model(self, dmd_test, **kwargs):
        raise NotImplementedError("")

    def fit(self, dmd_test: DMD, **kwargs):
        self.dmd_supported = GeneralUtils.dmd_supported(model=self.model,
                                                        dmd=dmd_test)

        self.fit_uncertainty_model(dmd_test, **kwargs)
        return self

    def predict(self, dmd: DMD):
        if self.dmd_supported:
            if not isinstance(dmd, DMD):
                dmd = DMD(x=dmd)
            return self.model.predict(dmd)
        else:
            if isinstance(dmd, DMD):
                x = dmd.values
            else:
                x = dmd
            return self.model.predict(x)

    def predict_proba(self, dmd: DMD):
        if self.dmd_supported:
            if not isinstance(dmd, DMD):
                raise ValueError("DMD supported but input is not dmd")
            return self.model.predict_proba(dmd)
        else:
            if isinstance(dmd, DMD):
                x = dmd.values
            else:
                x = dmd
            return self.model.predict_proba(x)

    def plot_calibration_curve(self):
        raise NotImplementedError


class UncertaintyModelRegressor(UncertaintyModelBase):

    def __init__(self, model, uncertainty_method='rmse'):
        super(UncertaintyModelRegressor, self).__init__(
            model=model, uncertainty_method=uncertainty_method,
            ptype=REGRESSION, supported_methods=['mae', 'rmse'])
        self._n_bins = 10
        self.actual_error = None
        self.mean_predicted_error = None
        self._cal_curve_uncertainty = None

    def fit_uncertainty_model(self, dmd_test, n_jobs=multiprocessing.cpu_count() - 1,
                              metric=Metrics.r2, **kwargs):

        dmd_test, cal_curve_samples = dmd_test.split(ratio=0.1)

        if self.uncertainty_method in ['mae']:
            estimator = RandomForestRegressor(
                random_state=0, n_jobs=n_jobs,
                n_estimators=kwargs.pop('n_estimators', 100))

            self.uncertainty_model = GeneralUtils.simple_imputation_pipeline(
                estimator)

            yp = self.predict(dmd_test)
            self.uncertainty_model.fit(dmd_test.values,
                                       numpy.abs(
                                           dmd_test.target.ravel() - yp.ravel()))
        elif self.uncertainty_method in ['rmse']:
            estimator = RandomForestRegressor(
                random_state=0, n_jobs=n_jobs,
                n_estimators=kwargs.pop('n_estimators', 100))

            self.uncertainty_model = GeneralUtils.simple_imputation_pipeline(
                estimator)

            yp = self.predict(dmd_test)
            self.uncertainty_model.fit(dmd_test.values,
                                       (dmd_test.target.ravel() - yp.ravel()) ** 2)

        else:
            raise NotImplementedError("Method {} is not implemented"
                                      .format(self.uncertainty_method))

        # calibration curve
        y_pred = self.predict(cal_curve_samples).ravel()
        y_true = cal_curve_samples.target.ravel()

        delta = numpy.abs(y_true - y_pred)
        uncertainty = self.uncertainty(cal_curve_samples).ravel()
        self._cal_curve_uncertainty = uncertainty

        bins = numpy.linspace(0., max(uncertainty) + 1e-8, self._n_bins + 1).ravel()
        binids = numpy.digitize(uncertainty, bins) - 1

        bin_sums = numpy.bincount(binids, weights=uncertainty, minlength=len(bins))
        bin_true = numpy.bincount(binids, weights=delta, minlength=len(bins))
        bin_total = numpy.bincount(binids, minlength=len(bins))

        nonzero = bin_total != 0
        self.actual_error = (bin_true[nonzero] / bin_total[nonzero])
        self.mean_predicted_error = (bin_sums[nonzero] / bin_total[nonzero])

        # calibration curve by metric

        performance = []
        uncertainty_levels_middle = []
        for ibin in range(len(bins) - 1):
            inds = binids == ibin
            if numpy.sum(inds) < 5:
                continue

            subset_score = metric.function(y_true=y_true[inds], y_pred=y_pred[inds])
            performance.append(subset_score)
            uncertainty_levels_middle.append((bins[ibin] + bins[ibin + 1]) / 2)

        self._cal_curve_metric = {'uncertainty': uncertainty_levels_middle,
                                  'score': performance,
                                  'metric': metric.name}

    def uncertainty(self, dmd: DMD):
        if isinstance(dmd, DMD):
            x = dmd.values
        else:
            x = dmd

        if self.uncertainty_method in ['mae']:
            out = self.uncertainty_model.predict(x)
            return out.reshape(-1, 1)
        elif self.uncertainty_method in ['rmse']:
            out = numpy.sqrt(self.uncertainty_model.predict(x))
            return out.reshape(-1, 1)
        else:
            raise NotImplementedError("Method {} is not implemented"
                                      .format(self.uncertainty_method))

    def plot_calibration_curve(self):
        fig = plt.figure(figsize=(10, 10))
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax2 = plt.subplot2grid((3, 1), (2, 0))

        ax1.plot([min(self.mean_predicted_error), max(self.mean_predicted_error)],
                 [min(self.mean_predicted_error), max(self.mean_predicted_error)], "k:", label="Perfectly calibrated")

        ax1.plot(self.mean_predicted_error, self.actual_error, "s-",
                 color='b')

        # todo: remove _cal_curve_uncertainty from self
        ax2.hist(self._cal_curve_uncertainty, range=(0, 1.),
                 bins=self._n_bins,
                 color='b',
                 histtype="step", lw=2)

        ax1.set_ylabel("Actual error")
        ax1.set_ylim([min(self.actual_error) * 0.95, max(self.actual_error) * 1.05])
        ax1.legend(loc="lower right")
        ax1.set_title("Calibartion Curve for method {} (Regression)".format(self.uncertainty_method))

        ax2.set_xlabel("Mean predicted error")
        ax2.set_ylabel("Count")
        ax2.legend(loc="upper center", ncol=2)

        # curve by metric
        plt.figure()
        plt.plot(self._cal_curve_metric['uncertainty'], self._cal_curve_metric['score'], '*-r')

        plt.xlabel("Uncertainty level ({})".format(self.uncertainty_method))
        plt.ylabel("{} score".format(self._cal_curve_metric['metric']))
        plt.title("{} score vs uncertainty level".format(self._cal_curve_metric['metric']))


class UncertaintyModelClassifier(UncertaintyModelBase):

    def __init__(self, model, uncertainty_method='confidence'):
        super(UncertaintyModelClassifier, self).__init__(model=model,
                                                         uncertainty_method=uncertainty_method,
                                                         ptype=CLASSIFICATION,
                                                         supported_methods=[
                                                             'probability',
                                                             'confidence']
                                                         )

        self._brier_loss = -1
        self._fraction_of_positives = None
        self._mean_uncertainty = None
        self._cal_curve_uncertainty = None
        self._n_bins = 10

    def fit_uncertainty_model(self, dmd_test, n_jobs=multiprocessing.cpu_count() - 1,
                              metric=Metrics.recall,
                              **kwargs):

        if self.uncertainty_method in ['probability']:
            cal_curve_samples = dmd_test
            # no fit logic required

        elif self.uncertainty_method in ['confidence']:
            dmd_test, cal_curve_samples = dmd_test.split(ratio=0.1)

            estimator = RandomForestClassifier(
                random_state=0, n_jobs=n_jobs, n_estimators=100)

            self.uncertainty_model = GeneralUtils.simple_imputation_pipeline(
                estimator)

            y_pred = self.predict(dmd_test)
            is_correct = numpy.array(y_pred.ravel() == dmd_test.target.ravel(),
                                     dtype=int)

            # bug here
            self.uncertainty_model.fit(dmd_test.values, is_correct.ravel())

        else:
            raise NotImplementedError("Method {} is not implemented"
                                      .format(self.uncertainty_method))

        # calibration curve

        y_pred = self.predict(cal_curve_samples).ravel()
        y_true = cal_curve_samples.target.ravel()
        uncertainty = self.uncertainty(cal_curve_samples)
        self._cal_curve_uncertainty = uncertainty

        self._fraction_of_positives, self._mean_uncertainty = sklearn.calibration.calibration_curve(
            y_true=y_pred == y_true,
            y_prob=uncertainty,
            normalize=True,
            n_bins=self._n_bins,
            strategy='uniform')

        sample_weight = None
        self._brier_loss = brier_score_loss(
            y_true=y_pred == y_true,
            y_prob=1 - uncertainty,
            sample_weight=sample_weight,
            pos_label=1)

        # calibration curve by metric

        uncertainty = uncertainty / max(uncertainty)

        bins = numpy.linspace(0., max(uncertainty) + 1e-8, 5 + 1)
        binids = numpy.digitize(uncertainty.flatten(), bins.ravel()) - 1

        performance = []
        uncertainty_levels_middle = []
        for ibin in range(len(bins) - 1):
            inds = binids == ibin
            if numpy.sum(inds) < 5:
                continue

            subset_score = metric.function(y_true=y_true[inds], y_pred=y_pred[inds])
            performance.append(subset_score)
            uncertainty_levels_middle.append((bins[ibin] + bins[ibin + 1]) / 2)

        self._cal_curve_metric = {'uncertainty': uncertainty_levels_middle,
                                  'score': performance,
                                  'metric': metric.name}

    def uncertainty(self, dmd: DMD):

        if self.uncertainty_method in ['probability']:
            yproba = self.predict_proba(dmd)
            yproba += 1e-10 * numpy.random.RandomState(0).rand(*yproba.shape)
            max_probability = numpy.max(yproba, axis=1).reshape(-1, 1)
            delta = max_probability - yproba
            yproba[delta == 0] = 0

            # delta[numpy.sum(delta, axis=1)>=20,:] = 0 # i
            out = numpy.max(yproba, axis=1).reshape(-1, 1) / max_probability
            return GeneralUtils.f5(out).reshape(-1, 1)
        elif self.uncertainty_method in ['confidence']:
            if isinstance(dmd, DMD):
                x = dmd.values
            else:
                x = dmd
            # return the probability it's a mistake
            out = self.uncertainty_model.predict_proba(x)[:, 0]
            return GeneralUtils.f5(out).reshape(-1, 1)
        else:
            raise NotImplementedError("Method {} is not implemented"
                                      .format(self.uncertainty_method))

    def plot_calibration_curve(self):
        fig = plt.figure(figsize=(10, 10))
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax2 = plt.subplot2grid((3, 1), (2, 0))

        ax1.plot([0, 1], [1, 0.5], "k:", label="Perfectly calibrated")

        ax1.plot(self._mean_uncertainty, self._fraction_of_positives, "s-",
                 color='b',
                 label="brier loss=%1.3f" % self._brier_loss)

        # todo: remove y_proba from self
        ax2.hist(self._cal_curve_uncertainty, range=(0, 1.),
                 bins=self._n_bins,
                 color='b',
                 histtype="step", lw=2)

        ax1.set_ylabel("Fraction of correct predictions")
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc="lower right")
        ax1.set_title("Calibartion Curve for method {}".format(self.uncertainty_method))

        ax2.set_xlabel("Mean uncertainty")
        ax2.set_ylabel("Count")
        ax2.legend(loc="upper center", ncol=2)

        # curve by metric
        plt.figure()
        plt.plot(self._cal_curve_metric['uncertainty'], self._cal_curve_metric['score'], '*-r')

        plt.xlabel("Uncertainty level ({})".format(self.uncertainty_method))
        plt.ylabel("{} score".format(self._cal_curve_metric['metric']))
        plt.title("{} score vs uncertainty level".format(self._cal_curve_metric['metric']))
