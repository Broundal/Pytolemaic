import multiprocessing

import numpy
import sklearn
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline

from pytolemaic.utils.constants import REGRESSION, CLASSIFICATION
from pytolemaic.utils.dmd import DMD
from pytolemaic.utils.general import GeneralUtils
from pytolemaic.utils.metrics import Metrics


class UncertaintyModelBase():

    def __init__(self, model, uncertainty_method: str,
                 ptype=None,
                 supported_methods: list = None):
        self.model = model
        self.uncertainty_method = uncertainty_method if uncertainty_method != 'default' else supported_methods[0]
        self.dmd_supported = None
        self.is_classification = GeneralUtils.is_classification(model)
        self.uncertainty_analysis_output = None

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

    @classmethod
    def _get_decision_paths(cls, values, tree, y, uncertainty_threshold, low):

        if low:
            vals = values[y.ravel() <= uncertainty_threshold]
        else:
            vals = values[y.ravel() >= uncertainty_threshold]

        # vals = vals[:min(100, len(vals)), :]
        # decision_path_nodes = [tree.decision_path(val.reshape(1,-1)).indices for val in vals]
        decision_path_nodes = tree.decision_path(vals)

        return decision_path_nodes

    @classmethod
    def _get_nodes(cls, tree, uncertainty_threshold, low, min_samples_in_node=100):
        # get candidates nodes by n_samples and value.
        ids = tree.tree_.n_node_samples >= min_samples_in_node
        values = tree.tree_.value.ravel()

        impurity = tree.tree_.impurity
        impurity[impurity < 0] = 1e100
        rmse = numpy.sqrt(impurity).ravel()
        rmse[tree.tree_.impurity < 0] = -1

        if low:
            ids = ids & (values <= uncertainty_threshold) & (values >= rmse * 2)
        else:
            ids = ids & (values >= uncertainty_threshold) & (values >= rmse * 2)

        if not any(ids):
            return numpy.array([])

        nodes = numpy.arange(tree.tree_.node_count)[ids]

        # prune nodes
        for i in reversed(nodes):  # high to low
            for children in [tree.tree_.children_left[i], tree.tree_.children_right[i]]:
                nodes[nodes == children] = -1
        nodes = nodes[nodes > 0]
        return nodes

    def uncertainty_analysis(self, dmd_train: DMD = None, dmd_test: DMD = None,
                             n_jobs=multiprocessing.cpu_count() - 1, max_depth=5, min_samples_in_area=0.05,
                             percentile=10) -> dict:
        """
        Provides an analysis of areas with low/high uncertainty. Can assist in understanding where the model is good, and where it's not.
        :param dmd_train:
        :param dmd_test:
        :param n_jobs:
        :param max_depth: Maximum number of conditions to define an area.
        :param min_samples_in_area: Minimal train+test samples in reported area.
        :param percentile: percentile over uncertainty to provide threshold for high and low uncertainties.
        :return:
        """
        if dmd_train is not None:
            dmd = dmd_train
            if dmd_test is not None:
                dmd.append(dmd_test)
        elif dmd_test is not None:
            dmd = dmd_test
        else:
            raise ValueError("Require either dmd_train or dmd_test or both")

        if min_samples_in_area < 1:
            min_samples_in_area = int(dmd.n_samples * min_samples_in_area)

        # train RF on uncertainty output (because it's not necessarity RF in uncertainty estimator)
        estimator = RandomForestRegressor(random_state=0, n_jobs=n_jobs, max_depth=max_depth, n_estimators=50,
                                          min_samples_split=max(min_samples_in_area//2, 2),
                                          max_features="sqrt")

        pipeline = GeneralUtils.simple_imputation_pipeline(
            estimator)  # todo better imputation scheme whenever a model is created by package
        y = self.uncertainty(dmd)
        pipeline.fit(dmd.values, y.ravel())

        # extract RF from pipeline
        names, estimators = zip(*pipeline.steps)
        imputer = estimators[-2]
        estimator = estimators[-1]

        low_uncertainty = numpy.percentile(y, percentile)
        high_uncertainty = numpy.percentile(y, 100 - percentile)

        areas_of_extreme_uncertainty = {'low': [], 'high': []}
        for mode in ['low', 'high']:
            extreme_areas = []
            for tree in estimator.estimators_:
                uncertainty_threshold = low_uncertainty if mode == 'low' else high_uncertainty

                # get nodes of interest
                nodes = self._get_nodes(tree, uncertainty_threshold=uncertainty_threshold, low=mode == 'low',
                                        min_samples_in_node=min_samples_in_area)
                if len(nodes) == 0:
                    continue

                # get relevant decision paths of nodes
                node_paths = self._get_decision_paths(imputer.transform(dmd.values), tree, y,
                                                      uncertainty_threshold=uncertainty_threshold, low=mode == 'low')

                # for each node, get msg
                for node in nodes:
                    # search for appropriate path.
                    found = None
                    for node_path_csr in node_paths:
                        nodes_path = node_path_csr.indices
                        if node in nodes_path:
                            found = nodes_path
                            break

                    assert found is not None

                    # this can be moved elsewhere if needed
                    from pytolemaic.analysis_logic.prediction_analysis.decision_tree_report import DecisionTreeExplainer
                    conditions = DecisionTreeExplainer._list_of_conditions(decision_tree=tree,
                                                                           nodes=found,
                                                                           feature_names=dmd.feature_names)

                    mean = tree.tree_.value[node][0][0]
                    rmse = numpy.sqrt(tree.tree_.impurity[node])

                    msg = "Uncertainty( {} ) = {}.".format(" & ".join(conditions),
                                                           "{:.2g}+-{:.2g}".format(mean, rmse))
                    # extreme_areas.append(msg)
                    extreme_areas.append(dict(where=" & ".join(conditions),
                                              uncertainty="{:.2g}+-{:.1g}".format(mean, rmse),
                                              n_samples="{} out of {}".format(tree.tree_.n_node_samples[node],
                                                                              dmd.n_samples)
                                              ))

            areas_of_extreme_uncertainty[mode] = extreme_areas

        areas_of_extreme_uncertainty = {'Low uncertainty': areas_of_extreme_uncertainty['low'],
                                        'High uncertainty': areas_of_extreme_uncertainty['high']}
        # from pprint import pprint
        # pprint(areas_of_extreme_uncertainty)
        return areas_of_extreme_uncertainty

class UncertaintyModelRegressor(UncertaintyModelBase):

    def __init__(self, model, uncertainty_method='mae'):
        super(UncertaintyModelRegressor, self).__init__(
            model=model, uncertainty_method=uncertainty_method,
            ptype=REGRESSION, supported_methods=['mae', 'rmse', 'quantile'])
        self._n_bins = 10
        self.actual_error = None
        self.mean_predicted_error = None
        self._cal_curve_uncertainty = None

    def fit_uncertainty_model(self, dmd_test, n_jobs=multiprocessing.cpu_count() - 1,
                              metric=Metrics.r2, **kwargs):

        dmd_test, cal_curve_samples = dmd_test.split(ratio=0.1)

        if self.uncertainty_method in ['mae', 'default']:
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
        elif self.uncertainty_method in ['quantile']:
            if isinstance(self.model, Pipeline):

                names, estimators = zip(*self.model.steps)
                estimator = estimators[-1]
                if hasattr(estimator, 'estimators_'):
                    self.model.estimators_ = estimator.estimators_

            if not hasattr(self.model, 'estimators_'):
                raise ValueError(
                    "'quantile' method can work only for models with estimators_ attrbiute. Use method='mae' instead")

        else:
            raise NotImplementedError("Method {} is not implemented"
                                      .format(self.uncertainty_method))

        self.uncertainty_analysis_output = self.uncertainty_analysis(dmd_train=None, dmd_test=dmd_test)

        # the following section is purely for plotting/analysis purposes

        ## calibration curve
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

        nonzero = bin_total > 10
        self.actual_error = (bin_true[nonzero] / bin_total[nonzero])
        self.mean_predicted_error = (bin_sums[nonzero] / bin_total[nonzero])

        # calibration curve by metric

        performance = []
        uncertainty_levels_middle = []
        for ibin in range(len(bins) - 1):
            inds = binids == ibin
            if numpy.sum(inds) < 10:
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
        elif self.uncertainty_method in ['quantile']:

            percentile = 10

            ys = numpy.zeros((len(x), len(self.model.estimators_)))
            for i, pred in enumerate(self.model.estimators_):
                ys[:, i] = pred.predict(x)

            err_down = numpy.percentile(ys, percentile, axis=1)
            err_up = numpy.percentile(ys, 100 - percentile, axis=1)

            return ((err_up - err_down) / 2).reshape(-1, 1)

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
                                                         supported_methods=['confidence', 'probability'])

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

        self.uncertainty_analysis_output = self.uncertainty_analysis(dmd_train=None, dmd_test=dmd_test)

        # analysis for plots :

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
            if numpy.sum(inds) < 10:
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
