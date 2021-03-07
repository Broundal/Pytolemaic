import logging

import numpy
from sklearn import tree
from sklearn.base import ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, _tree

from pytolemaic.utils.constants import FeatureTypes
from pytolemaic.utils.dmd import DMD
from pytolemaic.utils.general import GeneralUtils
from pytolemaic.utils.metrics import Metrics
from resources.datasets.california_housing import CaliforniaHousing


class DecisionTreeExplainer():

    def __init__(self, n_samples=100000, explanation_depth=4, min_samples_leaf=100, allowed_delta=0.05, digitize=2,
                 cat_imputer=None, num_imputer=None):
        """

        :param n_samples: number of samples to create near the sample for decision tree model training
        :param explanation_depth: default depth of decision tree. Actual depth may be changed according to allowed_delta or actual path needed.
        :param min_samples_leaf: min_samples_leaf parameter for decision tree.
        :param allowed_delta: created decision tree must match the model's prediction up to this tolerance
        :param digitize: rounding level of numeric values for better explainability. Increase value if high resolution is required.
        :param cat_imputer: imputer for categorical features. By default SimpleImputer("most_frequent")
        :param num_imputer: imputer for numerical features. By default SimpleImputer("mean")

        """
        self.n_samples = n_samples
        self.explanation_depth = explanation_depth
        self.min_samples_leaf = min_samples_leaf
        self.is_classification = None
        self.dt = None
        self.train_data_stats = None
        self.model = None
        self.cache = None
        self.allowed_delta = allowed_delta
        self.digitize = digitize

        self.imputers = (
        cat_imputer or SimpleImputer(strategy='most_frequent'), num_imputer or SimpleImputer(strategy='mean'))

    def _create_dt(self, max_depth):
        self.is_classification = GeneralUtils.is_classification(self.model)
        if self.is_classification:
            dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=self.min_samples_leaf, random_state=0,
                                        min_impurity_decrease=0.0001)
        else:
            dt = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=self.min_samples_leaf, random_state=0,
                                       min_impurity_decrease=0.0001)

        return dt

    def _call_imputer(self, values, features, oper='fit', key=0):
        if oper == 'fit':
            if features is not None and len(features) > 0:
                self.imputers[key].fit(values[:, features])
        else:
            if features is not None and len(features) > 0:
                values[:, features] = self.imputers[key].transform(values[:, features])

    def fit(self, dmd_train: DMD, model):

        self.model = model
        self.train_data_stats = self._calc_train_data_stats(dmd_train)
        if dmd_train.categorical_features is not None:
            self._call_imputer(dmd_train.values, features=dmd_train.categorical_features, oper='fit', key=0)
            self._call_imputer(dmd_train.values, features=dmd_train.numerical_features, oper='fit', key=1)
        else:
            self._call_imputer(dmd_train.values, features=numpy.arange(dmd_train.n_features), oper='fit', key=1)

    def _calc_train_data_stats(self, dmd_train: DMD):
        return dmd_train  # TODO: save stats instead of entire train data

    def _create_neighborhood(self, sample: numpy.ndarray, n_samples, train_data_stats: DMD):
        random_state = numpy.random.RandomState(0)

        values = random_state.permutation(train_data_stats.values)
        mean, std = numpy.nanmean(values, axis=0), numpy.nanstd(values, axis=0)
        mn, mx = numpy.nanmin(values, axis=0), numpy.nanmax(values, axis=0)

        if train_data_stats.categorical_features is not None:
            self._call_imputer(values, features=train_data_stats.categorical_features, oper='transform', key=0)
            self._call_imputer(values, features=train_data_stats.numerical_features, oper='transform', key=1)
        else:
            self._call_imputer(values, features=numpy.arange(train_data_stats.n_features), oper='transform', key=1)
        # make sure we have n_Samples points
        while len(values) < n_samples:
            values = numpy.concatenate([values, values], axis=0)

        values = values[:n_samples, :]

        # create neighborhood
        for icol, feature_type in enumerate(train_data_stats.feature_types):

            if feature_type == FeatureTypes.categorical:
                # values[:, icol] = values[:, icol] == sample[icol]
                vec = random_state.permutation(values[:, icol])
                if not numpy.isnan(sample[icol]):
                    vec[:len(vec) // 2] = sample[icol]
                values[:, icol] = random_state.permutation(vec)
            else:
                # quantiles = numpy.percentile(values[:,icol], numpy.arange(101))
                # quantiles[0] = quantiles[0]-1e-5
                # quantiles[-1] = quantiles[-1]+1e-5
                # up, low = numpy.argwhere(quantiles>sample[icol])[0], numpy.argwhere(sample[icol]>quantiles)[-1]
                # up = quantiles[min(up+5, len(quantiles)-1)]
                # low = quantiles[max(0, low-5)]
                # values[values[:, icol]>=up, icol] = up
                # values[values[:, icol]<=low, icol] = low
                if not numpy.isnan(sample[icol]):
                    sigma = std[icol] / 10

                    values[:, icol] = sample[icol] + random_state.randn(len(values)) * sigma
                    values[:, icol] = numpy.clip(values[:, icol], mn[icol], mx[icol])

                else:
                    values[:, icol] = random_state.permutation(values[:, icol])
                values[:, icol] = numpy.round(values[:, icol], self.digitize) + 0.005

                # uniques = numpy.unique(values[:, icol])
                # len_uniques = len(uniques)
                # if len_uniques > 100:
                #     bins = numpy.linspace(mn[icol], mx[icol], num=100).tolist() + [mx[icol]+1]
                # else:
                #     bins = uniques.tolist() + [mx[icol]+1]
                # bins = numpy.array(bins)
                # inds = numpy.digitize(values[:, icol], bins=bins, right=False)
                # values[:, icol] = numpy.round(0.5*(bins[inds-1] + bins[inds]), 2)

        logging.info('size of neighborhood:', values.shape)
        return values

    @classmethod
    def _decision_path_to_text(cls, decision_tree, sample, feature_names):
        nodes = decision_tree.decision_path(sample.reshape(1, -1)).indices
        return cls._nodes_path_to_text(decision_tree, nodes_path=nodes, feature_names=feature_names)

    @classmethod
    def _nodes_path_to_text(cls, decision_tree, nodes_path, feature_names):

        nodes = nodes_path

        list_of_conditions = cls._list_of_conditions(decision_tree, feature_names, nodes)

        is_classification = isinstance(decision_tree, ClassifierMixin)
        if is_classification:
            value_msg = 'the classes probabilities are '
            values = decision_tree.tree_.value[nodes[-1]][0]

            values = values / numpy.sum(values)
            values = numpy.round(values, 3)
            value_msg += '{}'.format(values)
        else:
            mean = decision_tree.tree_.value[nodes[-1]][0][0]
            mean = numpy.round(mean, 3)
            mse = decision_tree.tree_.impurity[nodes[-1]]
            sigma = numpy.sqrt(mse)
            value_msg = 'the predicted value is {} +- {:.1g}'.format(mean, sigma)

        msg = 'Since ' + ' and '.join(list_of_conditions) + ' then {}'.format(value_msg)
        depth = len(nodes)
        return msg, depth

    @classmethod
    def _list_of_conditions(cls, decision_tree, feature_names, nodes):
        features = decision_tree.tree_.feature[nodes]
        names = [feature_names[i] for i in features if i >= 0]
        thresholds = decision_tree.tree_.threshold[nodes]
        left = decision_tree.tree_.children_left[nodes]
        list_of_questions = []  # give condition per node in path
        dict_of_conditions = {}  # condition per feature in path.
        feature_lower_limit = {}
        feature_upper_limit = {}
        features_involved = []
        # since names is ordered, the condition gets tighter along the way, no need to compare limit with previous one.
        for i in range(len(names)):
            name = names[i]
            threshold = GeneralUtils.f3(thresholds[i])
            if left[i] in nodes:
                question = '[{} <= {}]'.format(name, threshold)
                feature_upper_limit[names[i]] = threshold
            else:
                question = '[{} > {}]'.format(name, threshold)
                feature_lower_limit[names[i]] = threshold

            list_of_questions.append(question)

        keys = set(list(feature_lower_limit.keys()) + list(feature_upper_limit.keys()))
        for k in keys:
            c1 = '{} < '.format(feature_lower_limit[k]) if k in feature_lower_limit else ''
            c2 = '"{}"'.format(k)
            c3 = ' <= {}'.format(feature_upper_limit[k]) if k in feature_upper_limit else ''

            condition = '[{}{}{}]'.format(c1, c2, c3)
            dict_of_conditions[k] = condition

        list_of_conditions = []
        for f in feature_names:
            if f in dict_of_conditions:
                list_of_conditions.append(dict_of_conditions[f])

        return list_of_conditions

    def _fit_decision_tree(self, neiborhood, sample=None):
        x = neiborhood
        y = self.model.predict(x)

        delta = 1e100
        prev_delta = delta
        depth = self.explanation_depth
        while delta > self.allowed_delta:

            dt = self._create_dt(max_depth=depth)
            dt.fit(x, y)
            if self.is_classification:
                yt_proba = self.model.predict_proba(sample.reshape(1, -1))
                yp_proba = dt.predict_proba(sample.reshape(1, -1))
                delta = numpy.max(numpy.abs(yt_proba - yp_proba))
            else:
                yt = self.model.predict(sample.reshape(1, -1))
                yp = dt.predict(sample.reshape(1, -1))
                delta = 2 * numpy.abs(yt - yp) / (numpy.abs(yt + yp) + 1e-3)

            depth += 1
            if delta == prev_delta:
                raise ValueError("Failed training surrogate dt: dt does not converge correctly on given sample.\n"
                                 "Either increase allowed_delta (>{}) or decrease min_samples_leaf".format(delta))
            else:
                prev_delta = delta

        msg, depth = self._decision_path_to_text(decision_tree=dt, sample=sample,
                                                 feature_names=self.train_data_stats.feature_names)

        # check dt performance
        n = len(x) // 5
        train, valid = x[n:, :], x[:n, :]

        dt = self._create_dt(max_depth=depth)
        dt.fit(train, y[n:])

        if self.is_classification:
            logging.info('recall score=', Metrics.recall.function(y[:n], dt.predict(valid)))
            # logging.info("model prediction = {}, surrogate dt model = {}".format(self.model.predict_proba(sample.reshape(1,-1)), dt.predict_proba(sample.reshape(1,-1))))

        else:
            logging.info('r2 score=', Metrics.r2.function(y[:n], dt.predict(valid)))
            # logging.info("model prediction = {}, surrogate dt model = {}".format(self.model.predict(sample.reshape(1,-1)), dt.predict(sample.reshape(1,-1))))

        # train dt on entire data
        dt.fit(x, y)

        return dt

    def _fit_dt_on_neighborhood(self, sample):
        if self.cache is None or self.cache != str(sample):
            self.cache = str(sample)
            self.dt = self._fit_decision_tree(self._create_neighborhood(sample=sample, n_samples=self.n_samples,
                                                                        train_data_stats=self.train_data_stats),
                                              sample=sample)
        else:
            pass  # dt is already fitted

    def explain(self, sample: numpy.ndarray):
        self._fit_dt_on_neighborhood(sample=sample)
        msg, depth = self._decision_path_to_text(decision_tree=self.dt, sample=sample,
                                                 feature_names=self.train_data_stats.feature_names)
        return msg

    def plot(self, sample: numpy.ndarray):
        self._fit_dt_on_neighborhood(sample=sample)
        tree.plot_tree(self.dt, feature_names=self.train_data_stats.feature_names, fontsize=10, filled=True)

        try:
            from matplotlib import pyplot as plt

            msg = self.explain(sample)
            msg = msg.replace('and', 'and\n').replace('then', '\nthen')
            plt.text(.0, 1., msg,
                     transform=plt.gca().transAxes, size=12,
                     horizontalalignment='left',
                     verticalalignment='top')
        except:
            raise


if __name__ == '__main__':
    dataset = CaliforniaHousing()
    model = dataset.get_model()
    from matplotlib import pyplot as plt

    plt.rcParams['figure.figsize'] = (20, 10)

    train = dataset.as_dmd()[0]
    sample = dataset.testing_data[0][3, :]
    explainer = DecisionTreeExplainer()
    explainer.fit(train, model)
    logging.info('\n'.join([train.feature_names[icol] + ' : ' + str(sample[icol]) for icol in range(train.n_features)]))

    logging.info('\n\nexplain')
    explainer.explain(sample)
    logging.info('\n\nplot')
    explainer.plot(sample)

    plt.show()
