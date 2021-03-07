import logging

import numpy

from pytolemaic.analysis_logic.dataset_analysis.dataset_analysis import DatasetAnalysis, CovarianceShift
from pytolemaic.analysis_logic.dataset_analysis.dataset_analysis_report import DatasetAnalysisReport
from pytolemaic.analysis_logic.model_analysis.scoring.scoring import \
    Scoring
from pytolemaic.analysis_logic.model_analysis.scoring.scoring_report import ScoringFullReport
from pytolemaic.analysis_logic.model_analysis.sensitivity.sensitivity import \
    SensitivityAnalysis
from pytolemaic.analysis_logic.model_analysis.sensitivity.sensitivity_reports import SensitivityFullReport
from pytolemaic.analysis_logic.prediction_analysis.lime_report import LimeExplainer
from pytolemaic.analysis_logic.quality_report import TestSetQualityReport, TrainSetQualityReport, QualityReport, \
    ModelQualityReport
from pytolemaic.prediction_uncertainty.uncertainty_model import \
    UncertaintyModelClassifier, UncertaintyModelRegressor, UncertaintyModelBase
from pytolemaic.pytrust_report import PyTrustReport
from pytolemaic.utils.constants import CLASSIFICATION, REGRESSION, FeatureTypes
from pytolemaic.utils.dmd import DMD, ShuffleSplitter, StratifiedSplitter
from pytolemaic.utils.general import GeneralUtils
from pytolemaic.utils.metrics import Metrics, Metric


def cache(func):
    def cache_wrapper(self, *args, **kwargs):
        if not hasattr(self, '_cache'):
            self._cache = {}

        if func.__name__ not in self._cache:
            self._cache[func.__name__] = func(self, *args, **kwargs)

        return self._cache[func.__name__]

    return cache_wrapper


class PyTrust():
    numerical = FeatureTypes.numerical
    categorical = FeatureTypes.categorical

    def __init__(self, model,
                 xtrain=None, ytrain=None,
                 sample_meta_train: dict = None,

                 xtest=None, ytest=None,
                 sample_meta_test: dict = None,

                 columns_meta: dict = None,

                 feature_names: list = None,
                 feature_types: list = None,
                 categorical_encoding: dict = None,
                 metric: [str, Metric] = None,
                 splitter: str = 'shuffled',
                 target_labels: dict = None):
        """

        :param model: Model trained on training data provided

        :param xtrain: X training data. if DMD is provided, ytrain and any additional metadata is ignored.
        :param ytrain: Y training data.
        :param sample_meta_train: generic way to provide meta information on each sample in train data (e.g. sample weight) {key : [list of values]}.

        :param xtest: X test data. if DMD is provided, ytest and any additional metadata is ignored..
        :param ytest: Y test data. if DMD is provided,
        :param sample_meta_test: generic way to provide meta information on each sample in test data (e.g. sample weight) {key : [list of values]}.

        :param columns_meta: generic way to provide meta information on each feature (e.g. feature name) {key : [list of values]}.
        :param feature_names: feature name for each feature
        :param feature_types: feature type for each feature: NUMERICAL or CATEGORICAL
        :param categorical_encoding: For each column of categorical feature type, provide a dictionary of the structure
        {index: class name}. This information will allow providing more readable reports.

        :param metric: Target metric
        :param splitter: Splitter
        :param target_labels: categorical encoding for target variable in the format of {index: class name}.
        """
        self.model = model

        if isinstance(splitter, str):
            if splitter == 'shuffled':
                splitter = ShuffleSplitter
            elif splitter == 'stratified':
                splitter = StratifiedSplitter
            else:
                raise NotImplementedError("splitter='{}' is not supported".format(splitter))
        else:
            if not hasattr(splitter, 'split'):
                raise ValueError("splitter='{}' does not supported split() operation".format(splitter))
            else:
                raise NotImplementedError("splitter='{}' is not supported".format(splitter))

        if ytrain is not None:
            shape = getattr(ytrain, 'shape', (1, 1))
            if len(shape) == 2 and shape[1] > 1:
                raise NotImplementedError("Pytrust does not support multilabel (ytrain.shape[1]>1) analysis. "
                                          "In order to use Pytolemaic package, please wrap you model so model.predict(X) will return a single vector. ")
        if ytest is not None:
            shape = getattr(ytest, 'shape', (1, 1))
            if len(shape) == 2 and shape[1] > 1:
                raise NotImplementedError("Pytrust does not support multilabel (ytest.shape[1]>1) analysis. "
                                          "In order to use Pytolemaic package, please wrap you model so model.predict(X) will return a single vector. ")

        self.train = xtrain
        if self.train is not None and not isinstance(self.train, DMD):
            self.train = DMD(x=xtrain, y=ytrain,
                             samples_meta=sample_meta_train,
                             columns_meta=columns_meta,
                             feature_names=feature_names,
                             feature_types=feature_types,
                             categorical_encoding=categorical_encoding,
                             splitter=splitter,
                             target_labels=target_labels)

        self.test = xtest
        if self.test is not None and not isinstance(self.test, DMD):
            self.test = DMD(x=xtest, y=ytest,
                            samples_meta=sample_meta_test,
                            columns_meta=columns_meta,
                            feature_names=feature_names,
                            feature_types=feature_types,
                            categorical_encoding=categorical_encoding,
                            splitter=splitter,
                            target_labels=target_labels)

        if metric is None:
            if GeneralUtils.is_classification(model):
                metric = Metrics.recall
            else:
                metric = Metrics.mae

        self.metric = metric.name if isinstance(metric, Metric) else metric

        # todo
        self._validate_input()

        self.sensitivity = SensitivityAnalysis()

        self._uncertainty_models = {}
        self.covariance_shift = None
        self._cache = {}

    def _validate_input(self):
        if not hasattr(self.model, 'predict'):
            raise ValueError("Model must support predict() function")

    # region report classmethods
    @classmethod
    def create_sensitivity_report(cls, model, train: DMD, test: DMD, metric: str,
                                  sensitivity: SensitivityAnalysis = None, **kwargs) -> SensitivityFullReport:
        """
        Create sensitivity report
        Args:
            model - model to be analyzed based on test data
            train - train data (DMD). Train data is used if test is None
            test - test data (DMD). If None, use train data instead.
            sensitivity - SensitivityAnalysis instance. If None, instance is initiated internally.

        Returns:
            sensitivity report
        """
        sensitivity = sensitivity or SensitivityAnalysis()
        sensitivity.calculate_sensitivity(
            model=model, dmd_test=test, dmd_train=train, metric=metric, **kwargs)

        return sensitivity.sensitivity_report(**kwargs)

    @classmethod
    def create_scoring_report(cls, model, test: DMD, metric: str, y_pred=None, y_proba=None,
                              scoring: Scoring = None, **kwargs) -> ScoringFullReport:
        """
        Create scoring report
        Args:
            model - model to be analyzed based on test data
            train - train data (DMD). Train data is used only for separability test
            test - test data (DMD). .
            scoring - Scoring instance. If None, instance is initiated internally.

        Returns:
            scoring report
        """
        metrics = Metrics.supported_metrics()

        scoring = scoring or Scoring(metrics=metrics)

        score_values_report, confusion_matrix, scatter, classification_report = \
            scoring.score_value_report(model=model,
                                       dmd_test=test,
                                       labels=test.labels,
                                       y_pred=y_pred,
                                       y_proba=y_proba)

        return ScoringFullReport(target_metric=metric,
                                 metric_reports=score_values_report,
                                 confusion_matrix=confusion_matrix,
                                 scatter=scatter,
                                 classification_report=classification_report)

    @classmethod
    def create_quality_report(cls, scoring_report: ScoringFullReport,
                              sensitivity_report: SensitivityFullReport,
                              dataset_analysis_report: DatasetAnalysisReport) -> QualityReport:
        """
        Create quality report by analyzing scoring_report & sensitivity_report

        Returns:
            quality report
        """
        test_set_report = TestSetQualityReport(scoring_report=scoring_report,
                                               dataset_analysis_report=dataset_analysis_report)

        train_set_report = TrainSetQualityReport(vulnerability_report=sensitivity_report.vulnerability_report)
        model_quality_report = ModelQualityReport(vulnerability_report=sensitivity_report.vulnerability_report,
                                                  scoring_report=scoring_report)

        return QualityReport(train_quality_report=train_set_report, test_quality_report=test_set_report,
                             model_quality_report=model_quality_report)

    @classmethod
    def create_dataset_analysis_report(cls, train: DMD, is_classification, test:DMD=None, dataset_analysis: DatasetAnalysis=None, **kwargs) -> DatasetAnalysisReport:
        """
        Create dataset analysis report by analyzing train data

        Args:
            train - train data
            test - test data, if available
            is_classification - whether the target is categorical.

        Returns:
            dataset analysis report
        """
        dataset_analysis = dataset_analysis or DatasetAnalysis(problem_Type=CLASSIFICATION if is_classification else REGRESSION)
        report = dataset_analysis.dataset_analysis_report(train=train, test=test)
        return report

    @classmethod
    def create_pytrust_report(cls, pytrust) -> PyTrustReport:
        """
             Creates final report encompassing all other reports for entire pytrust.
             Returns:
                 pytrust report
        """
        return PyTrustReport(pytrust=pytrust)

    @classmethod
    def create_pytrust_for_separability_test(cls, train: DMD, test: DMD, covariance_shift=None, **pytrust_kwargs):
        """
        Create scoring report
        Args:
            train - train data (DMD).
            test - test data (DMD).

        Returns:
            separability pytrust
        """

        covariance_shift = covariance_shift or CovarianceShift()
        covariance_shift.calc_covariance_shift(dmd_train=train, dmd_test=test)

        separability_pytrust = PyTrust(model=covariance_shift.classifier,
                                       xtrain=covariance_shift.train,
                                       xtest=covariance_shift.test,
                                       **pytrust_kwargs)
        return separability_pytrust

    # endregion

    # region reports
    def _create_sensitivity_report(self) -> SensitivityFullReport:
        return self.create_sensitivity_report(model=self.model, train=self.train, test=self.test, metric=self.metric,
                                              sensitivity=self.sensitivity)

    def _create_scoring_report(self) -> ScoringFullReport:
        metrics = Metrics.supported_metrics()

        self.scoring = Scoring(metrics=metrics)

        return self.create_scoring_report(model=self.model, train=self.train, test=self.test, metric=self.metric,
                                          y_pred=self.y_pred_test, y_proba=self.y_proba_test, scoring=self.scoring)

    def _create_quality_report(self) -> QualityReport:
        return self.create_quality_report(scoring_report=self.scoring_report,
                                          sensitivity_report=self.sensitivity_report,
                                          dataset_analysis_report=self.dataset_analysis_report)

    def _create_dataset_analysis_report(self, **kwargs) -> DatasetAnalysisReport:
        self.dataset_analysis = DatasetAnalysis(problem_Type=CLASSIFICATION if self.is_classification else REGRESSION)
        return self.create_dataset_analysis_report(train=self.train, test=self.test, is_classification=self.is_classification,
                                                   dataset_analysis=self.dataset_analysis)

    def _create_pytrust_report(self):
        return self.create_pytrust_report(pytrust=self)

    def _create_pytrust_for_separability_test(self):
        self.covariance_shift = CovarianceShift()
        return self.create_pytrust_for_separability_test(
            train=self.train, test=self.test, metric=self.metric, covariance_shift=self.covariance_shift)

    # endregion

    # region on predict

    def create_uncertainty_model(self, method='default') -> UncertaintyModelBase:
        if method not in self._uncertainty_models:

            if self.is_classification:
                uncertainty_model = UncertaintyModelClassifier(
                    model=self.model, uncertainty_method=method)
            else:
                uncertainty_model = UncertaintyModelRegressor(
                    model=self.model, uncertainty_method=method)

            uncertainty_model.fit(dmd_test=self.test)
            self._uncertainty_models[method] = uncertainty_model

        return self._uncertainty_models[method]

    @cache
    def create_lime_explainer(self, **kwargs):
        lime_explainer = LimeExplainer(n_features_to_plot=20, **kwargs)
        lime_explainer.fit(self.train, model=self.model)
        return lime_explainer

    # endregion

    # region properties

    @property
    @cache
    def is_classification(self):
        return GeneralUtils.is_classification(self.model)

    @property
    @cache
    def model_support_dmd(self):
        return GeneralUtils.dmd_supported(self.model, self.test)

    @property
    @cache
    def y_pred_test(self):

        test = self.test if self.model_support_dmd else self.test.values
        if self.y_proba_test is not None:  # save some time
            y_pred_test = numpy.argmax(self.y_proba_test, axis=1)
        else:
            y_pred_test = self.model.predict(test)
        return y_pred_test

    @property
    @cache
    def y_proba_test(self):
        if not self.is_classification:
            return None

        test = self.test if self.model_support_dmd else self.test.values
        y_proba_test = self.model.predict_proba(test)
        return y_proba_test

    @property
    @cache
    def sensitivity_report(self)-> SensitivityFullReport:
        return self._create_sensitivity_report()

    @property
    @cache
    def scoring_report(self)-> ScoringFullReport:
        return self._create_scoring_report()

    @property
    @cache
    def quality_report(self)-> QualityReport:
        return self._create_quality_report()

    @property
    @cache
    def dataset_analysis_report(self)->DatasetAnalysisReport:
        return self._create_dataset_analysis_report()

    @property
    @cache
    def report(self) -> PyTrustReport:
        return self._create_pytrust_report()

    @property
    @cache
    def insights(self) -> list:
        return self.report.insights()

    @property
    @cache
    def pytrust_for_separability_test(self):
        return self._create_pytrust_for_separability_test()

    @classmethod
    def print_initialization_example(cls):
        example = \
            """
        # example for initializing PyTrust object. In this example, All fields other than model, xtrain, ytrain are optional.
        PyTrust(model=model,
                xtrain=X_train,
                ytrain=y_train,
                xtest=X_test,
                ytest=Y_test,
                feature_names=['Feature_#{}'.format(i) for i in range(X_train.shape[1])],
                feature_types=[PyTrust.categorical for i in range(X_train.shape[1])],
                target_labels={0:'dog', 1:'cat'}, # for classification
                categorical_encoding=[{0:'blue', 1:'red', 2:'yellow'} for i in range(X_train.shape[1])]                                
                )
                
        # See usage example below:\n
        """
        logging.info(example)
        example += cls.print_usage_example()
        return example

    @classmethod
    def print_usage_example(cls, report_types=('report',), plot=True, insights=True, to_dict=True,
                            to_dict_meaning=True):
        def plot_usage(prefix):
            return "# Plotting relevant graphs:\n" \
                   "{}.plot()\n" \
                   "plt.show()".format(prefix)

        def insights_usage(prefix):
            return "# Printing useful insights & hints. Short and to the point.\n" \
                   "pprint({}.insights())".format(prefix)

        def to_dict_usage(prefix):
            return "# Prints all information gathered, either full (default) or in readable form (printable=True)\n" \
                   "pprint({}.to_dict())\n" \
                   "or\n" \
                   "pprint({}.to_dict(printable=True))".format(prefix, prefix)

        def to_dict_meaning_usage(prefix):
            return "# Prints explanation for every field within the dict returned by to_dict()\n" \
                   "pprint({}.to_dict_meaning())".format(prefix)

        def add_components_msgs(prefix, plot=True, insights=True, to_dict=True, to_dict_meaning=True):
            msgs = []
            if plot:
                msgs.append(plot_usage(prefix=prefix))

            if insights:
                msgs.append(insights_usage(prefix=prefix))

            if to_dict:
                msgs.append(to_dict_usage(prefix=prefix))

            if to_dict_meaning:
                msgs.append(to_dict_meaning_usage(prefix=prefix))

            return "\n\n".join(msgs)

        example = []
        for report in report_types:
            if report not in ['report']:
                raise NotImplementedError(report)

            example.append('\n*****  pytrust.{}  ******\n'.format(report))

            if report == 'report':
                msg = add_components_msgs(prefix="pytrust.report",
                                          plot=plot, insights=insights,
                                          to_dict=to_dict, to_dict_meaning=to_dict_meaning)
                if msg:
                    example.append(msg)

            example.append('\n**************\n')

        example = "\n".join(example)
        logging.info(example)
        return example


    # endregion
