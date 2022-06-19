from typing import List

from matplotlib import pyplot as plt
import numpy
from xgboost import XGBRegressor

from pytolemaic import FeatureTypes
from pytolemaic.utils.dmd import DMD
from pytolemaic.utils.general import get_logger
from pytolemaic.utils.metrics import Metrics, Metric
from sklearn.preprocessing import LabelEncoder
from pytolemaic.analysis_logic.dataset_analysis.anomaly_values_in_data_report import AnomaliesInDataReport

logger = get_logger(__name__)

class AnomalyValuesDetector():

    def __init__(self, use_target=True,
                 reg_metric='r2', clas_metric='recall',
                 clas_threshold=0.95,
                 reg_method = 'std',
                 contamination= 0.02,
                 n_stds = 3,
                 range_ratio = 0.05,
                 **kwargs):
        """
        reg_metric : metric to assess regression quality - no effect on anomalies analysis
        clas_metric : metric to assess classification quality - no effect on anomalies analysis
        clas_threshold : Threshold to determine level on confidence requried for anomaly score
        reg_method: 'std' or 'binning'. 'std': anomaly is measured by distance from mean; 'binning':
                     binning is performed and then anomaly is measured as mis-classification
        contamination: percentage of samples to be considered as anomalies - determine threshold but not anomaly score.
        n_std: if reg_method='std', set threshold distance from mean > # of stds - determine threshold but not anomaly score.
        range_ratio: if reg_method='std' anomaly score will be resetted if distance/range < range_ratio.
        """

        self.use_target = use_target
        self.models = {}

        self.reg_metric = reg_metric
        self.clas_metric = clas_metric
        self.clas_threshold = clas_threshold
        self.reg_method = reg_method
        self.contamination = contamination
        self.n_stds = n_stds
        self.range_ratio = range_ratio # to discard small anomalies when std is almost 0


    def _get_metric(self, metric_name)->Metric:
        metric = Metrics.supported_metrics().get(metric_name, None)
        if metric is None:
            raise NotImplementedError("metric {} is no implemented".format(metric_name))

        return metric

    def get_model(self, is_classification):
        from xgboost import XGBRegressor, XGBClassifier
        kwargs = dict(random_state=0,
                       n_estimators=250,
                       n_jobs=-1,
                       eval_metric='merror',
                       use_label_encoder=False)
        model = XGBClassifier(**kwargs) if is_classification else XGBRegressor(**kwargs)

        return model

    def analyze(self, dmd_train: DMD, features_to_analyze:List[str]=None, perform_extra_iteration=False,
                max_samples=None):

        if dmd_train.feature_types is None:
            raise ValueError("Analysis for anomalous values in dataset requires feature types information.")

        inds1, inds2, train1, train2 = self.split_train_into_2_parts(dmd_train)

        if features_to_analyze:
            features_subset_ = [i for i in range(dmd_train.n_features)
                                   if dmd_train.feature_names[i] in features_to_analyze]
            if len(features_subset_)==0:
                raise KeyError("{} not recognized".format(features_to_analyze))
            else:
                features_to_analyze = features_subset_
        else:
            features_to_analyze = range(dmd_train.n_features)

        reports = {}
        for ifeature in features_to_analyze:
            name = dmd_train.feature_names[ifeature]
            logger.info("\n\nAnalyzing feature #{}: {}".format(ifeature, name))

            report1, report2 = self.calculate_anomaly_scores(
                train1=train1, train2=train2, ifeature=ifeature,
                feature_types=dmd_train.feature_types, perform_extra_iteration=perform_extra_iteration,
                max_samples=max_samples)

            for report in [report1, report2]:
                report['anomaly_score_ratio'] = report['anomaly_score']/report['threshold']


            report = {}
            for k in report1:
                v1 = report1[k]
                v2 = report2[k]
                if isinstance(v1, list):
                    v1 = numpy.array(v1)
                    v2 = numpy.array(v2)
                if isinstance(v1, numpy.ndarray):
                    if len(v1.shape)==1:
                        v = numpy.zeros(dmd_train.n_samples, dtype=v1.dtype) #1D
                    else:
                        v = numpy.zeros((dmd_train.n_samples, v1.shape[1]), dtype=v1.dtype) #2D
                    v[inds1] = v1
                    v[inds2] = v2
                    report[k] = v
                elif v1 == v2:
                    report[k] = v1
                else:
                    report[k] = (v1+v2)/2

            v = report['anomaly_score_ratio']
            logger.info("Found {} samples with anomaly score >= {:.3g}, and {} (out of {}) samples with anomaly score > 0".format(
                sum(v>=1), report['threshold'], sum(v > 0), len(v)))

            reports[name] = report
        return reports
        #
        #     anomaly_score_raw[inds1, ifeature] = report1['anomaly_score']
        #     anomaly_score_raw[inds2, ifeature] = report2['anomaly_score']
        #
        #     anomaly_score_threshold[inds1, ifeature] = report1['anomaly_score']/report1['threshold']
        #     anomaly_score_threshold[inds2, ifeature] = report2['anomaly_score']/report2['threshold']
        #
        #     yt[inds1, ifeature] = report1['yt']
        #     yt[inds2, ifeature] = report2['yt']
        #
        #     yp[inds1, ifeature] = report1['yp']
        #     yp[inds2, ifeature] = report2['yp']
        #
        #
        #     v = anomaly_score_threshold[:, ifeature]
        #     logger.info("Found {} samples with anomaly score >= {:.3g}, and {} (out of {}) samples with anomaly score > 0".format(
        #         sum(v>=1), min(report1['threshold'], report2['threshold']), sum(v > 0), len(v)))
        #
        #     if False:
        #         sorted_inds[name] = sorted(numpy.arange(len(v)), key=lambda j: v[j], reverse=True)
        #         v = anomaly_score_raw[:,ifeature][sorted_inds[name]]
        #         plt.plot(v, '.')
        #         label = "Threshold for contamination = {}%".format(self.contamination*100) \
        #             if self.contamination else "Threshold for {} Sigma".format(self.n_stds)
        #         plt.plot([0,len(v)],[threshold1, threshold1], '-k', label=label)
        #         plt.plot([0,len(v)],[threshold2, threshold2], '-k')
        #         plt.show()
        #
        # if True:
        #     inds = numpy.arange(len(anomaly_score_threshold))
        #     sm = numpy.sum(anomaly_score_threshold>=1,axis=1)
        #     inds = sorted(inds, key=lambda k:sm[k])
        #     for ifeature in range(0, dmd_train.n_features):
        #         v = anomaly_score_threshold[inds, ifeature].copy()
        #         v[v<1]=0
        #
        #
        #         logger.info(ifeature, dmd_train.feature_names[ifeature], sum(v), sum(v)/len(v))
        #         plt.plot(v, '-', label='{}'.format(dmd_train.feature_names[ifeature]))
        #     plt.legend()
        #     plt.show()
        # if True:
        #     for ifeature in range(0, dmd_train.n_features):
        #         v = anomaly_score_threshold[:, ifeature].copy()
        #
        #         if numpy.max(v)<1:
        #             continue
        #
        #         fig, axs = plt.subplots(2,1, sharex=True)
        #         inds = numpy.arange(len(v))[v>=1]
        #         axs[0].plot(inds,anomaly_score_raw[v>=1, ifeature],'.', label=dmd_train.feature_names[ifeature])
        #         axs[1].plot(inds,yt[v>=1, ifeature],'.b', label='Data')
        #         axs[1].plot(inds,yp[v>=1, ifeature],'.r', label='Expected')
        #         plt.legend()
        #         plt.show()
        #
        # return anomaly_score_raw

    def calculate_anomaly_scores(self, train1, train2, feature_types, ifeature, perform_extra_iteration, max_samples=None):
        logger.info("Analysis started")

        report2 = self._analyze_one_way(ifeature, train=train1, validation=train2,
                                        feature_types=feature_types, metric_score_only=perform_extra_iteration,
                                        max_samples=max_samples)
        report1 = self._analyze_one_way(ifeature, train=train2, validation=train1,
                                        feature_types=feature_types,
                                        max_samples=max_samples)
        metric = report1['metric']
        logger.info("P1 train, P2 test ({}): {:.3g}".format(metric, report2['metric_score']))
        logger.info("P2 train, P1 test ({}): {:.3g}".format(metric, report1['metric_score']))

        if perform_extra_iteration: # clean the train data from anomalies and re-iterate
            # 1st iteration
            inds_to_keep1 = report1['anomaly_score'] < report1['threshold']

            report2 = self._analyze_one_way(ifeature=ifeature, train=train1[inds_to_keep1, :], validation=train2,
                                            feature_types=feature_types,
                                            max_samples=max_samples)

            # 2nd iteration
            inds_to_keep2 = report2['anomaly_score'] < report2['threshold']

            report1 = self._analyze_one_way(ifeature=ifeature, train=train2[inds_to_keep2, :], validation=train1,
                                            feature_types=feature_types,
                                            max_samples=max_samples)

            logger.info("Cleaned P1 train, P2 test ({}): {:.3g}".format(metric, report2['metric_score']))
            logger.info("Cleaned P2 train, P1 test ({}): {:.3g}".format(metric, report1['metric_score']))
        return report1, report2

    def split_train_into_2_parts(self, dmd_train):
        rs = numpy.random.RandomState(0)
        inds = rs.permutation(dmd_train.n_samples)
        inds1, inds2 = inds[:len(inds) // 2], inds[len(inds) // 2:]
        train1, train2 = dmd_train.split_by_indices(inds1), dmd_train.split_by_indices(inds2)
        if self.use_target and dmd_train.target is not None:
            train1 = numpy.concatenate([train1.values, train1.target], axis=1)
            train2 = numpy.concatenate([train2.values, train2.target], axis=1)
        else:
            train1 = train1.values
            train2 = train2.values
        train1 = train1.astype(float)
        train2 = train2.astype(float)
        return inds1, inds2, train1, train2

    def _get_xy(self, data: numpy.ndarray, itarget, drop_na):
        features_wo_i = list(range(data.shape[1]))
        features_wo_i.remove(itarget)

        if drop_na:
            isfinite = numpy.isfinite(data[:, itarget])
            data = data[isfinite, :]

        x = data[:, features_wo_i].copy()
        y = data[:, itarget].copy()

        return x, y

    def _encode_target(self, target, is_classification, le: LabelEncoder=None, bins=None):

        if not is_classification and self.reg_method=='binning':
            if bins is None:
                bins = self.reg_binning(target)

            target = numpy.digitize(numpy.clip(target, bins[0], bins[-1]),
                                    bins=bins)

        if is_classification or self.reg_method == 'binning':
            target = target.astype(int)
            if le is None:
                le = LabelEncoder()
                le.fit(target)
            try:
                target = le.transform(target).astype(int)
            except:
                target = numpy.array([le.transform([y_]).astype(int)[0]
                                  if y_ in le.classes_ and not numpy.isnan(y_)
                                  else numpy.nan
                                  for y_ in target])

        return target, bins, le

    def reg_binning(self, y, max_bins=20):
        u = numpy.unique(y)
        if numpy.nanmax(u-u.astype(int))==0 and len(u)<=max_bins:
            bins = u
        else:
            bins = numpy.histogram_bin_edges(y, bins='auto')  # todo: fix range

        y_ = numpy.digitize(y, bins=bins)
        u, c = numpy.unique(y_, return_counts=True)
        if len(bins) != max(y_) or len(bins) > max_bins or min(c) < 20:
            # empty bins
            logger.debug("len(bin)=={}, max(y)=={}".format(len(bins), max(y)))

            bins = numpy.percentile(y, numpy.arange(max_bins) * 100/max_bins)
            bins = numpy.histogram_bin_edges(y, bins=bins)
            bins = numpy.unique(bins)
            logger.debug("len(bin)=={}, max(y)=={}".format(len(bins), max(y)))
        return bins


    def _calculate_score(self, yv, yp, is_classification):
        metric_name = self.clas_metric if is_classification else self.reg_metric

        metric = self._get_metric(metric_name=metric_name)
        if metric.is_proba:
            raise NotImplementedError("Proba metrics are not supported yet")

        isfinite = numpy.isfinite(yv)
        score = metric.function(yv[isfinite], yp[isfinite])
        score = Metrics.metric_as_score(score, metric)
        return score

    # def _analyze_yproba_highpass(self, yv, yp, yproba):
    #     max_yproba = numpy.max(yproba, axis=1)
    #     # do not filter yv==nan to maintain correct indexing
    #     isfinite = numpy.isfinite(yv)
    #
    #     p_out = None
    #     for j in reversed(range(10)):  # p: 0.7 --> 0.97
    #         p = 1 - (j + 1) / 30
    #         inds = max_yproba >= p
    #         correct = yv[inds] == yp[inds]  # yv[j]==nan --> correct[i]=False
    #         if numpy.sum(correct) / numpy.sum(inds & isfinite) > self.clas_threshold:
    #             p_out = p
    #             break
    #
    #     if p_out is None:
    #         logger.info("highpass:clas_threshold {} not met".format(self.clas_threshold))
    #         return []
    #     else:
    #
    #         high_pass = numpy.arange(len(yv))[max_yproba > p_out]
    #         out = [j for j in high_pass if yp[j] != yv[j] and isfinite[j]]
    #         logger.info("highpass: clas_threshold with threshold {:.3g} - len(out)={}".format(p_out, len(out)))
    #
    #         return out

    def _analyze_yproba(self, yv, yproba, key='low'):
        """
        key=='low': model is very confident that the correct answer is incorrect:
            # locate candidates which are incorrect (candidates != max_yproba)
            # and the model is certain they are incorrect (candidates <= p_out)
            # for these samples, anomaly score is 1 - candidates
        key=='high': model is very confident in a wrong answer is correct:
                # locate candidates which are incorrect (candidates != max_yproba)
                # and the model is certain that some other class is correct (max_yproba >= p_out)
                # for these samples, anomaly score is candidates

        * candidates == y_proba for the expected class
        """
        isfinite = numpy.isfinite(yv)

        candidates = numpy.zeros_like(yv, dtype=float)
        candidates[~isfinite] = numpy.nan
        # yproba[i, int(yv[i])] is the probability for the correct class for sample 'i'
        candidates[isfinite] = [yproba[i, int(yv[i])] for i in numpy.arange(len(yv))[isfinite]]

        max_yproba = numpy.max(yproba, axis=1)

        # find a threshold p for which almost all candidates are > p
        p_out = None
        for j in reversed(range(11)): # Note order is reversed
            if key == 'low':
                p = j / 30 # gradually lower p
                correct = candidates > p  # candidate==nan --> cond=False. '>' because p can be 0
                denominator = numpy.sum((max_yproba>p) & isfinite) # how many samples has max_proba > p
            else:
                p = 1 - j/30 # gradually increase p
                correct = candidates >= p  # candidate==nan --> cond=False. '>=' because p can be 1
                denominator = numpy.sum((max_yproba>=p) & isfinite) # how many samples has max_proba >= p

            # logger.info(key, j, p, numpy.sum(correct), denominator, numpy.sum(correct) / denominator)

            # condition: at least some samples has max_proba > p, and enough candidates are correct:
            if denominator > 0 and numpy.sum(correct) / denominator > self.clas_threshold:
                p_out = p
                break
            else:
                continue

        out = numpy.zeros_like(candidates, dtype=float)
        if p_out is None:
            logger.debug("{}pass:clas_threshold {} not met".format(key, self.clas_threshold))
            return out
        else:
            if key=='low':
                # model is very confident (candidates <= p_out) that the expected class is not the correct answer (candidates != max_yproba)
                incorrect = (candidates <= p_out) & (candidates != max_yproba) & isfinite
            else:
                # model is very confident (max_yproba >= p_out) that the correct answer is not the expected class (candidates != max_yproba)
                incorrect = (max_yproba >= p_out) & (candidates != max_yproba) & isfinite

            out[incorrect] = 1-candidates[incorrect]
            return out

    def _analyze_one_way(self, ifeature, train: numpy.ndarray, validation: numpy.ndarray,
                         feature_types, max_samples=None, metric_score_only=False):

        """
        report_regression = \
            dict(yt=[],
                 yp=[],
                 n_stds = [],
                 yrange=None,
                 anomaly_score=[],
                 threshold=None,
                 metric_score=None)
        report_classification = \
            dict(yt=[],
                 yp=[],
                 probability_of_yt = [], # y_proba of correct class
                 probability_of_yp = [], # y_proba of highest proba
                 anomaly_score=[],
                 threshold=None,
                 metric_score=None)
        """

        is_classification = feature_types[ifeature] == FeatureTypes.categorical

        ## extract data. if reg is 'binning' convert to a classification task
        x, y = self._get_xy(data=train, itarget=ifeature, drop_na=True)
        xv, yv = self._get_xy(data=validation, itarget=ifeature, drop_na=False)

        yv_orig = yv.copy()

        if is_classification or self.reg_method == 'binning':
            if is_classification: # code to avoid 1 class in 1 set, but not in the other
                cond = numpy.in1d(y, yv)
                cond[~numpy.isfinite(y)] = True # ignore results for nans
                x = x[cond,:] # drop all instances which are not in yv
                y = y[cond] # drop all instances which are not in yv

                cond = numpy.in1d(yv, y)
                cond[~numpy.isfinite(yv)] = True  # ignore results for nans
                yv[~cond]=numpy.nan  # drop all instances which are not in y

            y, bins, le = self._encode_target(y, is_classification=is_classification, le=None, bins=None)
            yv, _, _ = self._encode_target(yv, is_classification=is_classification, le=le, bins=bins)
            is_classification = True
        else:
            le = None

        model = self.get_model(is_classification=is_classification) # e.g. XGBoost
        max_samples = max_samples or len(x)
        if max_samples < len(x):
            to_keep = numpy.arange(len(x)) < max_samples
            numpy.random.RandomState(0).shuffle(to_keep)

            if is_classification:
                u, inds = numpy.unique(y, return_index=True)
                to_keep[inds] = True # make sure all classes are represented to avoid bugs

            x = x[to_keep,:]
            y = y[to_keep]
        model.fit(x, y)

        yp = model.predict(xv)

        # average error (score) for entire validation set
        isfinite = numpy.isfinite(yv)
        metric = self.clas_metric if is_classification else self.reg_metric
        metric_score = self._calculate_score(yv[isfinite], yp[isfinite], is_classification=is_classification)
        if metric_score_only:
            return dict(metric_score=metric_score, metric=metric)

        if is_classification:
            yproba = model.predict_proba(xv)


            # logger.debug('highpass - certainty in incorrect answer')
            out_high = self._analyze_yproba(yv, yproba, key='high')
            # logger.debug('lowpass - certainty the correct answer is incorrect')
            out_low = self._analyze_yproba(yv, yproba, key='low')

            out_anomaly_score = numpy.max(
                numpy.concatenate([out_low.reshape(1,-1), out_high.reshape(1,-1)]),
                axis=0)

            # plt.plot(out_score,'.')
            # plt.show()

            threshold = numpy.percentile(out_anomaly_score, 100 * (1 - self.contamination))
            if threshold == numpy.min(out_anomaly_score):
                threshold = numpy.min(out_anomaly_score)+1
            threshold = max(threshold, 0.75)

            # yp[~numpy.isfinite(yv)] = numpy.nan
            # yt_int = yv.astype(int).astype(object)
            # yt_int[~numpy.isfinite(yv)] = numpy.nan
            report_classification = \
                dict(yt=yv_orig,
                     yp=le.inverse_transform(yp),
                     probability_of_yt=[yproba[i, int(yv[i])] if numpy.isfinite(yv[i]) else numpy.nan for i in numpy.arange(len(yv))],  # y_proba of correct class
                     probability_of_yp=yproba,  # y_proba of highest proba
                     anomaly_score=out_anomaly_score,
                     threshold=threshold,
                     metric_score=metric_score,
                     metric=metric)

            return report_classification #report_classification['metric_score'], report_classification['anomaly_score'], report_classification['threshold']
        else:
            scale = lambda d: (d - numpy.nanmean(d))/(numpy.nanstd(d)+1e-10)
            yrange = numpy.max(y) - numpy.min(y)


            deltas = yv - yp
            deltas_orig = deltas.copy()
            deltas_scaled = scale(deltas)

            if self.reg_method=='std':
                # trying to divide the space into subspaces with similar error.
                y_delta = deltas_scaled[isfinite]
                noise = 1 + 1e-3 * numpy.random.RandomState(0).rand(len(y_delta))
                y_delta *= noise

                x_ = xv[isfinite,:].copy()

                proximity_model = XGBRegressor(min_child_weight=100,
                                               n_estimators=1,
                                               # num_feature=xv.shape[1],
                                               subsample=1,
                                               colsample_bynode=1,
                                               # reg_lambda=0,
                                               tree_method='exact',
                                               max_depth=100,
                                               random_state=iter)
                # proximity_model = DecisionTreeRegressor(min_samples_leaf=100)
                proximity_model.fit(x_, y_delta)  # y=yv or y=delta?

                y_proximity = proximity_model.predict(xv)
                u,c = numpy.unique(y_proximity, return_counts=True)

                out_score = numpy.zeros(len(xv))
                n_stds = numpy.zeros(len(xv))*numpy.nan
                for leaf_ind, value in enumerate(u):
                    indices = numpy.argwhere((y_proximity==value) & isfinite).ravel()
                    if len(indices)==0:
                        continue

                    deltas_of_leaf_orig = deltas_orig[indices]

                    n_stds[indices] = numpy.abs(scale(deltas_of_leaf_orig))

                # score is 1-1/#std. e.g. std=2 --> score=0.5. std=3 --> score=0.66
                # high score --> high anomaly
                anomaly_score = 1 - 1 / (n_stds+1e-10)

                # points with low error are considered normal, even if out of distribution for leaf
                anomaly_score[numpy.abs(yp-yv) < self.range_ratio * yrange] = 0
                anomaly_score = numpy.clip(anomaly_score, 0, 1)

                if self.n_stds:
                    threshold = 1 - 1 / self.n_stds
                else:
                    threshold = numpy.percentile(anomaly_score, 100 * (1 - self.contamination))
                    if threshold == numpy.min(anomaly_score):
                        threshold = numpy.min(anomaly_score) + 1

                yp[~numpy.isfinite(yv)] = numpy.nan
                report_regression = \
                    dict(yt=yv_orig,
                         yp=yp,
                         n_stds=n_stds,
                         yrange=yrange,
                         anomaly_score=anomaly_score,
                         threshold=threshold,
                         metric_score=metric_score,
                         metric=metric)

                return report_regression #metric_score, out_score, threshold
            else:
                raise NotImplementedError(str(self.reg_method))


    def anomalies_in_data_report(self, train: DMD, test:DMD=None,
                                 features_to_analyze: List[str] = None,
                                 perform_extra_iteration=False,
                                 max_samples=None,
                                 **kwargs) -> AnomaliesInDataReport:

        if test is not None:
            train = DMD.concat([train, test])

        if not features_to_analyze:
            extra_msg = "This may take some time. Use features_to_analyze argument to analyze a subset of features."
            logger.info("Analyzing {} features for anomalous values. {}".format(train.n_features, extra_msg))
        else:
            logger.info("Analyzing {} features for anomalous values.".format(len(features_to_analyze)))

        analysis = self.analyze(dmd_train=train,
                                features_to_analyze=features_to_analyze,
                                perform_extra_iteration=perform_extra_iteration,
                                max_samples=max_samples)
        return AnomaliesInDataReport(
            anomalies_report = analysis,
            categorical_encoding_by_feature_name = train.categorical_encoding_by_feature_name)
