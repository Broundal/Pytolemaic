import numpy

from pytolemaic import PyTrust
from pytolemaic.analysis_logic.dataset_analysis.anomaly_values_in_data import AnomalyValuesDetector
from pytolemaic.analysis_logic.dataset_analysis.anomaly_values_in_data_report import AnomaliesInDataReport


def run(fast=False):
    from resources.datasets.uci_adult import UCIAdult

    dmd_train, dmd_test = UCIAdult().as_dmd()
    print(dmd_train.feature_names, '\n')

    # from pytolemaic.analysis_logic.model_analysis.scoring.scoring_report import ConfusionMatrixReport
    # labels = [dmd_train.categorical_encoding_by_icols[13][k] for k in sorted(dmd_train.categorical_encoding_by_icols[13].keys())]
    # y = dmd_train.values[:,13]
    # y = y[numpy.isfinite(y)]
    # yp = y.copy()
    # u,c = numpy.unique(y, return_counts=True)
    # for u_, c_ in zip(u,c):
    #     if c_<100:
    #         y[y==u_] = 0
    #     if c_<75:
    #         yp[yp == u_] = 0
    # cm = ConfusionMatrixReport(y_true=y, y_pred=yp, labels=labels)
    # cm.plot()
    # plt.show()

    pytrust = PyTrust(model=UCIAdult().model,
                      xtrain=dmd_train,
                      xtest=dmd_test)
    if fast:
        report = pytrust.create_anomalies_in_data_report(train=pytrust.train,
                                                         test=pytrust.test,
                                                         features_to_analyze=['education-num','marital-status','sex'],
                                                         reg_method='std', # reg_method='binning'
                                                         max_samples=2000)
        # ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
        #  'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    else:
        report = pytrust.anomalies_in_data_report

    print("\n".join(report.insights(n_top_features=3)))

    report.plot(plot_only_above_threshold=False)



if __name__ == '__main__':
    from matplotlib import pyplot as plt

    run(fast=False)
    plt.show()
