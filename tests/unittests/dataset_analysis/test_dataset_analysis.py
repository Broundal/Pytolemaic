import unittest
from pprint import pprint

import numpy

from pytolemaic.analysis_logic.dataset_analysis.dataset_analysis import DatasetAnalysis
from pytolemaic.utils.constants import REGRESSION, FeatureTypes, CLASSIFICATION
from pytolemaic.utils.dmd import DMD


class TestDatasetAnalysis(unittest.TestCase):

  def setUp(self) -> None:
    rs = numpy.random.RandomState(0)
    x = rs.randn(1000, 10)

    x[:, 0:5] = numpy.round(x[:, 0:5], 0)
    x[0, 0] = 10
    x[0, 9] = 10
    x[1, 9] = -4

    y = numpy.copy(x[:, 0])

    x[10, :] = numpy.nan
    x[:, 1] = numpy.nan
    x[:700, 2] = numpy.nan

    self.dataset = DMD(x=x, y=y, columns_meta={
      DMD.FEATURE_TYPES: 5 * [FeatureTypes.categorical] + 5 * [FeatureTypes.numerical]})

  def test_count_unique_classes(self):
    da = DatasetAnalysis(CLASSIFICATION, class_count_threshold=10, outliers_n_sigma=(3, 5),
                         nan_threshold_per_col=(0.1, 0.5, 0.9), nan_threshold_per_sample=(0.1, 0.5, 0.9))
    out = da.count_unique_classes(self.dataset)
    print("count_unique_classes:")
    pprint(out)

    self.assertTrue(0 in out)
    self.assertTrue(10.0 in out[0])
    self.assertTrue(out[0][10.0] == 1)

    self.assertTrue('target' in out)
    self.assertTrue(10.0 in out['target'])
    self.assertTrue(out['target'][10.0] == 1)

    for feature in out:
      for value in out[feature].values():
        self.assertLessEqual(value, da._class_count_threshold)

  def test_count_outliers(self):
    da = DatasetAnalysis(REGRESSION, class_count_threshold=10, outliers_n_sigma=(3, 5),
                         nan_threshold_per_col=(0.1, 0.5, 0.9), nan_threshold_per_sample=(0.1, 0.5, 0.9))
    out = da.count_outliers(self.dataset)
    print("count_outliers:")
    pprint(out)

    self.assertTrue(9 in out)
    self.assertTrue('3-sigma' in out[9])
    self.assertTrue('5-sigma' in out[9])
    self.assertTrue(out[9]['5-sigma']['n_outliers'] == 1)
    self.assertTrue(out[9]['3-sigma']['expected_outliers'] == 2)
    self.assertTrue(out[9]['5-sigma']['expected_outliers'] == 0)

    self.assertTrue('target' in out)
    self.assertTrue('3-sigma' not in out['target'])
    self.assertTrue('5-sigma' in out['target'])
    self.assertTrue(out['target']['5-sigma']['n_outliers'] == 1)

  def test_count_missing_values(self):
    da = DatasetAnalysis(REGRESSION, class_count_threshold=10, outliers_n_sigma=(3, 5),
                         nan_threshold_per_col=(0.1, 0.5, 0.9), nan_threshold_per_sample=(0.1, 0.9))
    nan_cols, nan_rows = da.count_missing_values(self.dataset)
    print("count_missing_values cols:")
    pprint(nan_cols)

    print("count_missing_values rows:")
    pprint(nan_rows)

    for th in da._nan_threshold_per_feature:
      self.assertTrue(th in nan_cols)

    for th in da._nan_threshold_per_sample:
      self.assertTrue(th in nan_rows)

    self.assertEqual(nan_cols, {0.1: {1: 1.0, 2: 0.7}, 0.5: {1: 1.0, 2: 0.7}, 0.9: {1: 1.0}})
    self.assertEqual(nan_rows[0.9], {10: 1.0})
    self.assertEqual(len(nan_rows[0.1]), self.dataset.n_samples)
    for th, vdict in nan_rows.items():
      print(th, vdict)
      self.assertTrue(th <= max(vdict.values()))

  def test_dataset_analysis_report(self):
    da = DatasetAnalysis(REGRESSION, class_count_threshold=10, outliers_n_sigma=(3, 5),
                         nan_threshold_per_col=(0.1, 0.5, 0.9), nan_threshold_per_sample=(0.1, 0.9))
    report = da.dataset_analysis_report(train=self.dataset)
    pprint(report.to_dict(printable=True))
    pprint(report.to_dict_meaning())
    pprint(report.plot())
    #
    # from matplotlib import pyplot as plt
    # plt.show()
