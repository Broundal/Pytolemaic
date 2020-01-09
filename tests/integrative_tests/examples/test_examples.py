import unittest

from examples.interesting_examples import adult_dataset, california_housing_dataset, kddcup99_dataset
from examples.toy_examples import example_quality_report, example_scoring_report, example_sensitivity_analysis, \
    example_prediction_uncertainty, example_prediction_explanation


class TestExamples(unittest.TestCase):

    def test_example_scoring_report(self):
        example_scoring_report.run()

    def test_example_sensitivity_analysis(self):
        example_sensitivity_analysis.run()

    def test_example_prediction_uncertainty(self):
        example_prediction_uncertainty.run()

    def test_example_quality_report(self):
        example_quality_report.run()

    def test_example_prediction_explanation(self):
        example_prediction_explanation.run()

    @unittest.skip("adult_dataset example - Takes time")
    def test_adult_dataset(self):
        adult_dataset.run()

    @unittest.skip("california_housing_dataset example - Takes time")
    def test_california_housing(self):
        california_housing_dataset.run()

    @unittest.skip("kddcup99 example - Takes time")
    def test_kddcup99(self):
        kddcup99_dataset.run()
