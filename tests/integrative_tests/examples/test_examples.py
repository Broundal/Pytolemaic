import unittest

from examples.toy_examples import example_quality_report, example_scoring_report, example_sensitivity_analysis, \
    example_prediction_uncertainty, example_prediction_explanation


class TestToyExamples(unittest.TestCase):

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
