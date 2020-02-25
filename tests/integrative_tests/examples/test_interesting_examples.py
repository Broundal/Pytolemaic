import unittest

from examples.interesting_examples import adult_dataset, california_housing_dataset, kddcup99_dataset, \
    prediction_uncertainty_adult_dataset, simple_example_classification, simple_example_regression


class TestInterestingExamples(unittest.TestCase):

    def test_simple_example_cla(self):
        simple_example_classification.run()

    def test_simple_example_reg(self):
        simple_example_regression.run()

    # @unittest.skip("adult_dataset example - Takes time")
    def test_adult_dataset(self):
        adult_dataset.run()

    def test_adult_dataset_uncertainty(self):
        prediction_uncertainty_adult_dataset.run()

    # @unittest.skip("california_housing_dataset example - Takes time")
    def test_california_housing(self):
        california_housing_dataset.run()

    # @unittest.skip("kddcup99 example - Takes time")
    def test_kddcup99(self):
        kddcup99_dataset.run()
