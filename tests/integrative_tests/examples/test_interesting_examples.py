import unittest

from examples.interesting_examples import adult_example, california_housing_example, kddcup99_example, \
    prediction_uncertainty_adult_example, simple_example_classification, simple_example_regression, \
    prediction_uncertainty_california_example, titanic_example


class TestInterestingExamples(unittest.TestCase):

    def test_simple_example_cla(self):
        simple_example_classification.run()

    def test_simple_example_reg(self):
        simple_example_regression.run()

    # @unittest.skip("adult_dataset example - Takes time")
    def test_adult_dataset(self):
        adult_example.run(fast=True)

    def test_adult_dataset_uncertainty(self):
        prediction_uncertainty_adult_example.run()

    def test_california_dataset_uncertainty(self):
        prediction_uncertainty_california_example.run()

    # @unittest.skip("california_housing_dataset example - Takes time")
    def test_california_housing(self):
        california_housing_example.run(fast=True)

    # @unittest.skip("kddcup99 example - Takes time")
    def test_kddcup99(self):
        kddcup99_example.run(fast=True)

    def test_titanic(self):
        titanic_example.run(fast=True)
