import unittest

from examples.interesting_examples import adult_dataset, california_housing_dataset, kddcup99_dataset

class TestInterestingExamples(unittest.TestCase):

    # @unittest.skip("adult_dataset example - Takes time")
    def test_adult_dataset(self):
        adult_dataset.run()

    # @unittest.skip("california_housing_dataset example - Takes time")
    def test_california_housing(self):
        california_housing_dataset.run()

    # @unittest.skip("kddcup99 example - Takes time")
    def test_kddcup99(self):
        kddcup99_dataset.run()
