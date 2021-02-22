import unittest

from pytolemaic import PyTrust


class TestPyTrust(unittest.TestCase):

    def test_example_usage(self):
        example = PyTrust.print_usage_example()
        for component in ['plot()', 'insights', 'to_dict()', 'to_dict_meaning()']:
            try:
                assert 'pytrust.report.{}'.format(component) in example
            except:
                print("component=", component)
                print("example=\n", example)
                raise

        # todo use eval?

    def test_init_example_usage(self):
        example = PyTrust.print_initialization_example()
        # todo use eval?
