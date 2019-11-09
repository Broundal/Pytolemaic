import unittest

from pytolemaic.utils.report import Report


class TestMetrics(unittest.TestCase):

    def test_get_function(self):
        report = Report({k: k for k in list('abcde')})
        self.assertEqual(report.get('a'), 'a')
        self.assertEqual(report.get('f'), None)

        report = Report({k: {k: k} for k in list('abcde')})
        self.assertEqual(report.get('a').report, {'a':'a'})
        self.assertEqual(report.get('f'), None)
