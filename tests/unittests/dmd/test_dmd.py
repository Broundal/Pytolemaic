import unittest

import numpy

from pytolemaic.utils.dmd import DMD


class TestDMD(unittest.TestCase):

    def _func(self, x, is_classification):
        y = numpy.sum(x, axis=1) + x[:, 0] - x[:, 1]
        if is_classification:
            y = numpy.round(y, 0).astype(int)
        return y.reshape(-1, 1)

    def get_data(self, is_classification):
        x = numpy.random.rand(1000, 10)

        # 1st is double importance, 2nd has no importance
        y = self._func(x, is_classification=is_classification)
        return DMD(x=x, y=y,
                   columns_meta={DMD.FEATURE_NAMES: ['f_' + str(k) for k in
                                                     range(x.shape[1])]},
                   samples_meta={'sample_weight': numpy.random.rand(x.shape[0])})

    def test_properties(self):
        dmd = self.get_data(is_classification=False)

        self.assertEqual(dmd.n_features, 10)
        self.assertEqual(dmd.n_samples, 1000)
        self.assertTrue(numpy.all(
            dmd.target == self._func(dmd.values, is_classification=False)))

        self.assertListEqual(dmd.feature_names,
                             ['f_' + str(k) for k in range(dmd.n_features)])

    def test_append(self):

        dmd1 = self.get_data(is_classification=False)
        dmd2 = self.get_data(is_classification=True)

        self.assertEqual(dmd1.n_samples, dmd2.n_samples)

        dmd1.append(dmd2, axis=0)

        self.assertEqual(dmd1.n_samples, 2*dmd2.n_samples)
        self.assertEqual(dmd1._x.shape[0], dmd1._y.shape[0])
        self.assertEqual(dmd1._x.shape[0], dmd1._samples_meta.shape[0])

        self.assertEqual(dmd1.n_features, dmd2.n_features)

    def test_concat(self):
        dmd1 = self.get_data(is_classification=False)
        dmd2 = self.get_data(is_classification=True)

        self.assertEqual(dmd1.n_samples, dmd2.n_samples)

        dmd = DMD.concat([dmd1, dmd2], axis=0)

        self.assertEqual(dmd.n_samples, 2 * dmd2.n_samples)
        self.assertEqual(dmd._x.shape[0], 2* dmd1._y.shape[0])
        self.assertEqual(dmd._x.shape[0], 2* dmd1._samples_meta.shape[0])

        self.assertEqual(dmd.n_features, dmd2.n_features)
