import unittest

import numpy

from pytolemaic import DMD
from pytolemaic.utils.pipeline_base import DMDPipeline, EstimatorBase, TransformerBase


class TransformerDummy(TransformerBase):
    def __init__(self, **kwargs):
        super(TransformerDummy, self).__init__(**kwargs)
        print('TransformerDummy', '__init__', 'name', self.name)

    def fit_transform(self, dmd: DMD, **kwargs):
        print('TransformerDummy', 'fit_transform', 'name', self.kwargs.get('name'))
        return dmd

    def transform(self, dmd: DMD, **kwargs):
        print('TransformerDummy', 'transform', 'name', self.kwargs.get('name'))
        return dmd


class EstimatorDummy(EstimatorBase):
    def __init__(self, **kwargs):
        super(EstimatorDummy, self).__init__(**kwargs)
        print('EstimatorDummy', '__init__', 'name', self.name)

    def fit(self, dmd: DMD, **kwargs):
        print('EstimatorDummy', 'fit', 'n_estimators', self.kwargs.get('n_estimators'))
        pass

    def predict(self, dmd: DMD, **kwargs):
        print('EstimatorDummy', 'predict', 'n_estimators', self.kwargs.get('n_estimators'))
        return DMD(x=dmd.values[:, 0:1])

    def predict_proba(self, dmd: DMD, **kwargs):
        print('EstimatorDummy', 'predict_proba', 'n_estimators', self.kwargs.get('n_estimators'))
        return DMD(x=dmd.values[:, 0:2])

class TestPyTrust(unittest.TestCase):

    def test_dummy(self):
        dmd_train = DMD(x=numpy.zeros((3, 3)), y=numpy.zeros((3, 1)))
        dmd_test = DMD(x=numpy.zeros((3, 3)), y=numpy.zeros((3, 1)))

        print('init model')
        model = DMDPipeline(transformers=[TransformerDummy],
                            estimator=EstimatorDummy,
                            hyperparameters=[{'name': 'Dummy'}, {'n_estimators': 10}])

        self.assertIsInstance(model.name, str)

        self.assertIsInstance(model.transformers, list)
        self.assertEqual(len(model.transformers), 1)
        self.assertIsInstance(model.transformers[0], TransformerBase)
        self.assertEqual(model.transformers[0].name, 'Dummy')

        self.assertIsInstance(model.estimator, EstimatorBase)
        self.assertEqual(model.estimator.kwargs['n_estimators'], 10)

        self.assertEqual(len(model.hyperparameters), 2)
        self.assertEqual((model.transformers[0].name, {'name': 'Dummy'}),
                         model.hyperparameters[0])
        self.assertEqual((model.estimator.name, {'n_estimators': 10}),
                         model.hyperparameters[1])


        print('fit model')
        model.fit(dmd_train)

        print('model predict')
        predictions = model.predict(dmd_test)
        self.assertIsInstance(predictions, DMD)
