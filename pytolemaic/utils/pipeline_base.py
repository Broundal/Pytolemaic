from typing import List, Tuple

import numpy

from pytolemaic import DMD

class TransformerBase():
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.name = kwargs.get('name', type(self).__name__)

    def fit_transform(self, dmd:DMD, **kwargs)->DMD:
        raise NotImplementedError("fit_transform")

    def transform(self, dmd:DMD, **kwargs)->DMD:
        raise NotImplementedError("transform")

class EstimatorBase():
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.name = kwargs.get('name', type(self).__name__)

    def fit(self, dmd:DMD, **kwargs)->None:
        raise NotImplementedError("fit")

    def predict(self, dmd:DMD, **kwargs)->DMD:
        raise NotImplementedError("predict")

    def predict_proba(self, dmd:DMD, **kwargs)->DMD:
        raise NotImplementedError("predict_proba")


class DMDPipeline():
    def __init__(self, transformers: List[type(TransformerBase)],
                 estimator: type(EstimatorBase),
                 hyperparameters: List[dict],
                 name=None,
                 **kwargs):

        self.transformers = [transformer_class(**hps) for transformer_class, hps in zip(transformers, hyperparameters[:-1])]
        self.estimator = estimator(**hyperparameters[-1])

        self.hyperparameters = [(transformer.name, hps) for transformer, hps in zip(self.transformers, hyperparameters[:-1])]
        self.hyperparameters.append((self.estimator.name, hyperparameters[-1]))

        default_name = '>'.join([transformer.name for transformer in self.transformers]) + '>' + self.estimator.name
        self.name = name if name is not None else default_name

    def fit(self, dmd, **kwargs):
        for transformer in self.transformers:
            dmd = transformer.fit_transform(dmd, **kwargs)

        self.estimator.fit(dmd, **kwargs)

    def transform(self, dmd:DMD, **kwargs)->DMD:
        for transformer in self.transformers:
            dmd = transformer.transform(dmd, **kwargs)

        return dmd

    def predict(self, dmd:DMD, **kwargs)->DMD:
        dmd = self.transform(dmd, **kwargs)
        return self.estimator.predict(dmd, **kwargs)

    def predict_proba(self, dmd:DMD, **kwargs)->DMD:
        dmd = self.transform(dmd, **kwargs)
        return self.estimator.predict_proba(dmd, **kwargs)

if __name__ == '__main__':
    class TransformerDummy():
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.name = kwargs.get('name', type(self).__name__)
            print('TransformerDummy', '__init__', 'name', self.name)


        def fit_transform(self, dmd: DMD, **kwargs):
            print('TransformerDummy','fit_transform','name',self.kwargs.get('name'))
            return dmd

        def transform(self, dmd: DMD, **kwargs):
            print('TransformerDummy', 'transform', 'name', self.kwargs.get('name'))
            return dmd


    class EstimatorDummy():
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.name = kwargs.get('name', type(self).__name__)
            print('EstimatorDummy', '__init__', 'name', self.name)


        def fit(self, dmd: DMD, **kwargs):
            print('EstimatorDummy','fit','n_estimators',self.kwargs.get('n_estimators'))
            pass

        def predict(self, dmd: DMD, **kwargs):
            print('EstimatorDummy','predict','n_estimators',self.kwargs.get('n_estimators'))
            return DMD(x=dmd.values[:,0:1])

        def predict_proba(self, dmd: DMD, **kwargs):
            print('EstimatorDummy','predict_proba','n_estimators',self.kwargs.get('n_estimators'))
            return DMD(x=dmd.values[:,0:2])

    dmd_train = DMD(x=numpy.zeros((3,3)), y=numpy.zeros((3,1)))
    dmd_test = DMD(x=numpy.zeros((3,3)), y=numpy.zeros((3,1)))

    print('init model')
    model = DMDPipeline(transformers=[TransformerDummy],
                        estimator=EstimatorDummy,
                        hyperparameters=[{'name': 'Dummy'}, {'n_estimators':10}])

    print('fit model')
    model.fit(dmd_train)

    print('model predict')
    predictions = model.predict(dmd_test)

    """
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestClassifier

    model = DMDPipeline(transformers=[SimpleImputer],
                        estimator=RandomForestClassifier,
                        hyperparameters=[{}, {'n_estimators':10}])
              
    dmd_train = DMD()
    dmd_test = DMD()
    model.fit(dmd_train)
    predictions = model.predict(dmd_test)
    score = evaluate(predictions, dmd_test)
    """