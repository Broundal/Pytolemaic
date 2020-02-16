import numpy
import sklearn.datasets
import sklearn.model_selection
from sklearn.tree import DecisionTreeClassifier
from pytolemaic.pytrust import PyTrust
from pytolemaic.utils.dmd import DMD


def run():
    # Dataset: xtrain, ytrain, xtest, ytest
    data = sklearn.datasets.load_wine(return_X_y=False)

    x = data['data']
    y = data['target']
    feature_names = data['feature_names']
    labels = data['target_names']

    train_inds, test_inds = sklearn.model_selection.train_test_split(
        numpy.arange(len(data['data'])), test_size=0.3)

    xtrain, ytrain = x[train_inds], y[train_inds]
    xtest, ytest = x[test_inds], y[test_inds]

    # Train estimator
    estimator = DecisionTreeClassifier()
    estimator.fit(xtrain, ytrain)

    # Initiating PyTrust
    pytrust = PyTrust(
        model=estimator,
        xtrain=xtrain, ytrain=ytrain,
        xtest=xtest, ytest=ytest,
        metric='recall')

    # Initiating PyTrust with more information
    pytrust = PyTrust(
        model=estimator,
        xtrain=xtrain, ytrain=ytrain,
        xtest=xtest, ytest=ytest,
        columns_meta={DMD.FEATURE_NAMES: feature_names},
        labels=labels,
        splitter='stratified',
        metric='recall')

if __name__ == '__main__':
    run()