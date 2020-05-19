from pprint import pprint

from pytolemaic import Metrics
from pytolemaic import PyTrust
## For this example we create train/test data representing a linear function
# both numpy and pandas.DataFrame is ok.
from pytolemaic.utils.dmd import DMD
from resources.datasets.linear import LinearRegressionDataset


def run():
    ## For this example we create train/test data representing a linear function
    # PyTrust supports both numpy and pandas.DataFrame.

    # Obtain simple regression dataset. Use LinearClassificationDataset for classification
    dataset = LinearRegressionDataset()
    columns_names = dataset.column_names()

    # for quality report, we need for train/test sets and model
    xtrain, ytrain = dataset.training_data
    xtest, ytest = dataset.get_samples()
    regressor = dataset.get_model()


    ## set metric
    metric = Metrics.mae.name

    ## set splitting strategy
    splitter = 'shuffled'

    ## sample meta data (e.g. sample weight) - empty in this example
    sample_meta_train = None
    sample_meta_test = None

    # set the feature names names
    columns_meta = {DMD.FEATURE_NAMES: columns_names}

    pytrust = PyTrust(
        model=regressor,
        xtrain=xtrain, ytrain=ytrain,
        xtest=xtest, ytest=ytest,
        sample_meta_train=sample_meta_train, sample_meta_test=sample_meta_test,
        columns_meta=columns_meta,
        metric=metric,
        splitter=splitter)

    quality_report = pytrust.quality_report
    print("Quality report - higher is better")
    pprint(quality_report.to_dict(printable=True), width=120)
    pprint(quality_report.to_dict_meaning(), width=120)


if __name__ == '__main__':
    run()
