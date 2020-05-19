from pprint import pprint

from matplotlib import pyplot as plt

## For this example we create train/test data representing a linear function
# both numpy and pandas.DataFrame is ok.
from pytolemaic import DMD
from pytolemaic import Metrics
from pytolemaic import PyTrust
from resources.datasets.linear import LinearClassificationDataset


def run():
    ## For this example we create train/test data representing a linear function
    # PyTrust supports both numpy and pandas.DataFrame.

    # Obtain simple classification dataset. Use LinearRegressionDataset for regression
    dataset = LinearClassificationDataset()
    columns_names = dataset.column_names()

    # for quality report, we need for train/test sets and model
    xtrain, ytrain = dataset.training_data
    xtest, ytest = dataset.get_samples()
    classifier = dataset.get_model()

    ## set metric
    metric = Metrics.recall.name

    ## set splitting strategy
    splitter = 'stratified'

    ## sample meta data (e.g. sample weight) - empty in this example
    sample_meta_train = None
    sample_meta_test = None

    # set the feature names names
    columns_meta = {DMD.FEATURE_NAMES: [name for name in columns_names]}

    pytrust = PyTrust(
        model=classifier,
        xtrain=xtrain, ytrain=ytrain,
        xtest=xtest, ytest=ytest,
        sample_meta_train=sample_meta_train, sample_meta_test=sample_meta_test,
        columns_meta=columns_meta,
        metric=metric,
        splitter=splitter)

    scoring_report = pytrust.scoring_report

    score_value = scoring_report.metric_scores[metric].value
    ci_low = scoring_report.metric_scores[metric].ci_low
    ci_high = scoring_report.metric_scores[metric].ci_high
    quality = scoring_report.separation_quality

    print('{} score is {:0.3f}'.format(metric, score_value))
    print('Score quality is {:0.3f}'.format(quality))
    print('Confidence interval is [{:0.3f}, {:0.3f}]'.format(ci_low, ci_high))

    pprint(scoring_report.to_dict(printable=True), width=160)
    pprint(scoring_report.to_dict_meaning(), width=120)

    scoring_report.plot()


if __name__ == '__main__':
    run()
    plt.show()
