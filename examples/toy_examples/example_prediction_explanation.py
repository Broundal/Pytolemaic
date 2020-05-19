from matplotlib import pyplot as plt

from pytolemaic import PyTrust
from resources.datasets.linear import LinearRegressionDataset, LinearClassificationDataset


def run():
    ## For this example we create train/test data representing a linear function
    # PyTrust supports both numpy and pandas.DataFrame.

    # Obtain simple regression dataset. Use LinearClassificationDataset for classification
    for dataset in [LinearRegressionDataset(), LinearClassificationDataset()]:
        columns_names = dataset.column_names()

        # for quality report, we need for train/test sets and model
        xtrain, ytrain = dataset.training_data
        xtest, ytest = dataset.get_samples()
        estimator = dataset.get_model()

        # set the feature names names
        pytrust = PyTrust(
            model=estimator,
            xtrain=xtrain, ytrain=ytrain,
            xtest=xtest, ytest=ytest,
            feature_names=columns_names)

        sample = xtest[0, :]

        # Create explanation for target sample
        print("\nLet's create a Lime explainer")
        lime_explainer = pytrust.create_lime_explainer(max_samples=8000)

        print("And plot explanation for the first sample in test data: {}".format(sample))
        lime_explainer.plot(sample)

        explanation = lime_explainer.explain(sample)
        print("Lime explanation is: {}".format(explanation))



if __name__ == '__main__':
    run()
    plt.show()
