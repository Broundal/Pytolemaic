from pprint import pprint

import numpy

from pytolemaic import Metrics
from pytolemaic import PyTrust
from resources.datasets.california_housing import CaliforniaHousing


def run(fast=False):
    dataset = CaliforniaHousing()
    estimator = dataset.get_model()
    train, test = dataset.as_dmd()

    metric = Metrics.rmse.name

    pytrust = PyTrust(
        model=estimator,
        xtrain=train,
        xtest=test,
        metric=metric)

    print("We've trained a ML model (details below) on California Housing dataset.\n"
          "We should note that the target values are in range of [{}], which probably mean they were normalized beforehand."
          "Let's see whether our model is a good one.".format((numpy.min(train.target), numpy.max(train.target))))

    print("Model details\n", estimator, '\n\n')

    print("Let's analyze the dataset")
    print("Calculating...")
    pytrust.dataset_analysis_report.plot()
    print('\n'.join(pytrust.dataset_analysis_report.insights()))
    print("Calculating... Done")

    print("Let's calculate score report")
    print("Calculating...")
    scoring_report = pytrust.scoring_report
    print("Calculating... Done")
    print("\nNow let's deepdive into the report!")
    scoring_report_deepdive(scoring_report)

    print("\n\nNext we'd like to check feature sensitivity")
    print("Calculating...")
    sensitivity_report = pytrust.sensitivity_report
    print("Calculating... Done")

    print("\nNow let's deepdive into the report!")
    sensitivity_deepdive(sensitivity_report)

    print("\nFinally let's review overall quality score!")
    quality_report = pytrust.quality_report

    print("Overall quality of train data: {:0.3f}".format(quality_report.train_quality_report.train_set_quality))
    print("Overall quality of test data: {:0.3f}".format(quality_report.test_quality_report.test_set_quality))
    print("Overall quality of model: {:0.3f}".format(quality_report.model_quality_report.model_quality))
    print('*** quality_report was commented out ***')
    # pprint(quality_report.to_dict(printable=True), width=120)
    # pprint(quality_report.to_dict_meaning(), width=120)

    print("Let's check for insights...")
    print('\n'.join(pytrust.insights))
    print("Done!")

    print("\nLet's create a Lime explainer")
    lime_explainer = pytrust.create_lime_explainer(max_samples=16000 if fast else 64000)

    sample = test.values[0, :]
    print("And plot explanation for the first sample in test data: {}".format(sample))
    lime_explainer.plot(sample)
    explanation = lime_explainer.explain(sample)
    print("Lime explanation is: {}".format(explanation))


def sensitivity_deepdive(sensitivity_report):
    print("\nlet's check which 3 features are most important. Does it make sense?")
    print(sensitivity_report.shuffle_report.sorted_sensitivities[:3], '\n')

    print("Compare sensitivities obtain which shuffle method and missing method")
    print('shuffle method', sensitivity_report.shuffle_report.sorted_sensitivities[:3], '\n'
                                                                                        'missing method: N/A')

    print("Looking on top 3 features we see similar values. "
          "This means that error caused by missing values affect the model similarly to a regular mistake")

    print("\nThe 3 least important features are:")
    print(sensitivity_report.shuffle_report.sorted_sensitivities[-3:])
    print("Thus there are {} features with 0 sensitivity and {} features with low sensitivity".format(
        sensitivity_report.shuffle_report.stats_report.n_zero, sensitivity_report.shuffle_report.stats_report.n_low))

    print(
        "\nUsing the sensitivity report we can obtain some vulnerability measures (lower is better). The meaning of the fields can be obtained with to_dict_meaning()")
    pprint(sensitivity_report.vulnerability_report.to_dict(printable=True), width=120)
    print('*** vulnerability_report explanation was commented out ***')
    # pprint(sensitivity_report.vulnerability_report.to_dict_meaning(), width=120)

    print("We see that the imputation measure is relatively high, which means the model is sensitive "
          "to imputation method to some extent")
    print("However, the other values are 0 which is reassuring")
    print('\n')

    print(
        "\nWe can see entire sensitivity report as well as explanation for the various fields using to_dict(printable=True) and to_dict_meaning()")
    print('*** sensitivity_report was commented out ***')
    # pprint(sensitivity_report.to_dict(printable=True), width=120)
    # pprint(sensitivity_report.to_dict_meaning(), width=120)
    print("\nNow let's plot some nice graphs")
    sensitivity_report.plot()


def scoring_report_deepdive(scoring_report):
    metric = scoring_report.target_metric
    score_value = scoring_report.metric_scores[metric].value
    ci_low = scoring_report.metric_scores[metric].ci_low
    ci_high = scoring_report.metric_scores[metric].ci_high
    ci_ratio = scoring_report.metric_scores[metric].ci_ratio
    quality = scoring_report.separation_quality
    print("\nLet's check the target score first - ")
    print("{} score is {:0.3f} which doesn't tell us much regarding model's quality".format(metric, score_value))
    print("So let's check {0} which is relative to std(target). {0} score is {1:0.3f} which is not very good".format(
        Metrics.normalized_rmse.name, scoring_report.metric_scores[Metrics.normalized_rmse.name].value))
    print('Confidence interval is [{:0.3f}, {:0.3f}] which implies a ci_ratio of {:0.3f} which is quite good'.format(
        ci_low, ci_high, ci_ratio))

    print("\nFinally, let's look on the separation quality")
    print("Score quality is {:0.3f} which perfect.".format(quality))
    print(
        "\nWe can see entire scoring report as well as explanation for the various fields using to_dict(printable=True) and to_dict_meaning()")
    print('*** scoring_report was commented out ***')
    # pprint(scoring_report.to_dict(printable=True), width=120)
    # pprint(scoring_report.to_dict_meaning(), width=120)

    print("\nAnd also plot some nice graphs")
    scoring_report.plot()


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    run()
    plt.show()
