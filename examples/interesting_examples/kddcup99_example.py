from pprint import pprint

from pytolemaic import Metrics
from pytolemaic import PyTrust
from pytolemaic.utils.general import tic, toc
from resources.datasets.kddcup99 import KDDCup99


def run(fast=False):
    dataset = KDDCup99(subset=fast)
    classifier = dataset.get_model()
    train, test = dataset.as_dmd()

    metric = Metrics.recall.name

    pytrust = PyTrust(
        model=classifier,
        xtrain=train,
        xtest=test,
        metric=metric)

    print("We've trained a ML model (details below) on KDDCup99 dataset. Let's see whether our model is a good one")
    print("Model details\n", classifier, '\n\n')

    print("Let's analyze the dataset")
    tic("dataset_analysis_report")
    pytrust.dataset_analysis_report.plot()
    toc("dataset_analysis_report")

    print("Let's calculate score report")
    tic("scoring_report")
    scoring_report = pytrust.scoring_report
    toc("scoring_report")
    print("\nNow let's deepdive into the report!")
    scoring_report_deepdive(scoring_report)

    print("\n\nNext we'd like to check feature sensitivity")
    tic("sensitivity_report")
    sensitivity_report = pytrust.sensitivity_report
    toc("sensitivity_report")

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
    tic("insights_summary")
    print('\n'.join(pytrust.insights))
    toc("insights_summary")

    print("\nLet's create a Lime explainer")
    lime_explainer = pytrust.create_lime_explainer(max_samples=6000 if fast else 64000)

    sample = test.values[0, :]
    print("And plot explanation for the first sample in test data: {}".format(sample))
    lime_explainer.plot(sample)
    explanation = lime_explainer.explain(sample)
    print("Lime explanation is: {}".format(explanation))


def sensitivity_deepdive(sensitivity_report):
    print("\nlet's check which 3 features are most important. Does it make sense?")
    print(sensitivity_report.shuffle_report.sorted_sensitivities[:3])
    print("Looking on top 3 features in sensitivity to missing values we see there is some difference")
    print(sensitivity_report.missing_report.sorted_sensitivities[:3])
    print("This means that error caused by missing values affect the model differently than a regular mistake")
    print(
        "\nThere are {} features with low sensitivity of which {} features are with 0 sensitivity (out of {} features)".format(
            sensitivity_report.shuffle_report.stats_report.n_low, sensitivity_report.shuffle_report.stats_report.n_zero,
            sensitivity_report.shuffle_report.stats_report.n_features))

    print("\nLet see the features w/o sensitivity. Does it make sense?")
    print(
        sensitivity_report.shuffle_report.sorted_sensitivities[-sensitivity_report.shuffle_report.stats_report.n_zero:])

    print(
        "\nUsing the sensitivity report we can obtain some vulnerability measures (lower is better). The meaning of the fields can be obtained with to_dict_meaning()")
    pprint(sensitivity_report.vulnerability_report.to_dict(printable=True), width=120)
    print('\n\n*** vulnerability_report explanation was commented out ***')
    # pprint(sensitivity_report.vulnerability_report.to_dict_meaning(), width=120)

    print("We see that the imputation measure is relatively high, which means the model is sensitive "
          "to imputation method")
    print("However, none of the values seems to be high, which is reassuring")

    print("\n\nWe can see entire sensitivity report as well as explanation for the various fields using"
          " to_dict(printable=True) and to_dict_meaning()")
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

    print("\nLet's check the target score first - ")
    print('{} score is {:0.3f} which is perfect'.format(metric, score_value))
    print(
        'Confidence interval is [{:0.3f}, {:0.3f}] which implies a ci_ratio of {:0.3f} which is quite large. Consider increase test set size.'.format(
            ci_low, ci_high, ci_ratio))

    print("\nNow check out the confusion matrix - regular and normalized version")
    print('Regular confusion matrix:\n', scoring_report.confusion_matrix.confusion_matrix[0], '\n',
          scoring_report.confusion_matrix.confusion_matrix[1], '\n')
    print('Normalized Confusion matrix:\n', scoring_report.confusion_matrix.normalized_confusion_matrix[0], '\n',
          scoring_report.confusion_matrix.normalized_confusion_matrix[1])
    print("Looking on matrices we see that the score is very good - almost perfoct")

    print("\nFinally, let's look on the separation quality")

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
