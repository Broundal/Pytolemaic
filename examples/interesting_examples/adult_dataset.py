from pprint import pprint

from examples.datasets.uci_adult import UCIAdult
from pytolemaic.pytrust import PyTrust
## For this example we create train/test data representing a linear function
# both numpy and pandas.DataFrame is ok.
from pytolemaic.utils.metrics import Metrics


def run():
    ## For this example we create train/test data representing a linear function
    # PyTrust supports both numpy and pandas.DataFrame.

    dataset = UCIAdult()
    classifier = dataset.get_model()
    train, test = dataset.as_dmd()

    metric = Metrics.recall.name

    pytrust = PyTrust(
        model=classifier,
        xtrain=train,
        xtest=test,
        metric=metric)

    print("We've trained a ML model (details below) on uci adult dataset. Let's see whether our model is a good one")
    print("Model details\n", classifier, '\n\n')

    print("First, let's calculate score report for Adult dataset ".format(metric))
    print("Calculating...")
    scoring_report = pytrust.scoring_report()
    print("Calculating... Done")
    print("\nNow let's deepdive intp the report!")
    scoring_report_deepdive(scoring_report, metric)



    print("\n\nNext we'd like to check feature sensitivity")
    print("Calculating...")
    sensitivity_report = pytrust.sensitivity_report()
    print("Calculating... Done")

    print("\nNow let's deepdive into the report!")
    sensitivity_deepdive(sensitivity_report)

    print("\nFinally let's review overall quality score!")
    quality_report = pytrust.quality_report()

    print("Overall quality of train data: {:0.3f}".format(quality_report.train_quality_report.train_set_quality))
    print("Overall quality of test data: {:0.3f}".format(quality_report.test_quality_report.test_set_quality))
    print("Overall quality of model: {:0.3f}".format(quality_report.model_quality_report.model_quality))
    print('*** quality_report was commented out ***')
    # pprint(quality_report.to_dict(), width=120)
    # pprint(quality_report.to_dict_meaning(), width=120)


def sensitivity_deepdive(sensitivity_report):
    print("\nlet's check which 3 features are most important. Does it make sense?")
    print(sensitivity_report.shuffle_report.sorted_sensitivities[:3])
    print("Looking on top 3 features in sensitivity to missing values we see there is some difference")
    print(sensitivity_report.missing_report.sorted_sensitivities[:3])
    print("This means that error caused by missing values affect the model differently than a regular mistake")
    print("\nThere are {} features with 0 sensitivity, and {} features with low sensitivity".format(
        sensitivity_report.shuffle_stats_report.n_zero, sensitivity_report.shuffle_stats_report.n_low))
    print("\nlet's check which 3 features are least important. Does it make sense?")
    print(sensitivity_report.shuffle_report.sorted_sensitivities[-3:])
    print("\nUsing the sensitivity report we can obtain some vulnerability measures (lower is better)")
    pprint(sensitivity_report.vulnerability_report.to_dict(), width=120)
    pprint(sensitivity_report.vulnerability_report.to_dict_meaning(), width=120)

    print("We see that the imputation measure is relatively high, which means the model is sensitive "
          "to imputation method")
    print("However, none of the values seems to be high, which is reassuring")
    print('*** sensitivity_report was commented out ***')
    # pprint(sensitivity_report.to_dict(), width=120)
    # pprint(sensitivity_report.to_dict_meaning(), width=120)
    print("\nNow let's plot some nice graphs")
    sensitivity_report.plot()


def scoring_report_deepdive(scoring_report, metric):
    score_value = scoring_report.metric_scores[metric].value
    ci_low = scoring_report.metric_scores[metric].ci_low
    ci_high = scoring_report.metric_scores[metric].ci_high
    ci_ratio = scoring_report.metric_scores[metric].ci_ratio
    quality = scoring_report.separation_quality
    print("\nLet's check the score first - ")
    print('{} score is {:0.3f} which is quite {}'.format(metric, score_value,
                                                         "good" if score_value > 0.9 else "reasonable"))
    print("\nChecking confidence interval - ")
    print('Confidence interval is [{:0.3f}, {:0.3f}] which implies ci_ratio of {:0.3f} which is {}'.format(ci_low,
                                                                                                           ci_high,
                                                                                                           ci_ratio,
                                                                                                           "quite good" if ci_ratio < 0.1 else "ok"))
    print("\nCheck out the confusion matrix - regular and normalized")
    print('Confusion matrix:\n', scoring_report.confusion_matrix.confusion_matrix[0], '\n',
          scoring_report.confusion_matrix.confusion_matrix[1], '\n')
    print('Normalized Confusion matrix:\n', scoring_report.confusion_matrix.normalized_confusion_matrix[0], '\n',
          scoring_report.confusion_matrix.normalized_confusion_matrix[1])
    print("Looking on matrices we see that performance is not too good")
    print("\nFinally, let's look on the separation quality")
    print('Score quality is {:0.3f} which is {}'.format(quality,
                                                        "very bad! --> test set isn't worth much" if quality < 0.3 else "reasonable"))
    print("\nWe can see entire scoring report as well as explanation for the various fields")

    print('*** scoring_report was commented out ***')
    # pprint(scoring_report.to_dict(), width=120)
    # pprint(scoring_report.to_dict_meaning(), width=120)

    print("\nAnd also plot some nice graphs")
    scoring_report.plot()


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    run()
    plt.show()
