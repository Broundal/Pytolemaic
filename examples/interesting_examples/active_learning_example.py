import time
from pprint import pprint

import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from pytolemaic import Metrics, DMD
from pytolemaic import PyTrust
from pytolemaic.utils.general import GeneralUtils
from resources.datasets.uci_adult import UCIAdult


def run(use_active_learning=True, max_samples = 12500, batch_size = 100, train_base_ratio=0.001):

    dataset = UCIAdult()
    train, test = dataset.as_dmd()

    metric = Metrics.recall.name

    unlabeled, train_base = train.split(ratio=train_base_ratio)

    print("# samples: Test: {}, train_base: {}, unlabeled:{}"
          .format(test.n_samples, train_base.n_samples, unlabeled.n_samples))


    x_samples=[]
    y_score =[]

    while train_base.n_samples < train.n_samples and unlabeled.n_samples > 100 and train_base.n_samples<max_samples:
        model = GeneralUtils.simple_imputation_pipeline(estimator=RandomForestClassifier(random_state=0))
        # model = GeneralUtils.simple_imputation_pipeline(estimator=KNeighborsClassifier(n_neighbors=5))
        # model = GeneralUtils.simple_imputation_pipeline(estimator=LogisticRegression())

        model.fit(train_base.values, train_base.target.ravel())

        pytrust = PyTrust(
            model=model,
            xtrain=train_base,
            xtest=test,
            metric=metric)

        score = pytrust.scoring_report.metric_scores['recall'].to_dict()['value']
        print("With {} samples, score is {:.3g}".format(train_base.n_samples,
                                                        score))
        x_samples.append(train_base.n_samples)
        y_score.append(score)

        uncertainty = pytrust.create_uncertainty_model('probability', do_analysis=False)
        y = uncertainty.uncertainty(unlabeled)

        if use_active_learning:
            inds = numpy.arange(len(y))
            sorted_y, sorted_inds = list(zip(*sorted(list(zip(y, inds)), key=lambda p: p[0], reverse=True)))
        else: # random sampling
            sorted_inds = numpy.random.permutation(len(y))

        batch_size = min(batch_size, len(sorted_inds)-100)
        train_base.append(unlabeled.split_by_indices(sorted_inds[:batch_size]))
        unlabeled = unlabeled.split_by_indices(sorted_inds[batch_size:])

    return x_samples, y_score

if __name__ == '__main__':
    from matplotlib import pyplot as plt

    x_samples, y_score = run(use_active_learning=True)
    plt.plot(x_samples, y_score, '.-b', label='active learning')
    x_samples, y_score = run(use_active_learning=False)
    plt.plot(x_samples, y_score, '.-r', label='random sampling')
    plt.legend()
    plt.show()
