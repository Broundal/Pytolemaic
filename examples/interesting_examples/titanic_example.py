import os

import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from pytolemaic import FeatureTypes, DMD, HOME_DIR, PyTrust


def run(fast=False):
    # read data
    train_path = os.path.join(HOME_DIR, 'resources', 'datasets', 'titanic-train.csv')

    df_train = pandas.read_csv(train_path)

    num, cat = FeatureTypes.numerical, FeatureTypes.categorical
    dmd_train, dmd_test = DMD.from_df(df_train=df_train, df_test=None,
                                      is_classification=True,
                                      target_name='Survived',
                                      feature_types=[num, cat, cat, cat, num, num, num, cat, num, cat, cat],
                                      categorical_encoding=True, nan_list=['?'],
                                      split_ratio=0.2)

    classifier = Pipeline(steps=[('Imputer', SimpleImputer()),
                                 ('Estimator', RandomForestClassifier(n_estimators=3))])

    classifier.fit(dmd_train.values, dmd_train.target)

    pytrust = PyTrust(
        model=classifier,
        xtrain=dmd_train,
        xtest=dmd_test,
        metric='recall')

    # some analysis
    print('\n'.join(pytrust.insights))

    pytrust.dataset_analysis_report.plot()
    pytrust.scoring_report.plot()
    pytrust.sensitivity_report.plot()


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    run()
    plt.show()
