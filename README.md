Pytolemaic package goal to help creating trust in machine learning models. This is achieved via analysis of a the models.
The package supports classification/regression models built for tabular datasets.

The package contains the following functionality:

- Sensitivity Analysis: Calculation of feature importance for given model, either via sensitivity to feature value or sensitivity to missing values. Additionaly, the feature sensitivity is used to estimate the model's vulnerability in respect to imputation, leakage, and overfit.
- Scoring report: Report given model's score with confidence interval.
- Prediction uncertainty: Provides an uncertainty measure for given model's prediction.


How to use: See examples in examples directory.
Output examples:
- Sensitivity Analysis:
 'perturbed_sensitivity': {'no importance feature': 0.00281,
                           'regular feature': 0.25153,
                           'triple importance feature': 0.74566},
 'perturbed_sensitivity_meta': {'n_features': 3,
                                'n_low': 1,
                                'n_non_zero': 3,
                                'n_zero': 0},
 'perturbed_sensitivity_scores': {'imputation_score': 0.66667,
                                  'leakge_score': 0.0,
                                  'overfit_score': 0.33333}}

- scoring report