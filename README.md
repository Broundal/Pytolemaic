## What is Pytolemaic 
Pytolemaic package aims to help you analyze your models to check their quality. 

The package supports classification/regression models built for tabular datasets (mainly sklearn),
 but will also support custom made models as long as they implement sklearn's API. 

The package is aimed for personal use and comes with no guarantees. 
I hope you will find it useful and appreciate feedback.

## supported features
The package contains the following functionality:

- Sensitivity Analysis: Calculation of feature importance for given model, either via sensitivity to feature value or sensitivity to missing values. Additionaly, the feature sensitivity is used to estimate the model's vulnerability in respect to imputation, leakage, and overfit.
- Scoring report: Report given model's score with confidence interval.
- Prediction uncertainty: Provides an uncertainty measure for given model's prediction.


## How to use: 
Examples can be found in examples directory.

## Output examples:

#### Sensitivity Analysis:

 - The sensitivity of each feature (normalized to 1):
 
```
 'perturbed_sensitivity': 
    {'no importance feature': 0.00281,
     'regular feature': 0.25153,
     'triple importance feature': 0.74566},
  
```
                                                        
 - Simple statistics on the feature sensitivity:
 ```
 'perturbed_sensitivity_meta': 
     {'n_features': 3,
      'n_low': 1,
      'n_non_zero': 3,
      'n_zero': 0},
 ```
 
 - Naive quality scores based on my experience:

   - *imputation score*: sensitivity of the model ot missing values.
   - *leakge score*: chance of the model to have leaking features.
   - *overfit score*: chance of the model is overfitted on the data.
 
 ```
 'perturbed_sensitivity_scores': 
    {'imputation_score': 0.66667,
     'leakge_score': 0.0,
      'overfit_score': 0.33333}
 ```


#### scoring report

For given metric, the score and confidence intervals (ci) is calculated
 ```
 'recall': 
    {'ci_high': 0.95513, 
     'ci_low': 0.95171, 
     'value': 0.95343}}'    
 ```
 
 Additionally, score quality measures the quality of the score based on the separability between train and test sets.
 ```
 'Quality': 0.987         
 ```
  
 
#### prediction uncertainty

The module can be used to yield uncertainty measure for predictions. 
```
    uncertainty_model = pytrust.create_uncertainty_model(method=method)
    predictions = uncertainty_model.predict(x_pred) # same as model.predict(x_pred)
    uncertainty = uncertainty_model.uncertainty(x_pred)
```