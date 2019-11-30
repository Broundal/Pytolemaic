# Pytolemaic

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
Examples can be found in /examples.

## Output examples:

#### Sensitivity Analysis:

 - The sensitivity of each feature (\[0,1\], normalized to sum of 1):
 
```
 'sensitivity_report': {
    'method': 'shuffled',
     'sensitivities': {
          'f0': 0.18533,
          'f1': 0.0,
          'f2': 0.09715,
          'f3': 0.10528,
          'f4': 0.09941,
          'f5': 0.10491,
          'f6': 0.10788,
          'f7': 0.1063,
          'f8': 0.09109,
          'f9': 0.10264
      }
  }
```
                                                        
 - Simple statistics on the feature sensitivity:
 ```
 'shuffle_stats_report': {
      'n_features': 10,
      'n_low': 1,
      'n_zero': 1
 }
 ```
 
 - Naive vulnerability scores (\[0,1\], lower is better):

   - **Imputation**: sensitivity of the model to missing values.
   - **Leakge**: chance of the model to have leaking features.
   - **Too many features**: Whether the model is based on too many features.
 
 ```
 'vulnerability_report': {
      'imputation': 0.569,
      'leakage': 0,
      'too_many_features': 0.316
 }  
 ```

#### scoring report

For given metric, the score and confidence intervals (CI) is calculated
 ```
 'auc': {
     'ci_high': 0.949,
     'ci_low': 0.947,
     'ci_ratio': 0.057,
     'metric': 'auc',
     'value': 0.948,
 },
 'recall': {
     'ci_high': 0.870,
     'ci_low': 0.866,
     'ci_ratio': 0.022,
     'metric': 'recall',
     'value': 0.868
}    
 ```
 
 Additionally, score quality measures the quality of the score based on the separability (auc score) between train and test sets.
 ```
 'separation_quality': 0.969         
 ```
  
Combining the above measures into a single number we provide the overall quality of the model/dataset.

Higher quality value (\[0,1\]) means better dataset/model.
 ```
 
{'test_quality_report': {
    'test_set_quality': 0.930,
    'ci_ratio': 0.039,
    'separation_quality': 0.969,
    
},
 'train_quality_report': {
    'train_set_quality': 0.333,
    'vulnerability_report': {
        'imputation': 0.666,
        'leakage': 0.0,
        'too_many_features': 0.0}}}         
 ```

 
#### prediction uncertainty

The module can be used to yield uncertainty measure for predictions. 
```
    uncertainty_model = pytrust.create_uncertainty_model(method='confidence')
    predictions = uncertainty_model.predict(x_pred) # same as model.predict(x_pred)
    uncertainty = uncertainty_model.uncertainty(x_pred)
```