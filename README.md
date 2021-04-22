![Unittests](https://github.com/Broundal/Pytolemaic/workflows/Unittests/badge.svg?branch=master)

# Pytolemaic

## What is Pytolemaic 
Pytolemaic package analyzes your model and dataset and measure their quality. 

The package supports classification/regression models built for tabular datasets (e.g. sklearn's regressors/classifiers),
 but will also support custom made models as long as they implement sklearn's API. 

The package is aimed for personal use and comes with no guarantees. 
I hope you will find it useful. I will appreciate any feedback you have.

## Install
Pytolemaic package may be installed using ```pip```:
```
pip install pytolemaic
```

## supported features
The package contains the following functionalities:

#### On model creation
- **Dataset Analysis**: Analysis aimed to detect issues in the dataset.
- **Sensitivity Analysis**: Calculation of feature importance for given model, either via sensitivity to feature value or sensitivity to missing values. 
- **Vulnerability report**: Based on the feature sensitivity we measure model's vulnerability in respect to imputation, leakage, and # of features.
- **Scoring report**: Report model's score on test data with confidence interval.
- **separation quality**: Measure whether train and test data comes from the same distribution.
- **Overall quality**: Provides overall quality measures

#### On prediction
- **Prediction uncertainty**: Provides an uncertainty measure for given model's prediction.
- **Lime explanation**: Provides Lime explanation for sample of interest.



## How to use: 

Get started by calling help() function (*Recommended!*):
```
   from pytolemaic import help
   supported_keys = help()
   # or
   help(key='basic usage')
```

Example for performing all available analysis with PyTrust:
```
   from pytolemaic import PyTrust

   pytrust = PyTrust(
       model=estimator,
       xtrain=xtrain, ytrain=ytrain,
       xtest=xtest, ytest=ytest)
       
   # run all analysis and get a list of distilled insights",
   insights = pytrust.insights()
   print("\n".join(insights))
    
   # run all analysis and plot all graphs
   pytrust.plot()
   
   # print all data gathered
   import pprint
   pprint(report.to_dict(printable=True))
```

In case of need to access only specific analysis (usually to save time)
```
   # dataset analysis report
   dataset_analysis_report = pytrust.dataset_analysis_report
   
   # feature sensitivity report
   sensitivity_report = pytrust.sensitivity_report
   
   # model's performance report
   scoring_report = pytrust.scoring_report
   
   # overall model's quality report
   quality_report = pytrust.quality_report
   
   # with any of the above reports
   report = <desired report>
   print("\n".join(report.insights()))
   
   report.plot() # plot graphs
   pprint(report.to_dict(printable=True)) # export report as a dictionary
   pprint(report.to_dict_meaning()) # print documentation for above dictionary
          
```

Analysis of predictions
```
 
   # estimate uncertainty of a prediction
   uncertainty_model = pytrust.create_uncertainty_model()
   
   # explain a prediction with Lime
   create_lime_explainer = pytrust.create_lime_explainer()
   
```

Examples on toy dataset can be found in [/examples/toy_examples/](./examples/toy_examples/)
Examples on 'real-life' datasets can be found in [/examples/interesting_examples/](./examples/interesting_examples/) 

## Output examples:

#### Sensitivity Analysis:

 - The sensitivity of each feature (\[0,1\], normalized to sum of 1):
 
```
 'sensitivity_report': {
    'method': 'shuffled',
    'sensitivities': {
        'age': 0.12395,
        'capital-gain': 0.06725,
        'capital-loss': 0.02465,
        'education': 0.05769,
        'education-num': 0.13765,
        ...
      }
  }
```
                                                        
 - Simple statistics on the feature sensitivity:
 ```
 'shuffle_stats_report': {
      'n_features': 14,
      'n_low': 1,
      'n_zero': 0
 }
 ```
 
 - Naive vulnerability scores (\[0,1\], lower is better):

   - **Imputation**: sensitivity of the model to missing values.
   - **Leakge**: chance of the model to have leaking features.
   - **Too many features**: Whether the model is based on too many features.
 
 ```
 'vulnerability_report': {
      'imputation': 0.35,
      'leakage': 0,
      'too_many_features': 0.14
 }  
 ```

#### scoring report

For given metric, the score and confidence intervals (CI) is calculated
 ```
'recall': {
     'ci_high': 0.763,
     'ci_low': 0.758,
     'ci_ratio': 0.023,
     'metric': 'recall',
     'value': 0.760,
},
'auc': {
     'ci_high': 0.909,
     'ci_low': 0.907,
     'ci_ratio': 0.022,
     'metric': 'auc',
     'value': 0.907
}    
 ```
 
 Additionally, score quality measures the quality of the score based on the separability (auc score) between train and test sets.
 
 Value of 1 means test set has same distribution as train set. Value of 0 means test set has fundamentally different distribution. 
 ```
 'separation_quality': 0.00611         
 ```
  
Combining the above measures into a single number we provide the overall quality of the model/dataset.

Higher quality value (\[0,1\]) means better dataset/model.
 ```
quality_report : { 
'model_quality_report': {
    'model_loss': 0.24,
    'model_quality': 0.41,
    'vulnerability_report': {...}},
    
'test_quality_report': {
    'ci_ratio': 0.023, 
    'separation_quality': 0.006, 
    'test_set_quality': 0},
    
'train_quality_report': {
    'train_set_quality': 0.85,
    'vulnerability_report': {...}}
   
 ```

 
#### prediction uncertainty

The module can be used to yield uncertainty measure for predictions. 
```
    uncertainty_model = pytrust.create_uncertainty_model(method='confidence')
    predictions = uncertainty_model.predict(x_pred) # same as model.predict(x_pred)
    uncertainty = uncertainty_model.uncertainty(x_pred)
```


#### Lime explanation

The module can be used to produce lime explanations for sample of interest. 
```
    explainer = pytrust.create_lime_explainer()
    explainer.explain(sample) # returns a dictionary
    explainer.plot(sample) # produce a graphical explanation    
```
