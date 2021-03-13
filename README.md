# Optimizing an ML Pipeline in Azure
> Student : Amir Haddad / March 2021 . 
## Table of contents
   * [Overview](#Overview)
   * [Summary](#Summary)
   * [Hyperparameters](#Hyperparameters)
   * [AutoML](#AutoML)
   * [Benchmark](#Benchmark)
   * [Future work](#Future-work)
   * [Proof of cluster clean up](#Proof-of-cluster-clean-up)
   * [Citation](#Citation)
   * [References](#References)

***
## Overview
Project Name : Creating and optimizing Ml pipeline . 
the meain goal is to find the best model accuracy after a benchmark between two methods "Azure SDK hyperparameter"  and "Auto ML using ML" studion in AZURE paas . 
> Worksapace :

config.json was uploaded to the root file so the workspace could be created from the config informations available in the json file 

>  Cluster :

the cluster created wfor this project "STANDARD_D2_V2" witn 4 maximum nodes  

> Dataset :
- The data set used : bankmarketing.csv  - Tabular Dataset from this [set](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv)

- hyperparameters optimization tehnics :  HyperDrive.Azure AutoML 

>> _Step 1_ Creating an environement :

to avoid bugs with packages versions we create a new envirnment to train the model with the two methods 
the python version 3.6.9 installed in the enivronement also the packages : 
 - scikit-learn==0.24.1
  - xgboost==0.90
  - the environement named "logireg_env" the dependencies was stored in yml image 'conda_dependencies.yml ' and the environment was registred to our workspace ="udacity-project".
  

>> _Step 2_: Training the Model :

using the script (train.py) train the model with a logistic regression (Scikit-learn package ).

>. _Step 3_:  Model Tuning :

using ScriptRunConfig class to instantiate the configuration that will be used for tunig as well as the module HyperDrive to reach the best hyperparameters from the obtained model in the _Step 2_ .

>_Step 4_:  Benchmarking between the two methods results. 

***
## Summary
Results :
- Azure SDK hyperparameter : 
     - accuracy = 0.916843.

- AutoML :
    - accuracy =  0.91569 best algorithm "VotingEnsemble".

given the two methods the best model was provided by the Azure SDK hyperparameter with hyperdrive . 

## Hyperparameters 
**in the step-3- we used Hyperparameters for mode tuning . 
>>RandomParameterSampling:
this algorithm is more efficient and faster by the option of early termination in case when the model didn't improve than we avoid a high budget for the time spent in traing  .

>>Parameter sampler:

I specified the parameter sampler as such:
>>the hyperparameter '_C_' :  the Regularization
>>the hyperparameter '--max_iter':number of iterations.
```
ps = RandomParameterSampling(
    {
        '--C' : choice(0.001,0.01,0.1,1,10,20,50,100,200,500,1000),
        '--max_iter': choice(50,100,200,300)
    }
)
```
>> stopping policy: 

the early stopping policy was  used to automatically terminate  the training in case that the model is poorly performing after iterations 
>> _BanditPolicy_ : the policy terminat a run if it doesn't fall within the slack ratio or slack amount of the evaluation metric with respect to the best performing run will be terminated . concequently keeps executing only the best performing runs until they finish .

```
policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1)
```
>> _evaluation_interval_: (optional) it's the frequency for applying the policy. counts as one interval for each training script logs .

>> _slack_factor_: The ratio for the slack allowed with respect to the best performing training run

>configuration using ScriptRunConfig  :

```
src = ScriptRunConfig(source_directory='./',
                      script='train.py',
                      compute_target=compute_target,
                      environment=logireg_env)
```
- The Sklearn modlule was deprecated by the fact we used the new Class ScriptRunConfig to instantiate the configuration that will be used by the Hyperdrive tuner .         
we specified the scipt for training 'train.py' the comupte target the cluster that should be used for the training and the environment 'logireg_env' already created and registred with our workspace .
 
>Hyeperdrive for tuning : 
```
hyperdrive_config = HyperDriveConfig(run_config=src,
                                     policy=policy,
                                     hyperparameter_sampling=param_sampling, 
                                     primary_metric_name='Accuracy',
                                     primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                                     max_total_runs=16,
                                     max_concurrent_runs=8)
```
in the Hyeperdrive tuner we used the arguments for reaching the best model .
the policy , the goal metric to reach the maimum model accuray .
concurent runs the algorithm will be launched in 4 nodes . 

## AutoML
Model and hyperparameters :

```
automl_config = AutoMLConfig(
    compute_target = compute_target,
    experiment_timeout_minutes=15,
    task='classification',
    primary_metric='accuracy',
    training_data=ds,
    label_column_name='y',
    enable_onnx_compatible_models=True,
    n_cross_validations=2)
```
>> _experiment_timeout_minutes=15_ : exit criterion to define how long, in minutes, the experiment should continue to run (15 minutes ) 


>> _task='classification'_

defines the experiment type which in this case is classification.

> _primary_metric='accuracy'_ : primary metric accuracy 


> _enable_onnx_compatible_models=True_

set to 'True' using ONNX Algorithm the  Open Neural Network Exchange  .

> _n_cross_validations=2_ : 

This parameter sets how many cross validations to perform, based on the same number of folds (number of subsets). As one cross-validation could result in overfit, in my code I chose 2 folds for cross-validation; thus the metrics are calculated with the average of the 2 validation metrics.

## Benchmark 
**Comparison of the two models and their performance. Differences in accuracy & architecture - comments**


>> SDK HyperDrive Model 

- Accuracy : 0.9168437025796662 | --C : 500 | --max_iter :100

>.  AutoML Model ML STUDIO 

- Accuracy : 0.915690440060698| AUC_weighted : 0.9466849800575805 | Best Algortithm : VotingEnsemble |

***

## Future work

>> Imbalenced data : 

the model was trained more on positive majority so it would favorite the prediction of positive class .
there is two solutions :
>.underbalance the majority class :

so we obtain a balanced data in term of class target distribution  the cons of this technic is to lose some observations that we may need to train our model .(the more sample size we have the more accuracy we could get from the training )

>> Oversampling the minority SMOTE technique :

we use the SMOTE technique to create a syntetic version of observations similar to the minority and in this case we will have a full balanced sample .
(i prefer the Oversampling technic ) 
Smote techning : 
https://towardsdatascience.com/5-smote-techniques-for-oversampling-your-imbalance-data-b8155bdbe2b5

> Traing the AUTOML with other Hyperparameters : 

the AUTOML gave a good performance in term of "AUC_weighted = 0.9466 "
in my point of view , if we try AUTOML with a balanced data and other tuning Hyperparameters such as _RandomParameterSampling_ for an exhaustive research 
it may take time and consume more budget but will likly provide more accurate results  depends on the business goals (Accuracy Vs Budget) .

the AUTOML is known with the availability of trainig the models with sampling (cross-validation) that will automaticaly avoid the overfitting and improve the model performance with the least effort rather than customizing piplines using SDK and packages such us SKlearn.  
***

## Proof of cluster clean up
compute_target.delete() executed 

***
## Citation

[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014

***
## References
 - Example using Hyperdrive 
 https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/machine-learning-pipelines/intro-to-pipelines/aml-pipelines-parameter-tuning-with-hyperdrive.ipynb
 - Microsoft docs for model optimization .
 https://docs.microsoft.com/en-us/python/api/overview/azure/ml/install?view=azure-ml-py
 
 https://docs.microsoft.com/en-us/answers/questions/248696/using-estimator-or-sciptrunconfig-for-pipeline-wit.html
  https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-first-experiment-automated-ml
 https://docs.microsoft.com/en-us/azure/machine-learning/studio-module-reference/tune-model-hyperparameters
 - AutoML 
 https://docs.microsoft.com/en-us/python/api/azureml-train-automl-client/azureml.train.automl.automlconfig.automlconfig?view=azure-ml-py
 
- Sklearn - logistic regression 
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- randomparametersampling
https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.randomparametersampling?view=azure-ml-py
- banditpolicy 
https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.banditpolicy?view=azure-ml-py
- Cross validation 
https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cross-validation-data-splits

- ONNX
https://docs.microsoft.com/en-us/azure/machine-learning/concept-onnx

- Bank Marketing, UCI Dataset: [Original source of data](https://www.kaggle.com/henriqueyamahata/bank-marketing)


