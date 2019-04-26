# Machine Learning Nano Degree Capstone Project

## Project: AllState Claim Severity

This project is competition from kaggle 'AllState Claim Severity' predictive modelling challenge. Goal of this project is to predict severity 
of Claim based on Loss value. Predicting numeric Loss values from given set of Claim features is task of Regression problem in ML. So, there 
are several Supervised ML Regression techniques or approach which can be used as solution to this problem domain. 


## Install


This project requires **Python 3** with GPU Hardware accelerator(for enchanced performance) and the following Python libraries installed:


- [NumPy]

- [Pandas]
- [matplotlib]
- [scikit-learn]
- [seaborn]
- [keras]
- [xgboost]
- [scipy]
Code implementation for this project is done on google colab cloud environment. 
Clone repository from github(https://github.com/alathiya/AllState-Claim-Severity-Prediction) for detailed reports and full implementation. 


## Data

Train and Test data for this project can be referenced from kaggle competition[Allstate Claims Severity Repo](https://www.kaggle.com/c/allstate-claims-severity/data).
Train and Test 

has 131 features with 14 continuous and 116 categorical. Loss target value is given for each row in Train data for Supervise
Regression training. 'Id' Columns refers to unique row key for each claim row. 
**Features**

1) `Cont1`...`Cont14`: Represents 14 continuous numeric features; 

2) `Cat1`...`Cat116`: Represents 116 categorical features with type string; 
3) `loss`: Represents target numeric feature; 

4) `Id`: Represents unique row Id for each Claim row;



## Data Preprocessing and Visualization
Data visualization is done using seaborn heatmap and pandas scatter matrix to visualization correlation among features.
- [seaborn heatmaps](https://seaborn.pydata.org/generated/seaborn.heatmap.html)
- [pandas scatter matrix](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.plotting.scatter_matrix.html)  

Data Preprocessing is done using PCA to reduce dimensioanlity of data and transformation is done to log scale to achieve normal distribution.
- [Principal Component Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [Log transform](https://docs.scipy.org/doc/numpy/reference/generated/numpy.log.html)


## Implementation
Three ML approaches discussed and implemented in this project: 
- [DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
- [XGBRegressor](https://xgboost.readthedocs.io/en/latest/python/python_api.html)
- [Deep Neural Network using Keras](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)


## Model Evaluation and Validation
Model is tested on 10% of train data splitted from orginial train data in ratio of 9:1. Remaining 90% of train data is used for 
model training and validation.Metric used to measure performance of model is Mean Absolute error (MAE). Goal of all three ML approches 
implemented is to reduce MAE. MAE is average over difference between predicted and actual loss on test data.
Please note that test data provided in kaggle competition do not have true loss values so MAE cannot be computed on that but free form
visualization is done to understand Loss pattern predicted from test data and true values from train data.


## Results
MAE score from all three implementations are reported and final model is chosen with best MAE score.      


## Problem Domain
Some of the references for predictive modeling on Claim Severity:  
https://riskandinsurance.com/georgia-pacific/  
https://www.investopedia.com/terms/a/average-severity.asp  
https://www.casact.org/pubs/forum/05spforum/05spf215.pdf  