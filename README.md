# Overview:
This repository contains all the code used to do prediction of LAI using UAV based hyperspectral and LiDAR observations.
Furthermore, weather features like precipitaion and growing degree days are also incorporated into the model.

There is a significant amount of pre-processing, which is found in the dataloading and processing folders. 

The acutal modeling code (for statistical and deep learning models) are found in the models folder. 

The goal for the LSTM and Transformer models is to predict LAI at each time point. 

Models:
Support Vector Regression to predict LAI based on extracted hyperspectral and LiDAR features.

Many-to-many LSTM to predict LAI at every timestep. 

How to use SVR model:
call statistical_model() with cv_stratify True means we will do 10 fold cross validation and make sure we have an equal proportion of hybrid and inbreds as is reflected in our existing dataset in our train and test splits.

How to use LSTM to predict LAI at every timestep:
Run train_and_test.py.
This will run the LSTM model on the traditionally extracted remote sensing features, and predict LSTM at every time point. 

How to use Transformer to predict LAI at every timestep:
... to do ...

### 
Add info about LSTM