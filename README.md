# research_code
Hyperspectral and LiDAR data processing. ML and deep learning models to do time series predictions. 

Models:
Support Vector Regression to predict LAI based on extracted hyperspectral and LiDAR features.

Many-to-many LSTM to predict LAI at every timestep. 

How to use SVR model:
call svr() with cv_stratify True means we will do 10 fold cross validation and make sure we have an equal proportion of hybrid and inbreds as is reflected in our existing dataset in our train and test splits.
