import torch
import numpy as np
import torchvision
from torch import optim
import torch.nn.functional as F

from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score



from matplotlib import pyplot as plt
import sys
#sys.path.append('/Volumes/depot/iot4agrs/data/students/Alim/research_code') # this works
sys.path.append('..')
from vectorization_scripts.read_purnima_features import get_svr_features


# Note on imports: this also works:
# sys.path.append('..')
# Why do both work. How can I not use sys.path.append('..') or sys.path.append('long_folder_struct')? 

# If I want to print out all the paths python is reading from, use this:
# for i in sys.path:
#     print(i)

# if I do:
# from vectorization_scripts import read_purnima_features
# then,
# from read_purnima_features import get_svr_features
# I will get an import error. However:
# from vectorization_scripts.read_purnima_features import get_svr_features works. Why??

def lstm():
    input_size = 10
    hidden_size = 5

    num_layers=3
    lstm_model = torch.nn.LSTM(input_size, hidden_size, num_layers)

    random_data = torch.rand((1, 3, 10))

    output = lstm_model(random_data)
    print(output[0].shape)
    print((output[1][0].shape))

    return 0


def svr():
    """
    This aims to replicate the Support Vector Regerssion from Purnima's paper. 

    
    """
    df = get_svr_features(debug=False, data_path = 'hips_2021')
    print('list of features', df.columns)

    df = df.drop(columns=['date'])
    n_samples = df.shape[0]
    n_features = df.shape[1] - 1 # subtract one because we are storing the date, but the date isn't a predictor in SVR. It is in the 
    # dataframe to provide context as to which date observations are from.
    rng = np.random.RandomState(0) # generate random state for reproducibility
    print('this is rng', rng)
    
    train, test = train_test_split(df, test_size=0.2)
    print('THIS IS TRAIN\n')
    print(train)

    scaler = StandardScaler() # instatiate object to mean center data and scale it by 1 / std deviation
    scaler.fit_transform()

    print('THIS IS TEST\n')
    print(test)
    # get ground truth:
    # divide up for train and test
    y = train['LAI'].values
    print('shape of GT:', y.shape)

    # get input features
    X = train.iloc[:, 1:-1].values
    # n_samples, n_features
    print(X.shape)
    print('this is X', X)
    
    print(X.shape)

    
    print(type(train))

    model = SVR(C=1.0, epsilon=0.2)
    #regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
    model.fit(X, y)

    print('done with fitting training data')
   



    # prediction:
    print(type(test))
    test_features = test.iloc[:, 1:-1]
    #test = test.reshape(1,5)
    print(test_features)
    out = model.predict(test_features)
    print(out)

    # plt.scatter(X,y, color='red', label='train')
    # plt.scatter(test, out, color= 'blue', label = 'test')
    # plt.xlabel('predictor')
    # plt.ylabel('LAI')
    # plt.title('SVR on Extracted Feature')
    # plt.legend()
    # plt.savefig('test.jpg')

    # model metrics:
    print('test is:', test_features)
    print('pred is:', out)
    score = r2_score(y_true = test['LAI'].values, y_pred = out)
    print('r2: ', score)



if __name__ == "__main__":
    svr()