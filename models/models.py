import torch
import numpy as np
import torchvision
from torch import optim
import torch.nn.functional as F
import torch.nn as nn


from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, StratifiedGroupKFold


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

    class LSTM(nn.Module):
        """
        In an LSTM, hidden state is for immediate, short term memory while cell state is for 
        long term memory.

        
        """
        def __init__(self, input_size, hidden_size, cell_size):
            super(LSTM, self).__init__()
            self.input_size = input_size # size of xt
            self.hidden_size = hidden_size # size of ht
            self.cell_size = cell_size # size of ct
            
            # learnable matrices for output gate
            self.Wo = nn.Linear(self.input_size, self.hidden_size)
            self.Uo = nn.Linear(self.hidden_size, self.hidden_size)
            self.Vo = nn.Linear(self.cell_size, self.hidden_size)
            
            # learnable matrices for forget gate:
            self.Wf = nn.Linear(self.input_size, self.hidden_size)
            self.Uf = nn.Linear(self.hidden_size, self.hidden_size)
            self.Vf = nn.Linear(self.cell_size, self.hidden_size)
            
            # learnable matrices for input gate:
            self.Wi = nn.Linear(self.input_size, self.hidden_size)
            self.Ui = nn.Linear(self.hidden_size, self.hidden_size)
            self.Vi = nn.Linear(self.cell_size, self.hidden_size)
            
            # learnable matrices for new memory content:
            self.Wc = nn.Linear(self.input_size, self.hidden_size)
            
            # activations
            self.sigmoid = nn.Sigmoid()
            
            self.tanh = nn.Tanh()
            
        def forward(self, xt, ht_minus_1, ct):
            output_gate = self.sigmoid(self.Wo(xt) + self.Uo(ht_minus_1) + self.Vo(ct))
            ht = output_gate * self.tanh(ct) # output gate modulates the amount of memory content exposure.
            # memory content exposure is in ct
            
            # need to understand how to enforce diagonality on self.Vo? Do we just mask everything on non-diagonal 
            # to be zero?
            
            forget_gate = self.sigmoid(self.Wf(xt) + self.Uf(ht_minus_1) + self.Vf(ct_minus_1))
            
            input_gate = self.sigmoid(self.Wi(xt) +  self.Ui(ht_minus_1) + self.Vi(ct_minus_1))
            
            new_memory_content = self.tanh(self.Wc(xt) + self.Uc(ht_minus_1))
            
            ct_update = forget_gate * ct + input_gate * new_memory_content
            
            return ht, ct_update

    return 0


def svr(debug=False, produce_plot=False, produce_metrics=True, cross_validation=False, field_id = 'hips_2021', cv_stratify=False, groups=False):
    """
    This aims to replicate the Support Vector Regerssion from Purnima's paper. 
    """
    np.random.seed(0) # set seed for reproducibility
    
    # instantiate SVR model 
    model = SVR(kernel = 'rbf', C=1.0, epsilon=0.2)

    df = get_svr_features(debug=False, data_path = field_id)
    if debug:
        print('list of features', df.columns)

    df = df.drop(columns=['date']) # we will not need the date information as it is not a predictor
    # in the SVR model.
    print(df.columns)

    
    if cv_stratify==False:
        train, test = train_test_split(df, test_size=0.2, shuffle=True) # split data into train and test
        if debug:
            print('THIS IS TRAIN\n')
            print(train)
            print('THIS IS TEST\n')
            print(test)

        scaler = StandardScaler() # instatiate object to mean center data and scale it by 1 / std deviation
        # this will also save the parameters of the data transformation so we can apply the same transformation
        # to the test dataset. Note, this scalar object is for a regular train/tets split (i.e, when cv_stratify is false)

        # get ground truth data from training set. 
        y = df['LAI'].values
        if debug:
            print('this is y: ', y, '\n', 'this is shape of y: ', y.shape)

        # get input features from training set
        X = df.iloc[:, 1:-3].values # remove plot number and LAI from predictors.
        if debug:
            print('this is X', X, '\n', 'this is the shape of X', X.shape)
            
        # fit and transform input feature data.
        X_fit_transformed = scaler.fit_transform(X) # correct this later - we should not scalar transform on whole dataset!!!
        if debug:
            print('this is X AFTER TRANSFORMATION', X)
            print('mean is: ', scaler.mean_)
            print('std dev is: ', scaler.var_)

    if cv_stratify:
        sss = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=1)

        print('shapes of split!!!')
        print(df.iloc[:, 1:-1].shape)
        print(df['hybrid_or_inbred'].shape)
        print(df['hybrid_or_inbred'].unique())

        rmse_dict = {}
        r_2_dict = {}
        fig, axs = plt.subplots(5, 2, figsize=(12, 20), gridspec_kw={'top': 0.93})
        plt.subplots_adjust(hspace=0.3) 
        groups = df['pedigree']

        for i, (train_indices, test_indices) in enumerate(sss.split(df.iloc[:, 1:-3].values, df['hybrid_or_inbred'], groups)):
            # train_indices and test_indices are arrays of the train and test indices for a split (fold).
            # they will not necessarily be the same length, untless the train/test sizes are 0.5/0/5.
            
            # do scalar transform on train_indices:
            scaler_k_fold = StandardScaler()
            X_fit_transformed_kth_fold = scaler_k_fold.fit_transform(df.iloc[train_indices, 1:-3])
            print('mean is', scaler_k_fold.mean_, ' \n', 'var is:',  scaler_k_fold.var_)
            if debug:
                print('this is X AFTER TRANSFORMATION', X_fit_transformed_kth_fold)
                print('mean is: ', scaler_k_fold.mean_)
                print('std dev is: ', scaler_k_fold.var_)

            # fit model on training data:
            model_k = SVR(kernel = 'rbf', C=1.0, epsilon=0.2)
            model_k.fit(X_fit_transformed_kth_fold, df.loc[train_indices, 'LAI'])

            # to do: double check for stratification!!

            if debug:
                print(f"Fold {i}:")
                print(f"  Train: index={train_indices}")
                print(f"  Test:  index={test_indices}")
                print('train data shape:\n', X_fit_transformed[train_indices, :].shape)
                print('test data shape:', df.loc[test_indices, 'LAI'].shape)

            # test model on k-th test fold:
            # scale test data based on transform for training data
            test_transformed = scaler_k_fold.transform(df.iloc[test_indices, 1:-3])
            out = model_k.predict(test_transformed)
            y_true = df.loc[test_indices, 'LAI'].values
            score_for_fold = r2_score(y_true = y_true, y_pred = out)
            print('R2 score:', score_for_fold)
            r_2_dict['fold_' + str(i)] = score_for_fold
            rmse_for_fold = mean_squared_error(y_true = df.loc[test_indices, 'LAI'].values, y_pred = out , squared=False)
            print('RMSE is:', rmse_for_fold)
            rmse_dict['fold_' + str(i)] = rmse_for_fold

            if produce_plot:
                # a plot of 1:1 prediction to GT values for each fold. Since we are doing 10 fold validation,
                # we will have a 5 x 2 layout of subplots. Each of those plots will have the predictions for their fold.
                ax = axs[i//2, i%2]  # Calculate subplot index based on loop variable
                ax.scatter(y_true, out, color = 'blue', label='Predictions')
                ax.plot([0,7], [0,7], color = 'black', label= '1:1 line')
                ax.set_xlabel('Ground Truth LAI')
                ax.set_ylabel('Predicted LAI')
                ax.set_title('GT vs Predicted LAI on ' + field_id + ', for fold ' + str(i+1))
                ax.set_xlim(left = 0, right = 7)
                ax.set_ylim(bottom = 0, top = 7)
                ax.set_aspect('equal', adjustable='box')
                ax.legend(loc='lower right')
                ax.annotate(f'R^2 = {score_for_fold:.2f}', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=10)
                ax.annotate(f'RMSE = {rmse_for_fold:.2f}', xy=(0.05, 0.8), xycoords='axes fraction', fontsize=10)
                
                #save plot on final iteration, get summary statistics:
                if i == 9:
                    print('computing average scores...')
                    # Calculate the sum of all values in the dictionary
                    total_rmse = sum(rmse_dict.values())
                    average_rmse = round(total_rmse / sss.n_splits , 3)
                    total_r_2 = sum(r_2_dict.values())
                    average_r_2 = round(total_r_2 / sss.n_splits, 4)

                    print('saving plot...')

                    fig.suptitle('Stratified ' + str(sss.n_splits) + ' Fold Validation for '  + field_id + 
                                '\n' + 'Average RMSE: ' + str(average_rmse) + '\n' +
                                'Average R^2: ' + str(average_r_2), fontsize=16, fontweight='bold')
                    fig.savefig('Stratified_K_Fold_' + field_id)

        if debug:
            inb = 0
            hyb = 0
            for j in test_indices:
                print(df['hybrid_or_inbred'][j])
                if df['hybrid_or_inbred'][j] == 'hybrid':
                    hyb += 1
                else:
                    inb += 1

            print('total hybrid:', hyb, ' and total inbred:', inb)
        
        print("Done with cv_stratify...")
        print("RMSE and R^2 values from cv_stratify:\n", "RMSEs:", rmse_dict, "R^2s:", r_2_dict)


    # fit the model based on training data
    if cross_validation == False and cv_stratify == False:
        model.fit(X_fit_transformed, y) # just fit on entire train dataset

    if cross_validation:
        scores = cross_val_score(model, X_fit_transformed, y, cv=10) # default score method for SVR is R^2
        print('CV scores: ', scores)

        grid_params = { 'C': [0.01, 0.1, 0.5, 1.0], 'kernel': ['linear', 'poly', 'rbf'], 'gamma': [1, 2, 3, 4]}

        SVR_GS = GridSearchCV(estimator = SVR(), param_grid = grid_params, cv = 5) # Specifies the number of fractions for cross validation
        print('searching for best model...')
        SVR_GS.fit(X_fit_transformed, y) # fits over all model hyper parameter configurations outlined in grid_params
        print(f'The best parameters are {SVR_GS.best_params_}')
        print(f'The best accuracy on the training data is {SVR_GS.score(X_fit_transformed, y)}')
        
    if debug:
        print('done with fitting training data')
        print('starting predictions on test set')

    # prediction on test data:
    if cv_stratify == False:
        test_features = test.iloc[:, 1:-3].values # remove plot and LAI ground truth from test features.
        if debug:
            print('this is the test data: ', test_features, '\n', 'this is the test data size:', test_features.shape)

        print('transforming test data...')
        # transform test data input features with same transformation applied to training data
        test_features_transformed = scaler.transform(test_features)
        
        if cross_validation:
            out = SVR_GS.predict(test_features_transformed)
    
        if cross_validation == False:
            out = model.predict(test_features_transformed)
        print('predictions are: \n', out)
        #print('The best accuracy on the testing data using gridsearch is ',SVR_GS.score(test_features_transformed, test['LAI'].values))

        # model metrics and product plot when cv_stratify is False:
        if produce_metrics:
            print('Computing R^2 Score:')
            print('test size is:', test_features_transformed.shape)
            print('pred size is:', len(out))
            score = r2_score(y_true = test['LAI'].values, y_pred = out)
            print('r2: ', score)
            print('Computing RMSE')
            rmse = mean_squared_error(y_true = test['LAI'].values, y_pred = out , squared=False)
            print('RMSE: ', rmse)

        if produce_plot:
            print('Creating plot...')
            plt.scatter(test['LAI'].values, out, color= 'blue', label = 'Test Predictions')
            plt.plot([0, 7], [0, 7], color = 'black', label= '1:1 line' )
            plt.annotate(f'R^2 = {score:.2f}', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=10)
            plt.annotate(f'RMSE = {rmse:.2f}', xy=(0.05, 0.8), xycoords='axes fraction', fontsize=10)

            plt.xlabel('Ground Truth LAI')
            plt.ylabel('Predicted LAI')
            plt.title('Ground Truth vs Predicted LAI on ' + field_id + ', CV:' + str(cross_validation))
            plt.xlim(left = 0, right = 7)
            plt.ylim(bottom = 0, top = 7)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.legend(loc='lower right')
            plt.savefig('Ground_truth_vs_predicted_LAI_' + field_id + '.jpg')


if __name__ == "__main__":
    #svr(debug=False, produce_plot=True, cross_validation=False, field_id = 'hips_2021')
    # svr(debug=False, produce_plot=True, cross_validation=False, field_id = 'hips_2021', cv_stratify=True, groups=False)
    # svr(debug=False, produce_plot=True, cross_validation=False, field_id = 'hips_2022', cv_stratify=True, groups=False)
    # svr(debug=False, produce_plot=True, cross_validation=False, field_id = 'hips_both_years', cv_stratify=True, groups=False)
    svr(debug=False, produce_plot=True, cross_validation=False, field_id = '2022_f54', cv_stratify=True, groups=False)




    