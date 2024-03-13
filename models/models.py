import numpy as np
import torch
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
from dataloading_scripts.read_purnima_features import get_svr_features


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



class LSTM_addition_based(nn.Module):
    """
    In an LSTM, hidden state is for immediate, short term memory while cell state is for 
    long term memory.

    Equations from torch documentation:
    input_gate = sigmoid(Wii @ xt + bii + Whi @ ht-1 + bhi) = it                        #A
    forget_gate = sigmoid(Wif @ xt + bif + Whf @ ht-1 + bhf) = ft                       #B
    cell_gate = tanh(Wig @ xt + big + Whg @ ht-1 + bhg) = gt                            #C
    output_gate = sigmoid(Wio @ xt + bio + Who @ ht-1 + bho) = ot                       #D
    cell_state = ft * ct-1 + it * gt = ct                                               #E
    hidden_state = ot * tanh(ct)                                                        #F

    but, an LSTM paper from Google (https://arxiv.org/pdf/1402.1128.pdf), has some more details in the 
    equations. Namely,

    input_gate =  sigmoid(Wix @ xt + Wim @ mt-1 + Wic @ ct-1 + bi) = it                 #G
    forget_gate = sigmoid(Wfx @ xt + Wmf @ mt-1 + Wcf @ ct - 1 + bf) = ft               #H
    cell_activation_vector = ft * ct-1 + it * tanh(Wcx @ xt + Wcm @ mt-1 + bc) = ct     #I
    Note, this cell_activation_vector is effectively the cell_state. The torch and
    LSTM paper terminology is differing. 

    output_gate = sigmoid(Wox @ xt + Wom @ mt-1 + Woc @ ct-1 + bo) = ot                 #J
    cell_output_activation_vetor = ot * tanh(ct) = mt . Side note: this looks exactly  #K 
    like the equation for hidden_state in the torch documentation. So, I think we should
    be able to infer that hidden_state is the same as cell_output_activation_vector in 
    this LSTM paper.
    output = Wym @ mt + by                                                              #L

    Question: Does the pytorch documentation combine the cell state and memory state into 
    one "hidden" state?

    Yet another good resource on LSTMs is this blog: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
    I have seen it referenced by both Dr. Bouman and Dr. Inouye. I see it has good details about the mathematics.
    However, one diffrence in this blog's equations from the other 2 sets of equations above is the input features
    are concatenated, not added. This changes the dimensions of all the weight matrices.

    """
    def __init__(self, input_size, hidden_size, cell_size):
        """
        Using matrix notation from LSTM paper referenced above.
        Note, that because ht = ot * tanh(ct), where * denotes element wise multiplication 
        (Hadamard product), the cell state size and hidden size state must be the same.
        """
        super(LSTM_addition_based, self).__init__()
        self.input_size = input_size # size of xt
        self.hidden_size = hidden_size # size of ht
        self.cell_size = cell_size # size of ct
        #assert(hidden_size == cell_size, "hidden and cell size must be the same...")

        # learnable matrices for input gate:
        self.Wix = nn.Linear(self.input_size, self.hidden_size)
        self.Wih = nn.Linear(self.hidden_size, self.hidden_size)
        self.Wic = nn.Linear(self.cell_size, self.hidden_size)

        # learnable matrices for forget gate:
        self.Wfx = nn.Linear(self.input_size, self.hidden_size)
        self.Wfh = nn.Linear(self.hidden_size, self.hidden_size)
        self.Wfc = nn.Linear(self.cell_size, self.hidden_size)

        # learnable matrices for cell state / cell activation vector:
        self.Wcx = nn.Linear(self.input_size, self.hidden_size)
        self.Wch = nn.Linear(self.input_size, self.hidden_size)
        
        # learnable matrices for output gate
        self.Wox = nn.Linear(self.input_size, self.hidden_size)
        self.Woh = nn.Linear(self.hidden_size, self.hidden_size)
        self.Woc = nn.Linear(self.cell_size, self.hidden_size)
        
        # learnable matrices for output content:
        self.Wy = nn.Linear(self.input_size, self.hidden_size)
        
        # activations
        self.sigmoid = nn.Sigmoid()
        
        self.tanh = nn.Tanh()
        
    def forward(self,xt, ht_minus_1, ct_minus_1):
        input_gate = self.sigmoid(self.Wix(xt) +  self.Wih(ht_minus_1) + self.Wic(ct_minus_1))
        
        forget_gate = self.sigmoid(self.Wfx(xt) + self.Wfh(ht_minus_1) + self.Wfc(ct_minus_1))

        ct = forget_gate * ct_minus_1 + input_gate * self.tanh(self.Wcx(xt) + self.Wch(ht_minus_1))

        output_gate = self.sigmoid(self.Wox(xt) + self.Woh(ht_minus_1) + self.Woc(ct_minus_1))
        ht = output_gate * self.tanh(ct) # output gate modulates the amount of memory content exposure.
        # memory content exposure is in ct
        
        output = self.Wy(ht)

        return output, ht, ct

class LSTM_concat_based(nn.Module):
    """
    This implemetation of an LSTM is based on concatenation of input (xt) and hidden (ht) vectors.

    Details of this implementation can be found in:
    https://colah.github.io/posts/2015-08-Understanding-LSTMs/
    I have seen it referenced by both Dr. Bouman and Dr. Inouye. I see it has good details about the mathematics.
    However, one diffrence in this blog's equations from the other 2 sets of equations above is the input features
    are concatenated, not added. This changes the dimensions of all the weight matrices.

    Forget gate:
    Sigmoid helps us understand how much to "remember" (on a scale of 0 to 1) from the input + hidden state.

    Input gate:
    Sigmoid helps us understand how much to "forget" on scale from 0 to 1.
    """
    def __init__(self, input_size, hidden_size, cell_size):
        """
        Using matrix notation from LSTM paper referenced above.
        Note, that because ht = ot * tanh(ct), where * denotes element wise multiplication 
        (Hadamard product), the cell state size and hidden size state must be the same.
        """
        super(LSTM_concat_based, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_size = cell_size
        # first step is to decide which information to throw away from the cell state.
        # this is done by ft, the forget gate layer. we will learn a 
        # linear transformation, Wf, for this:
        self.Wf = nn.Linear(input_size + hidden_size, cell_size)
        self.sigmoid = nn.Sigmoid()

        # second step is to decide what we will store in the cell state. First, the input gate
        # layer decides which values we will update, then, a tanh layer creates 
        # new candidate values that should be added to create an update to the 
        # cell state:
        self.Wi = nn.Linear(input_size + hidden_size, cell_size)
        self.Wc = nn.Linear(input_size + hidden_size, cell_size)
        self.tanh = nn.Tanh()

        # To actually update the cell state, we element-wise multiply the forget gate
        # by the previous cell state (i.e, tell the previous cell what to forget)
        # and we also add the element wise product of the input gate's output 
        # to the candidate cell state. It's like saying "forget these values" and "add these!".
        # (at least in my interpretation of the math :)) . The actual math for this will
        # be done in the forward operation. There isn't anything learnable here.

        # Then, we decide what to actually output as the final hidden layer.
        # It'll be based on the updated cell state, but controlled by the output gate.
        self.Wo = nn.Linear(input_size + hidden_size, cell_size)

        # the hidden state will be the result of a sigmoid applied to Wo,
        # then element wise multiplied by a tanh applied to ct! 

    def forward(self, xt, ct_minus_1, ht_minus_1):
        # print('shape of xt:', xt.shape)
        # print('shape of ct_minus_1:', ct_minus_1.shape)
        # print('shape of ht_minus_1', ht_minus_1.shape)
        feature_concat = torch.concat([ht_minus_1, xt])
        feature_concat = feature_concat.to(torch.float32)

        # forget gate - what we want to forget from cell state
        ft = self.sigmoid(self.Wf(feature_concat))

        # input gate - what we want to add to cell state
        it = self.sigmoid(self.Wi(feature_concat))

        # candidate cell update
        ct_candidate = self.tanh(self.Wc(feature_concat))

        # update cell state
        ct = ft * ct_minus_1 + it * ct_candidate

        # output gate - what we keep from the candidate cell state:
        ot = self.sigmoid(self.Wo(feature_concat))

        # update hidden state:
        ht = ot * self.tanh(ct)

        return ct, ht


class RNN(nn.Module):
    """
    This RNN class wraps an instantiated LSTM class.
    """
    def __init__(self, batch_size : int, concat_based_LSTM : bool, addition_based_LSTM : bool, debug=False,
                 hidden_size = 100, cell_size = 100):
        super(RNN, self).__init__()
        self.debug=debug
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.cell_size = cell_size

        if addition_based_LSTM:
            self.lstm = LSTM_addition_based(input_size= 17, hidden_size= hidden_size, cell_size=cell_size)
            print('instantiating addition based LSTM')
        if concat_based_LSTM:
            self.lstm = LSTM_concat_based(input_size=17, hidden_size = hidden_size, cell_size=cell_size)
            print('instantiating concat based LSTM')

        self.fc = nn.Linear(self.lstm.hidden_size, 1) # fc layer to do prediction of LAI at every step.
        self.relu = nn.ReLU() # non-linearity after each LSTM cell output

    def forward(self, timeseries):
        # timeseries will have to come from the pytorch dataloader. It is a series of xt where t = {0, 1, ... k-1, k}, where k 
        # is the number of observations in the time series.
        predictions_in_series = torch.empty((timeseries.shape[0], self.batch_size), device = "mps", dtype=torch.float32) # length of timeseries x batch_size
        for n, xt in enumerate(timeseries):
            if n == 0:
                if self.debug:
                    print('xt shape:', xt.shape)
                    print('lstm.hidden_size', self.lstm.hidden_size)
                ht_minus_1 = torch.zeros((self.lstm.hidden_size), device = "mps", dtype = torch.float32) # size 100
                ct_minus_1 = torch.zeros((self.lstm.cell_size), device = "mps", dtype= torch.float32) # size 100

                ht, ct = self.lstm(xt, ct_minus_1, ht_minus_1)
                pred = self.fc(self.relu(ht))
                predictions_in_series[n] = pred
            else:
                ht, ct = self.lstm(xt, ht, ct)
                pred = self.fc(self.relu(ht))
                predictions_in_series[n] = pred
        
        return ct, ht, pred, predictions_in_series # prediction is final prediciton. predictions_in_series is every timestep's prediction.


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer).__init__()
        self.some = 'placeholder'
        # consider using class from pytorch: https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html

class HyperspectralAE(nn.Module):
    """
    Class for hyperspectral autoencoder. Used for a feature extractor. 
    """
    def __init__(self, input_channels, height, width, debug=False):
        super(HyperspectralAE, self).__init__()
        self.input_channels =input_channels
        self.height = height
        self.width = width
        self.debug = debug
        #self.conv_out_shape_row = ((in_rows + 2*p - k) / stride) + 1
        #self.conv_out_shape_col = ((in_cols + 2*p - k) / stride) + 1
        # input data is 130 rows by 42 cols. Stride is 1, kernel is 5, padding is 0. Therefore, after the first conv:
        # row shape is (130 + 0 - 5 / 1 ) + 1 = 126
        # col shape is (42 + 0 - 5) / 1) + 1  = 38
        
        self.conv1 = nn.Conv2d(in_channels = input_channels, out_channels = int(input_channels / 2), kernel_size= 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels= int(input_channels / 2), out_channels = int(input_channels / 4), kernel_size=3)
        self.maxpool = nn.MaxPool2d(kernel_size=2, return_indices=True)
        self.maxunpool = nn.MaxUnpool2d(kernel_size = 2)

        self.conv_output_dims = (input_channels // 2, height - 3 + 1, width - 3 + 1)
        self.conv_output_dims_2 = (self.conv_output_dims[0] // 2, self.conv_output_dims[1] + 0 - 3 + 1, self.conv_output_dims[2] + 0 - 3 + 1)
        self.conv_output_dims_2_total_dims = int(self.conv_output_dims_2[0] * self.conv_output_dims_2[1] * self.conv_output_dims_2[2])
        print(self.conv_output_dims_2_total_dims, 'total_dims!') # 162,792
        self.fc = nn.Linear(9486, 1000)
        self.fc2 = nn.Linear(1000, 500)

        self.fc3 = nn.Linear(500, 1000)
        self.fc4 = nn.Linear(1000, 9486)


        self.transpose_conv1 = nn.ConvTranspose2d(in_channels=int(input_channels/4), out_channels=int(input_channels/2), kernel_size=3)
        self.transpose_conv2 = nn.ConvTranspose2d(in_channels=int(input_channels/2), out_channels=136, kernel_size=3)
        # result is row/col x kernel - 1
        if debug:
            print('all weights initialized...')
        
    def encoder(self, x):
        # code to push to latent dimension
        x = self.conv1(x)
        if self.debug:
            print(self.conv1.weight.shape)
            print('after first conv:', x.shape, 'is it 272 x 126 x 17??')
        x, ind_mp1 = self.maxpool(x)
        x = self.relu(x)
        x = self.conv2(x)
        x, ind_mp2 = self.maxpool(x)
        x = self.relu(x)
        if self.debug:
            print('shape after second conv:', x.shape)
            print('indices of maxpool', ind_mp1, ind_mp2)
        x = torch.flatten(x)
        if self.debug:
            print('at flatten')
            print(x.shape)
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.debug:
            print('end of encoder shape:' , x.shape)
        return x, ind_mp1, ind_mp2

    def decoder(self, x, ind_mp1, ind_mp2):

        # code to push back up to image size
        
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        if self.debug:
            print('in decoder - shp is', x.shape)
        
        
        # reshape vector back to image shape to do transpose conv ops:
        x = x.reshape(34, 31, 9)
        x = self.maxunpool(x, ind_mp2)

        x = self.transpose_conv1(x)
        x = self.relu(x)

        x = self.maxunpool(x, ind_mp1)
        x = self.transpose_conv2(x)
        if self.debug:
            print('at end of network w/ shape', x.shape)

        return x

    def forward(self, x):
        x, ind_mp1, ind_mp2 = self.encoder(x)
        x = self.decoder(x, ind_mp1, ind_mp2)
        return x 

class LiDARAE(nn.Module):
    def __init__(self):
        super(LiDARAE, self).__init__()

        latent_dim = 32

        # encoder ops
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, latent_dim)

        # decoder ops:
        self.fc4 = nn.Linear(latent_dim, 64)
        self.fc5 = nn.Linear(64, 128)
        self.fc6 = nn.Linear(128,3)

        # activations
        self.relu = nn.ReLU()


    def encoder(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x

    def decoder(self, x):
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x) 
        x = self.fc6(x)

        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class statistical_model():
    def __init__(self, debug=False, produce_plot=False, produce_metrics=True, cross_validation=False,
                 field_id = 'hips_2021', cv_stratify=False, groups=False, verbose_stratification_and_group_data=True,
                 grid_search=False, kernel='rbf', C = 1.0, epsilon=0.2):

        self.field_id = field_id
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        

    def svr(self, debug=False, produce_plot=False, produce_metrics=True, cross_validation=False, 
            cv_stratify=False, groups=False, verbose_stratification_and_group_data=True,
            grid_search=False):
        """
        This aims to replicate the Support Vector Regerssion from Purnima's paper. 

        The hyperparams arg takes in kernel, gamma, and epsilon values
        """
        np.random.seed(0) # set seed for reproducibility
        
        # instantiate SVR model 
        model = SVR(kernel = 'rbf', C=1.0, epsilon=0.2)

        df = get_svr_features(debug=False, data_path = self.field_id)
        if debug:
            print('list of features', df.columns)
        
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
            X = df.iloc[:, 1:-5].values # remove plot number, LAI, date, hyrid_or_inbred, pedigree, and
            # nitrogen treatment from predictors.
            if debug:
                print('this is X', X, '\n', 'this is the shape of X', X.shape)
                
            # fit and transform input feature data.
            X_fit_transformed = scaler.fit_transform(X) # correct this later - we should not scalar transform on whole dataset!!!
            if debug:
                print('this is X AFTER TRANSFORMATION', X)
                print('mean is: ', scaler.mean_)
                print('std dev is: ', scaler.var_)

        # if grid_search:
        #     print('in grid search:')

        #     # prep data:
        #     X = df.iloc[:, 1:-5] # remove plot number, LAI, date, hyrid_or_inbred, pedigree, and
        #     # nitrogen treatment from predictors.

        #     if field_id != '2022_f54': 
        #         groups = df['pedigree']
        #         y = df['hybrid_or_inbred'] # used for stratification
        #     else:
        #         groups = df['nitrogen_treatment']
        #         y = df['date']

        #     # define grid search hyperparameters
        #     grid_params = { 'C': [0.01, 0.1, 0.5], 'kernel': ['linear', 'rbf'], 'gamma': [1, 2]}
            
        #     # define model
        #     model = SVR()

        #     sss = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=1)
        #     grid_search = GridSearchCV(estimator = model, param_grid = grid_params, cv=sss)
        #     print('made it here')
        #     print('X is ', X.shape, 'y is', y.shape, 'groups is', groups.shape)
        #     print(groups)
        #     grid_search.fit(X, df['LAI'], groups=groups) # fits over all model hyper parameter configurations outlined in grid_params
        #     print(f'The best parameters are {grid_search.best_params_}')
        #     print(f'The best accuracy on the training data is {grid_search.score(X, y)}')


        if cv_stratify:
            sss = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=1)

            print('shapes of split!!!')
            print('shape on input features:', df.iloc[:, 1:-5].shape)
            print(df['hybrid_or_inbred'].shape)
            print('unique vals of the column hybrid_or_inbred', df['hybrid_or_inbred'].unique())

            rmse_dict = {}
            r_2_dict = {}
            stratification_dict = {}
            fig, axs = plt.subplots(2, 5, figsize=(20, 12), gridspec_kw={'top': 0.93})
            plt.subplots_adjust(hspace=0.1, wspace=0.3)

            X = df.iloc[:, 1:-5] # remove plot number, LAI, date, hyrid_or_inbred, pedigree, and
            # nitrogen treatment from predictors.

            if self.field_id != '2022_f54': 
                groups = df['pedigree']
                y = df['hybrid_or_inbred'] # used for stratification
            else:
                groups = df['nitrogen_treatment']
                y = df['date'] 

            for i, (train_indices, test_indices) in enumerate(sss.split(df.iloc[:, 1:-5], y, groups)):
                # train_indices and test_indices are arrays of the train and test indices for a split (fold).
                # they will not necessarily be the same length, untless the train/test sizes are 0.5/0/5.
                # stratification is done based on the y labels (sss.split(X, y, group))
                
                # do scalar transform on train_indices:
                scaler_k_fold = StandardScaler()
                X_fit_transformed_kth_fold = scaler_k_fold.fit_transform(df.iloc[train_indices, 1:-5])
                
                if debug:
                    print('mean is', scaler_k_fold.mean_, ' \n', 'var is:',  scaler_k_fold.var_)
                    print('this is X AFTER TRANSFORMATION', X_fit_transformed_kth_fold)
                    print('mean is: ', scaler_k_fold.mean_)
                    print('std dev is: ', scaler_k_fold.var_)
                
                if verbose_stratification_and_group_data:
                    print('Info about the whole dataset:')
                    print(df.loc[:,'hybrid_or_inbred'].value_counts())
                    
                    print('Information about the train fold:\n')
                    print(df.loc[train_indices, 'hybrid_or_inbred'].value_counts())
                    print(df.loc[train_indices, 'pedigree'].unique())

                    print('Info about the test split:\n')
                    print(df.loc[test_indices, 'hybrid_or_inbred'].value_counts())
                    print(df.loc[test_indices, 'pedigree'].unique())

                    intersect_result = set(df.loc[train_indices, 'pedigree'].unique().tolist()).intersection(set(df.loc[test_indices, 'pedigree'].unique().tolist()))
                    print('intersection result:', intersect_result)

                    stratification_dict['fold_' + str(i)] = (df.loc[train_indices, 'hybrid_or_inbred'].value_counts(),
                                                            df.loc[test_indices, 'hybrid_or_inbred'].value_counts(),
                                                            intersect_result)

                # fit model on training data:
                model_k = SVR(kernel = self.kernel, C = self.C, epsilon = self.epsilon)
                # above, we create SVR model based on the hyperparams list
                model_k.fit(X_fit_transformed_kth_fold, df.loc[train_indices, 'LAI'])

                if debug:
                    print(f"Fold {i}:")
                    print(f"  Train: index={train_indices}")
                    print(f"  Test:  index={test_indices}")
                    print('train data shape:\n', X_fit_transformed[train_indices, :].shape)
                    print('test data shape:', df.loc[test_indices, 'LAI'].shape)

                # test model on k-th test fold:
                # scale test data based on transform for training data
                test_transformed = scaler_k_fold.transform(df.iloc[test_indices, 1:-5])
                out = model_k.predict(test_transformed)
                y_true = df.loc[test_indices, 'LAI'].values
                score_for_fold = r2_score(y_true = y_true, y_pred = out)
                r_2_dict['fold_' + str(i)] = score_for_fold
                rmse_for_fold = mean_squared_error(y_true = df.loc[test_indices, 'LAI'].values, y_pred = out , squared=False)
                rmse_dict['fold_' + str(i)] = rmse_for_fold
                if debug:
                    print('R2 score:', score_for_fold)
                    print('RMSE is:', rmse_for_fold)



                if produce_plot:
                    # a plot of 1:1 prediction to GT values for each fold. Since we are doing 10 fold validation,
                    # we will have a 5 x 2 layout of subplots. Each of those plots will have the predictions for their fold.
                    ax = axs[i%2, i//2]  # Calculate subplot index based on loop variable
                    ax.scatter(y_true, out, color = 'blue', label='Predictions')
                    ax.plot([0,7], [0,7], color = 'black', label= '1:1 line')
                    ax.set_xlabel('Ground Truth LAI')
                    ax.set_ylabel('Predicted LAI')
                    ax.set_title('GT vs Predicted LAI on\n' + self.field_id + ', for fold ' + str(i+1))
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

                        fig.suptitle('Stratified ' + str(sss.n_splits) + ' Fold Validation for '  + self.field_id + 
                                    '\n' + 'Average RMSE: ' + str(average_rmse) + '\n' +
                                    'Average R^2: ' + str(average_r_2), fontsize=16, fontweight='bold')
                        fig.savefig('Stratified_K_Fold_' + self.field_id)

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
            print("Stratification info:\n", stratification_dict)

            return rmse_dict, r_2_dict, stratification_dict, average_r_2, average_rmse


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
                plt.title('Ground Truth vs Predicted LAI on ' + self.field_id + ', CV:' + str(cross_validation))
                plt.xlim(left = 0, right = 7)
                plt.ylim(bottom = 0, top = 7)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.legend(loc='lower right')
                plt.savefig('Ground_truth_vs_predicted_LAI_' + self.field_id + '.jpg')


if __name__ == "__main__":
    # svr(debug=False, produce_plot=True, cross_validation=False, field_id = 'hips_2021')
    # svr(debug=False, produce_plot=True, cross_validation=False, field_id = 'hips_2021', cv_stratify=True, groups=False)
    # svr(debug=False, produce_plot=True, cross_validation=False, field_id = 'hips_2022', cv_stratify=True, groups=False)
    # svr(debug=False, produce_plot=True, cross_validation=False, field_id = 'hips_both_years', cv_stratify=True, groups=False)
    # svr(debug=False, produce_plot=True, cross_validation=False, field_id = 'hips_2021', cv_stratify=True, groups=False)
    # svr(debug=False, produce_plot=True, cross_validation=False, field_id = 'hips_both_years', cv_stratify=True, groups=False)
    svr_model = statistical_model(field_id='hips_2022')
    svr_model.svr(debug=False, produce_plot=True, cross_validation=False, cv_stratify=True, groups=False,
        grid_search=True)






    