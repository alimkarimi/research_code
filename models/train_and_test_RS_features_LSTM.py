from models import RNN

import torch
import torch.nn as nn
from torch import optim

import sys
sys.path.append('..')
from dataloading_scripts.feature_dataloader import FeaturesDataset
from dataloading_scripts.read_purnima_features import get_svr_features

from matplotlib import pyplot as plt

import numpy as np

from sklearn.metrics import r2_score, mean_squared_error

epochs = 20
criterion = nn.MSELoss()
batch_size = 1

torch.manual_seed(0)
np.random.seed(0)

# instantiate model
rnn  = RNN(batch_size = batch_size, concat_based_LSTM = True, addition_based_LSTM = False,
           hidden_size = 100, cell_size = 100) # lstm gets instantiated inside RNN class.
rnn = rnn.to(torch.float32)

# instantiate optimizer
optimizer = torch.optim.Adam(rnn.parameters(), lr = 1e-3, betas = (0.9, 0.99))

cpu_override = False

if torch.backends.mps.is_available():
    device = torch.device("mps")
    rnn = rnn.to(device)
    print('USING DEVICE:', device)
else:
    device = torch.device("cpu")
    rnn = rnn.to(device)
    print('USING DEVICE:', device)

if cpu_override:
    print('CUSTOM OVERRIDE TO CPU EVEN THOUGH GPU IS AVAILABLE')
    device = torch.device("cpu")
    rnn = rnn.to("cpu")

field_dict = {
    1 : 'HIPS 2021',
    2 : 'HIPS 2022',
    3 : 'HIPS 2021 + 2022',
    4 : 'N-Variation 2022'
}

# instantiate dataset
training_data = FeaturesDataset(field = '2022_f54', train=True, test=False, return_split=0)
testing_data     = FeaturesDataset(field = '2022_f54', train=False, test=True, return_split=0)

# instantiate dataloaders for train/test
training_dataloader = torch.utils.data.DataLoader(training_data, batch_size=1, num_workers = 0, drop_last=False, shuffle=True)
testing_datalodaer  = torch.utils.data.DataLoader(testing_data,  batch_size=1, num_workers = 0, drop_last=False)

running_loss = []
r_2_list = [] # used in function test_after_epoch
rmse_list = [] # used in function test_after_epoch
relative_rmse_list = [] # used in function test_after_epoch
total_loss = 0

def get_field_metadata(field_id):
    if field_id == 1:
        field = 'hips_2021'
    if field_id == 2:
        field = 'hips_2022'
    if field_id == 3:
        field = 'hips_both_years'
    if field_id == 4:
        field = '2022_f54'
    df = get_svr_features(debug=False, data_path=field)

    return df

def get_subplot_metadata(df, plot_id):
    plot_id = np.array(plot_id.unique(), dtype=np.float64)
    # query for the pedigree, hybrid/inbred, and dates:
    pedigree = df[df['Plot'] == int(plot_id)]['pedigree'].unique()
    nitrogen_treatment = df[df['Plot'] == int(plot_id)]['nitrogen_treatment'].unique()
    hybrid_or_inbred = df[df['Plot'] == int(plot_id)]['hybrid_or_inbred'].unique()
    dates = df[df['Plot'] == int(plot_id)]['date'].unique()
    if hybrid_or_inbred.shape[0] != 1 or pedigree.shape[0] != 1 or nitrogen_treatment.shape[0] != 1:
        print('WARNING: MULTIPLE PEDIGREES OR N-TREATMENTS MIXED IN A PLOT PREDICITON')

    return *pedigree, *hybrid_or_inbred, *nitrogen_treatment, dates


def test_after_epoch(epoch, field_id, plot_t_preds=True, plot_1to1=True):
    df = get_field_metadata(field_id) # load the right dataframe so we can get the pedigree/genotype data when plotting.
    
    """
    Currently, this only tests where batch size = 1. Update to accomodate larger batch size. 
    """
    y_pred = [] # we reset this to empty for every epoch
    y_true = [] # we reset this to empty for every epoch

    if plot_t_preds:
        num_testing_samples = len(testing_datalodaer)
        print('length of test', num_testing_samples)
        rows = 2
        cols = 5
        fig, ax = plt.subplots(rows, cols, figsize=(30,12))
        fig.suptitle('Prediction vs GT for Testing Split: Epoch ' + str(epoch))

    for n, testing_sample in enumerate(testing_datalodaer):
        features, GT, plot_id, field_id = testing_sample

        if device == torch.device("mps") or device == torch.device("cuda"): # handle input to GPU
            features = features.to(torch.float32) 
            features = features.to(device)

            # if input and network are in float32, we want to move the GT to float 32 as well
            # because we want precision of input/target in loss computation to be the same:
            GT = GT.to(torch.float32)
            GT = GT.to(device) # move GT to GPU

        features = torch.squeeze(features, 0)

        GT = torch.squeeze(GT, 0)
        
        out = rnn(features)
        _, _, final_pred, all_pred = out
        #print(final_pred.shape, all_pred.shape, GT.shape)
        # generate plot of prediction vs GT:
        

        for x in range(GT.shape[0]):
            y_true.append(float(GT[x].cpu().detach().numpy())) # append y_true to list.
            y_pred.append(float(all_pred[x].cpu().detach().numpy())) # append y_pred to list

        if plot_t_preds:
            # get the right pedigree and hybrid/inbred data for the plot:
            metadata = get_subplot_metadata(df, plot_id)
            pedigree, hybrid_inbred, nitrogen_treatment, dates = metadata
            # the lists above will be needed to compute r_2_avgs and rmse_avgs after the entire testing sample is iterated through.
            # plot GT vs Predicted:
            row_idx= n // cols  # Calculate the row index
            col_idx = n % cols   # Calculate the column index
            #print(row_idx, col_idx)
            ax[row_idx, col_idx].plot(dates, GT.cpu().detach().numpy(), label='Ground Truth')
            ax[row_idx, col_idx].plot(dates, all_pred.cpu().detach().numpy(), label = 'Prediction')
            ax[row_idx, col_idx].legend()
            ax[row_idx, col_idx].set_title('Predictions for ' + pedigree + ': ' + dates[0][0:4])
            ax[row_idx, col_idx].set_xlabel('Date')
            ax[row_idx, col_idx].set_ylabel('LAI')

            #print('lenght is', GT.cpu().detach().numpy().shape)
            #
            # print('y_true is', y_true)
            # print('y_pred is', y_pred)

            #loss = criterion(all_pred, GT)

    r_2 = r2_score(y_true, y_pred) # compute r_2 
    rmse = mean_squared_error(y_true, y_pred, squared=False) 
    max_y_true = np.max(np.array(y_true))
    min_y_true = np.min(np.array(y_true)) 
    relative_rmse = rmse / (max_y_true - min_y_true) # relative RMSE def from Purnima's revised RMSE paper (on my iPad)

    r_2_list.append(r_2) # save r_2 for each epoch. List isn't currently used anywhere
    rmse_list.append(rmse) # save RMSE for each epoch. List isn't currently used anywhere
    relative_rmse_list.append(relative_rmse) # save relative RMSE for each epoch. List isn't currently used anywhere
    print('r_2, rmse, relative_rmse testing after epoch' , epoch, ': ', r_2, rmse, relative_rmse)

    if plot_t_preds:
        fig.savefig('/Users/alim/Documents/prototyping/research_lab/research_code/models/GT_vs_pred_over_time/GT_vs_pred_sample_e' + str(epoch) + '_f' + str(int(field_id)) + '.jpg')
        fig.clf() # clear figure for future plots
    if plot_1to1:
        # plot the 1:1 line of predictions, GT values:
        plt.figure(figsize=(8,8))
        plt.scatter(y_true, y_pred, label='Pred vs GT')
        plt.plot([0,7], [0,7], label='One to one', color = 'black')
        plt.title('Ground Truth vs Predicted LAI on ' + str(field_dict[int(field_id)]) + ' After Epoch ' + str(epoch))
        plt.xlabel('Ground Truth LAI')
        plt.ylabel('Predicted LAI')
        plt.xlim(0,7)
        plt.ylim(0,7)
        plt.annotate(f'R^2 = {r_2:.2f}', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=10)
        plt.annotate(f'RMSE = {rmse:.2f}', xy=(0.05, 0.8), xycoords='axes fraction', fontsize=10)
        plt.annotate(f'rRMSE = {relative_rmse:.2f}', xy=(0.05, 0.7), xycoords='axes fraction', fontsize=10)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend(loc='lower right')
        # save after creating plot:
        plt.savefig('/Users/alim/Documents/prototyping/research_lab/research_code/models/one_to_one_plots/one_to_one_e' + str(epoch) + '_f' + str(int(field_id)) + '.jpg')
        plt.clf() # clear plot/figure

        


# training loop below. Note, not possible to do batched gradient descent. Should implement this, so that we can find a better optimized 
# function.
for epoch in range(epochs):
    field_id = None
    for i, timeseries in enumerate(training_dataloader):
        optimizer.zero_grad()

        features, GT, plot_id, field_id_holder = timeseries
        field_id = field_id_holder # save field id for bringing up metadata in testing.
        if device == torch.device("mps") or device == torch.device("cuda"): # handle input to GPU
            features = features.to(torch.float32) 
            features = features.to(device)

            # if input and network are in float32, we want to move the GT to float 32 as well
            # because we want precision of input/target in loss computation to be the same:
            GT = GT.to(torch.float32)
            GT = GT.to(device) # move GT to GPU
        
        features = torch.squeeze(features, 0)
        GT = torch.squeeze(GT, 0)
        
        
        
        out = rnn(features)
        
        _, _, final_pred, all_pred = out

        loss = criterion(all_pred, GT) # compute loss - should be on same device
        #print('loss is', loss)

        loss.backward() # compute gradient of loss wrt each parameter
        optimizer.step() # take a step based on optimizer learning rate and hyper-parameters.
        total_loss += loss.item()

        if (i + 1) % 50 == 0:
            avg_loss = total_loss / 50
            running_loss.append(avg_loss)
            print("Loss in epoch", epoch, " is ", avg_loss)
            total_loss = 0

    print('KICKING OFF TEST AFTER EPOCH')
    test_after_epoch(epoch = epoch, field_id = field_id ) 


# print training loss curve after training run:
plt.plot(running_loss)
plt.xlabel('Iteration * 50')
plt.ylabel('Loss')
plt.title('Loss over training for ' + testing_data.field)
plt.savefig('training_running_loss_' + testing_data.field + '.jpg')
plt.clf() # close figure so we can save r_2, rmse values later.

# print rmse and r_2 after each epoch:
plt.plot(r_2_list, label='R^2 values')
plt.plot(rmse_list, label = 'RMSE values')
plt.title('R^2 and RMSE on Test Data - ' + testing_data.field)
plt.xlabel('Epoch')
plt.ylabel('RMSE or R_2')
plt.legend()
plt.savefig('r_2_and_rmse_over_training_' + testing_data.field + '.jpg')


# save the model:
torch.save(rnn.state_dict(), 'trained_rnn_model.pth')

total_params = sum(p.numel() for p in rnn.parameters())
print("Total number of parameters: {}".format(total_params))