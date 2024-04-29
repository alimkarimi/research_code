import torch
import torch.nn as nn
from torch import optim
import dask.dataframe as dd

import sys
sys.path.append('..')
from dataloading_scripts.feature_dataloader import FeaturesDataset
from dataloading_scripts.dataloading_utils import get_field_metadata, get_subplot_metadata, field_dict
from models import Transformer
from hyperspectral_lidar_processing.hyperspectral_plot_extraction import get_visual_hyperspectral_rgb

from matplotlib import pyplot as plt

import numpy as np
import os

import time

from sklearn.metrics import r2_score, mean_squared_error

torch.manual_seed(0) # to replicate results
epochs = 20
cpu_override=False
field = 'hips_2022'
if field == 'hips_2021':
    timepoints=4
if field == 'hips_2022':
    timepoints=3
if field == '2022_f54':
    timepoints=5
# init transformer:
model = Transformer(input_size = 17, embedding_dim=17, timepoints=timepoints, num_transformer_blocks=8,
                    custom_mha=False)

# get number of model params:
total_params = sum(p.numel() for p in model.parameters())
print("Total number of parameters: {}".format(total_params))

optimizer = optim.Adam(params=model.parameters(), lr= 1e-3, betas = (0.9, 0.99))


# instantiate dataset
training_data = FeaturesDataset(field = field, train=True, test=False, return_split=0)
testing_data     = FeaturesDataset(field = field, train=False, test=True, return_split=0)

# instantiate dataloaders for train/test
training_dataloader = torch.utils.data.DataLoader(training_data, batch_size=1, num_workers = 0, drop_last=False, shuffle=True)
testing_datalodaer  = torch.utils.data.DataLoader(testing_data,  batch_size=1, num_workers = 0, drop_last=False)

criterion = nn.MSELoss()
running_loss = []
r_2_list = [] # used in function test_after_epoch
rmse_list = [] # used in function test_after_epoch
relative_rmse_list = [] # used in function test_after_epoch
total_loss = 0

# move models to correct location (GPU, CPU):
if torch.backends.mps.is_available():
    device = torch.device("mps")
    model = model.to(device)
    print('USING DEVICE:', device, '(GPU)')
else:
    device = torch.device("cpu")
    model = model.to(device)
    print('USING DEVICE:', device, '(CPU)')

if cpu_override:
    print('CUSTOM OVERRIDE TO CPU EVEN THOUGH GPU IS AVAILABLE')
    device = torch.device("cpu")
    model = model.to("cpu")

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
        
        out = model(features)
        all_pred = out
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
            if field_id <= 3: # make sure the plot title makes logical sense for the evaluation task
                ax[row_idx, col_idx].set_title('Predictions for ' + pedigree + ': ' + dates[0][0:4])
            if field_id > 3: # make sure the plot title makes logical sense for the evaluation task
                ax[row_idx, col_idx].set_title('Predictions for ' + str(nitrogen_treatment) + ': ' + dates[0][0:4])
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
        fig.savefig('/Users/alim/Documents/prototyping/research_lab/research_code/models/GT_vs_pred_over_time/trans_GT_vs_pred_sample_e' + str(epoch) + '_f' + str(int(field_id)) + '.jpg')
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
        plt.savefig('/Users/alim/Documents/prototyping/research_lab/research_code/models/one_to_one_plots/trans_one_to_one_e' + str(epoch) + '_f' + str(int(field_id)) + '.jpg')
        plt.clf() # clear plot/figure

for epoch in range(epochs):
    for n, timeseries in enumerate(training_dataloader):
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

        out = model(features)

        loss = criterion(out, GT)
        total_loss = total_loss + loss.item()

        loss.backward()

        optimizer.step()

        if (n + 1) % 10 == 0:
            avg_loss = total_loss / 10
            running_loss.append(avg_loss)
            print('loss in epoch', epoch, ' is ', avg_loss)
            total_loss = 0
    
    print('Starting test...')
    test_after_epoch(epoch=epoch, field_id=field_id, plot_t_preds=True, plot_1to1 = True)

plt.plot(running_loss[10:])
plt.xlabel('Iteration * 10')
plt.ylabel('Loss')
plt.title('Loss over training for ' + testing_data.field)
plt.savefig('transformer_training_running_loss_' + testing_data.field + '.jpg')
plt.clf() # close figure so we can save r_2, rmse values later.
    
            




