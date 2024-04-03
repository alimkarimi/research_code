import torch
import torch.nn as nn
from torch import optim

import sys
sys.path.append('..')
from dataloading_scripts.hyperspectral_lidar_weather_dataloader import FeaturesDataset
from dataloading_scripts.read_purnima_features import get_svr_features
from models import HyperspectralAE, LiDARAE, RNN
from hyperspectral_lidar_processing.hyperspectral_plot_extraction import get_visual_hyperspectral_rgb

from matplotlib import pyplot as plt

import numpy as np
import os

import time

from sklearn.metrics import r2_score, mean_squared_error

# first, we need to load the .pth files for LiDAR and Hyperspectral based autoencoders

in_channels = 136
height = 130
width = 42

cpu_override = False

# init hyperspectral model:
hyp_ae_model = HyperspectralAE(in_channels, height, width, debug=False, encoder_only=True)

# init lidar model:
lidar_ae_model = LiDARAE(encoder_only = True)

# Load Model State Dictionaries for Lidar and hyperspectral models
model_base_path = '/Users/alim/Documents/prototyping/research_lab/research_code/models/'

hyp_ae_model_path = model_base_path + 'trained_hyperspectral_autoencoder_model_hips_both_years.pth'
hyp_state_dict = torch.load(hyp_ae_model_path)

lidar_ae_model_path = model_base_path + 'trained_lidar_autoencoder_model_hips_both_years.pth'
lidar_state_dict = torch.load(lidar_ae_model_path)

# init RNN:
rnn = RNN(batch_size = 1, concat_based_LSTM = True, addition_based_LSTM = False,
           hidden_size = 50, cell_size = 50, input_size=34, debug=False) # lstm gets instantiated inside RNN class.
rnn = rnn.to(torch.float32)

# instantiate optimizer
optimizer = torch.optim.Adam(rnn.parameters(), lr = 1e-3, betas = (0.9, 0.99))


# Load state dictionaries into the respective models
hyp_ae_model.load_state_dict(hyp_state_dict)
lidar_ae_model.load_state_dict(lidar_state_dict)

# move models to correct location (GPU, CPU):
if torch.backends.mps.is_available():
    device = torch.device("mps")
    lidar_ae_model = lidar_ae_model.to(device)
    hyp_ae_model = hyp_ae_model.to(device)
    rnn = rnn.to(device)
    print('USING DEVICE:', device, '(GPU)')
else:
    device = torch.device("cpu")
    lidar_ae_model = lidar_ae_model.to(device)
    hyp_ae_model = hyp_ae_model.to(device)
    rnn = rnn.to(device)
    print('USING DEVICE:', device, '(CPU)')

if cpu_override:
    print('CUSTOM OVERRIDE TO CPU EVEN THOUGH GPU IS AVAILABLE')
    device = torch.device("cpu")
    lidar_ae_model = lidar_ae_model.to("cpu")
    hyp_ae_model = hyp_ae_model.to("cpu")
    rnn = rnn.to("cpu")



# then, we need to concatenate the latent representations of those vectors with the weather observations

training_data = FeaturesDataset(field = 'hips_2021', train=True, test=False, 
                                load_individual=False, load_series = True, debug=False)
testing_data     = FeaturesDataset(field = 'hips_2021', train=False, test=True, 
                                load_individual=False, load_series=True, debug=False)

training_dataloader = torch.utils.data.DataLoader(training_data, batch_size=1, num_workers = 0, drop_last=False, shuffle=True)
testing_dataloader= torch.utils.data.DataLoader(testing_data, batch_size=1, num_workers = 0, drop_last=False, shuffle=True)

running_loss = []
r_2_list = []
rmse_list = []
relative_rmse_list = []

epochs = 20
#init loss function:
criteron = nn.MSELoss()
total_loss = 0

field_dict = {
        1 : 'HIPS 2021',
        2 : 'HIPS 2022',
        3 : 'HIPS 2021 + 2022'
    }

def get_field_metadata(field_id):
    if field_id == 1:
        field = 'hips_2021'
    if field_id == 2:
        field = 'hips_2022'
    if field_id == 3:
        field = 'hips_both_years'
    df = get_svr_features(debug=False, data_path=field)

    return df

def get_subplot_metadata(df, plot_id):
    plot_id = np.array(plot_id.unique(), dtype=np.float64)
    # query for the pedigree, hybrid/inbred, and dates:
    pedigree = df[df['Plot'] == int(plot_id)]['pedigree'].unique()
    hybrid_or_inbred = df[df['Plot'] == int(plot_id)]['hybrid_or_inbred'].unique()
    dates = df[df['Plot'] == int(plot_id)]['date'].unique()
    if hybrid_or_inbred.shape[0] != 1 or pedigree.shape[0] != 1:
        print('WARNING: MULTIPLE PEDIGREES MIXED IN A PLOT PREDICITON')

    return *pedigree, *hybrid_or_inbred, dates


def test_after_epoch(epoch, field_id, plot_t_preds=True, plot_1to1=True):
    df = get_field_metadata(field_id) # load the right dataframe so we can get the pedigree/genotype data when plotting.
    
    """
    Currently, this only tests where batch size = 1. Update to accomodate larger batch size. 
    """
    y_pred = [] # we reset this to empty for every epoch
    y_true = [] # we reset this to empty for every epoch

    if plot_t_preds:
        num_testing_samples = len(testing_dataloader)
        print('length of test', num_testing_samples)
        rows = 2
        cols = 5
        fig, ax = plt.subplots(rows, cols, figsize=(30,12))
        fig.suptitle('Prediction vs GT for Testing Split: Epoch ' + str(epoch))

    for n, testing_sample in enumerate(testing_dataloader):
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
            pedigree, hybrid_inbred, dates = metadata
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

for epoch in range(epochs):
    for n, batch in enumerate(training_dataloader):
        # first, unpack batch:
        img, GT, point_cloud, GDD, PREC = batch # point_cloud is a list of tensors b/c of their variable size.
        #print(img.shape, GT.shape, len(point_cloud), GDD.shape, PREC.shape)
        T = img.shape[1]
        #print(T, 'this is T')
        # create empty tensor to hold the concatenated embeddings:
        full_feature_vector = torch.zeros((T, 34))

        # concat GDD and PREC into weather tensor:
        weather = torch.concat([GDD, PREC]).to(torch.float32)
        weather = weather.to(device)
        #print(weather.shape) # 2 x timepoints

        # move hyperspectral and lidar to float32 precision:
        img = img.squeeze(0)
        img = img.to(torch.float32)
        img = img.to(device)
        for t in range(len(point_cloud)):
            point_cloud[t] = point_cloud[t].to(torch.float32)
            point_cloud[t] = point_cloud[t].to(device)

        # run each hyperspectral img through hyperspectral AE and append latent output to RNN input (full_feature_vector)
        for t in range(img.shape[0]):
            hyp_embedding = hyp_ae_model(img[t]) # hyp_embdding is 500 dims
            full_feature_vector[t, 0:32] = hyp_embedding
        
        # # run point_cloud through lidar AE and append latent vector to RNN input (full_feature_vector)
        # for t in range(T):
        #     #temp_pc = torch.squeeze(point_cloud[t], 0)
        #     lidar_embedding = lidar_ae_model(point_cloud[t]) # lidar_embedding is 1 x 32 x 1
        #     lidar_embedding = torch.squeeze(lidar_embedding, 0)
        #     lidar_embedding = torch.squeeze(lidar_embedding, 1)
        #     full_feature_vector[t, 32: 64] = lidar_embedding

        # append the weather data into full feature vector:
        weather = weather.T # reshape weather data to T x data
        for t in range(T):
            full_feature_vector[t, 32:] = weather[t]

        full_feature_vector = full_feature_vector.to(device)

        rnn_out = rnn(full_feature_vector)

        ct, ht, pred, predictions_in_series = rnn_out

        loss = criteron(predictions_in_series, GT.to(torch.float32).to(device).T)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if (n + 1) % 10 == 0:
            avg_loss = total_loss / 10
            running_loss.append(avg_loss)
            print("Loss in epoch", epoch, " is ", avg_loss)
            total_loss = 0

    #test_after_epoch(epoch = epoch, field_id = field_id)
        

# print loss:
plt.plot(running_loss)
plt.xlabel("Batch * 10")
plt.ylabel("Loss")
plt.title("Loss over Training for Multi-modal LSTM")
plt.savefig("Loss for Image Based LSTM.jpg")

