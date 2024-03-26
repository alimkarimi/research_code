import torch
import torch.nn as nn
from torch import optim

import sys
sys.path.append('..')
from dataloading_scripts.hyperspectral_lidar_weather_dataloader import FeaturesDataset
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
           hidden_size = 100, cell_size = 100, input_size=534, debug=True) # lstm gets instantiated inside RNN class.
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

training_data = FeaturesDataset(field = 'hips_2021', train=True, test=False, load_individual=False, load_series = True, debug=False)
training_dataloader = torch.utils.data.DataLoader(training_data, batch_size=1, num_workers = 0, drop_last=False)

running_loss = []
epochs = 1
#init loss function:
criteron = nn.MSELoss()
total_loss = 0

for epoch in range(epochs):
    for n, batch in enumerate(training_dataloader):
        # first, unpack batch:
        img, GT, point_cloud, GDD, PREC = batch # point_cloud is a list of tensors b/c of their variable size.
        print(img.shape, GT.shape, len(point_cloud), GDD.shape, PREC.shape)
        T = img.shape[1]
        print(T, 'this is T')
        # create empty tensor to hold the concatenated embeddings:
        full_feature_vector = torch.zeros((T, 534))

        # concat GDD and PREC into weather tensor:
        weather = torch.concat([GDD, PREC]).to(torch.float32)
        weather = weather.to(device)
        print(weather.shape) # 2 x timepoints

        # move hyperspectral and lidar to float32 precision:
        img = img.squeeze(0)
        img = img.to(torch.float32)
        img = img.to(device)
        for t in range(len(point_cloud)):
            point_cloud[t] = point_cloud[t].to(torch.float32)
            point_cloud[t] = point_cloud[t].to(device)

        # put each hyperspectral image from the timeseries through autoencoder:
        print(img.shape)
        # point_cloud = point_cloud.to(torch.float32)
        # point_cloud = point_cloud.to(device)

        # run img through hyperspectral AE, run point_cloud through lidar AE:
        print('before hyp embedding')
        for t in range(img.shape[0]):
            hyp_embedding = hyp_ae_model(img[t]) # hyp_embdding is 500 dims
            full_feature_vector[t, 0:500] = hyp_embedding
        print('ran this')

        for t in range(T):
            print(point_cloud[t].shape)
            #temp_pc = torch.squeeze(point_cloud[t], 0)
            lidar_embedding = lidar_ae_model(point_cloud[t]) # lidar_embedding is 1 x 32 x 1
            lidar_embedding = torch.squeeze(lidar_embedding, 0)
            print(lidar_embedding.shape)
            lidar_embedding = torch.squeeze(lidar_embedding, 1)
            full_feature_vector[t, 500: 532] = lidar_embedding

        # append the weather data into full feature vector:
        weather = weather.T # reshape weather data to T x data
        for t in range(T):
            full_feature_vector[t, 532:] = weather[t]

        full_feature_vector = full_feature_vector.to(device)

        rnn_out = rnn(full_feature_vector)

        ct, ht, pred, predictions_in_series = rnn_out
        print(GT.shape, predictions_in_series.shape)

        loss = criteron(predictions_in_series, GT.to(torch.float32).to(device).T)
        print(loss) 

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if (n + 1) % 50 == 0:
            avg_loss = total_loss / 50
            running_loss.append(avg_loss)
            print("Loss in epoch", epoch, " is ", avg_loss)
            total_loss = 0

