import torch
import torch.nn as nn
from torch import optim

import sys
sys.path.append('..')
from dataloading_scripts.hyperspectral_dataloader import FeaturesDataset
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
           hidden_size = 534, cell_size = 534) # lstm gets instantiated inside RNN class.
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
    print('USING DEVICE:', device, '(GPU)')
else:
    device = torch.device("cpu")
    lidar_ae_model = lidar_ae_model.to(device)
    hyp_ae_model = hyp_ae_model.to(device)
    print('USING DEVICE:', device, '(CPU)')

if cpu_override:
    print('CUSTOM OVERRIDE TO CPU EVEN THOUGH GPU IS AVAILABLE')
    device = torch.device("cpu")
    lidar_ae_model = lidar_ae_model.to("cpu")
    hyp_ae_model = hyp_ae_model.to("cpu")


# then, we need to concatenate the latent representations of those vectors with the weather observations

training_data = FeaturesDataset(field = 'hips_2021', train=True, test=False, load_individual=True, load_series = False, debug=False)
training_dataloader = torch.utils.data.DataLoader(training_data, batch_size=1, num_workers = 0, drop_last=False)

for n, batch in enumerate(training_dataloader):
    # first, unpack batch:
    img, GT, freq, point_cloud, GDD, PREC = batch

    # concat GDD and PREC into weather tensor:
    weather = torch.concat([GDD, PREC]).to(torch.float32)
    weather = weather.to(device)

    # move hyperspectral and lidar data to float32 precision:
    img = img.to(torch.float32)
    img = img.to(device)
    point_cloud = point_cloud.to(torch.float32)
    point_cloud = point_cloud.to(device)

    # run img through hyperspectral AE, run point_cloud through lidar AE:

    hyp_embedding = hyp_ae_model(img) # hyp_embdding is 500 dims
    lidar_embedding = lidar_ae_model(point_cloud) # lidar_embedding is 1 x 32 x 1
    lidar_embedding = torch.squeeze(lidar_embedding, 0).squeeze(1) # squeeze from 1 x 32 x 1 to 32

    # concatenate full feature vector
    full_feature_vector = torch.concat([hyp_embedding, lidar_embedding, weather]) # 500 + 32 + 2 = 534 dimensions
    



