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

# init hyperspectral model:
hyp_ae_model = HyperspectralAE(in_channels, height, width, debug=False, encoder_only=True)

# init lidar model:
lidar_ae_model = LiDARAE(encoder_only = True)

# Step 2: Load the Model State Dictionary
model_path = 'your_model.pth'  # Path to your saved model
state_dict = torch.load(model_path)

# Step 3: Load the State Dictionary into the Model
model.load_state_dict(state_dict)

# then, we need to concatenate the latent representations of those vectors with the weather observations

training_data = FeaturesDataset(field = 'hips_2021', train=True, test=False, load_individual=True, load_series = False, debug=False)
training_dataloader = torch.utils.data.DataLoader(training_data, batch_size=1, num_workers = 0, drop_last=False)

