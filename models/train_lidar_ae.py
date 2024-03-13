import torch
import torch.nn as nn
from torch import optim

import sys
sys.path.append('..')
from dataloading_scripts.hyperspectral_dataloader import FeaturesDataset
from models import HyperspectralAE, LiDARAE
from hyperspectral_lidar_processing.hyperspectral_plot_extraction import get_visual_hyperspectral_rgb

from matplotlib import pyplot as plt

import numpy as np
import os

import time

from sklearn.metrics import r2_score, mean_squared_error

lidar_ae = LiDARAE()

# init optimizer:
optimizer = optim.Adam(lidar_ae.parameters(), lr = 1e-3, betas = (0.9, 0.99))


cpu_override = False
epochs = 5

# get number of model params:
total_params = sum(p.numel() for p in lidar_ae.parameters())
print("Total number of parameters: {}".format(total_params))

if torch.backends.mps.is_available():
    device = torch.device("mps")
    lidar_ae = lidar_ae.to(device)
    print('USING DEVICE:', device)
else:
    device = torch.device("cpu")
    lidar_ae = lidar_ae.to(device)
    print('USING DEVICE:', device)

if cpu_override:
    print('CUSTOM OVERRIDE TO CPU EVEN THOUGH GPU IS AVAILABLE')
    device = torch.device("cpu")
    lidar_ae = lidar_ae.to("cpu")

training_data = FeaturesDataset(field = 'hips_2021', train=True, test=False, load_individual=True, load_series = False, debug=False)
training_dataloader = torch.utils.data.DataLoader(training_data, batch_size=1, num_workers = 0, drop_last=False)

criterion = nn.MSELoss()
running_loss = []
total_loss = 0

for n, data in enumerate(training_dataloader):
    # reset gradients
    optimizer.zero_grad()
    
    # unpack data:
    hyp_data, GT, freq, point_cloud = data

    # make point_cloud have float32 precision for nn.Linear operation and to run on Mac GPU
    point_cloud = point_cloud.to(torch.float32)
    point_cloud = point_cloud.to(device)

    output = lidar_ae(point_cloud)

    # compute loss: MSE on original point cloud and predicted point cloud in the decoder.

    loss = criterion(output, point_cloud) # criterion(input, target)  

    # compute gradients wrt loss for each parameter
    loss.backward()

    # backpropogate loss:
    optimizer.step()

    total_loss += loss.item()

    if (n + 1) % 25 == 0:
        avg_loss = total_loss / 25
        print('avg loss in epoch currently is', avg_loss)
        running_loss.append(avg_loss)
        total_loss = 0

plt.plot(running_loss)
plt.title("Loss in LiDAR Autoencoder Training")
plt.xlabel("Batch * 25")
plt.ylabel("Loss")
plt.savefig("LiDAR_AE_loss.jpg")