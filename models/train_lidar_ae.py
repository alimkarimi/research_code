import torch
import torch.nn as nn
from torch import optim

import sys
sys.path.append('..')
from dataloading_scripts.hyperspectral_lidar_weather_dataloader import FeaturesDataset
from models import HyperspectralAE, LiDARAE
from hyperspectral_lidar_processing.hyperspectral_plot_extraction import get_visual_hyperspectral_rgb, visualize_lidar_points

from matplotlib import pyplot as plt

import numpy as np
import os

import time

from sklearn.metrics import r2_score, mean_squared_error

lidar_ae = LiDARAE(debug=True)

# init optimizer:
optimizer = optim.Adam(lidar_ae.parameters(), lr = 1e-3, betas = (0.9, 0.99))


cpu_override = False
epochs = 1
field = 'hips_2021'

# get number of model params:
total_params = sum(p.numel() for p in lidar_ae.parameters())
print("Total number of parameters: {}".format(total_params))

if torch.backends.mps.is_available():
    device = torch.device("mps")
    lidar_ae = lidar_ae.to(device)
    print('USING DEVICE:', device, '(GPU)')
else:
    device = torch.device("cpu")
    lidar_ae = lidar_ae.to(device)
    print('USING DEVICE:', device, '(CPU)')

if cpu_override:
    print('CUSTOM OVERRIDE TO CPU EVEN THOUGH GPU IS AVAILABLE')
    device = torch.device("cpu")
    lidar_ae = lidar_ae.to("cpu")

training_data = FeaturesDataset(field = field, train=True, test=False, load_individual=True, load_series = False, debug=True)
training_dataloader = torch.utils.data.DataLoader(training_data, batch_size=1, num_workers = 0, drop_last=False)

criterion = nn.MSELoss()
running_loss = []
total_loss = 0

for epoch in range(epochs):
    for n, data in enumerate(training_dataloader):
        # reset gradients
        optimizer.zero_grad()
        
        # unpack data:
        hyp_data, GT, freq, point_cloud, GDD, PREC = data

        # make point_cloud have float32 precision for nn.Linear operation and to run on Mac GPU
        point_cloud = point_cloud.to(torch.float32)
        point_cloud = point_cloud.to(device)

        output = lidar_ae(point_cloud)
        output = torch.squeeze(output, 0)
        point_cloud = torch.squeeze(output, 0)
        print(output.shape, point_cloud.shape)

        print(point_cloud[0:10, :])
        print(output[0:10, :])

        output_np = output.cpu().detach().numpy()
        point_cloud_np = point_cloud.cpu().detach().numpy()

        # Count the number of NaN values
        num_nan = np.sum(np.isnan(point_cloud_np))
        num_finite = np.sum(np.isfinite(point_cloud_np))
        print("Number of NaN values:", num_nan)
        print("number of finite values:", num_finite)
        print("min and max of output_np x:", np.min(output_np[:,0]), np.max(output_np[:,0]))
        print("min and max of output_np y:", np.min(output_np[:,1]), np.max(output_np[:,1]))

        print("min and max of point_cloud_np:", np.min(point_cloud_np), np.max(point_cloud_np))


        # Get indices of NaN values
        nan_indices = np.argwhere(np.isnan(output_np))
        print("Indices of NaN values:")
        for idx in nan_indices:
            print(idx)

        visualize_lidar_points(lidar_data = point_cloud.cpu().detach().numpy(), filename='original.jpg', save=True)
        visualize_lidar_points(lidar_data = output.cpu().detach().numpy(), filename='reconstructed.jpg', save=True) # some nan number?

        # compute loss: MSE on original point cloud and predicted point cloud in the decoder.

        loss = criterion(output, point_cloud) # criterion(input, target)  

        # compute gradients wrt loss for each parameter
        loss.backward()

        # backpropogate loss:
        optimizer.step()

        total_loss += loss.item()

        if (n + 1) % 50 == 0:
            avg_loss = total_loss / 50
            print('avg loss in epoch', epoch, ' currently is', avg_loss)
            running_loss.append(avg_loss)
            total_loss = 0

plt.plot(running_loss)
plt.title("Loss in LiDAR Autoencoder Training")
plt.xlabel("Batch * 25")
plt.ylabel("Loss")
plt.savefig("LiDAR_AE_loss.jpg")

# save the model:
torch.save(lidar_ae.state_dict(), 'trained_lidar_autoencoder_model_' + field + '.pth')
print('model saved')