import torch
import torch.nn as nn
from torch import optim

import sys
sys.path.append('..')
from dataloading_scripts.hyperspectral_dataloader import FeaturesDataset
from models import HyperspectralAE
from hyperspectral_lidar_processing.hyperspectral_plot_extraction import get_visual_hyperspectral_rgb

from matplotlib import pyplot as plt

import numpy as np
import os

from sklearn.metrics import r2_score, mean_squared_error

# load up sample image - will need to be part of dataloader at some point..
path = '/Users/alim/Documents/prototyping/research_lab/HIPS_Hyperspectral/20210727'
files = os.listdir(path)
for file in files:
    if '.npy' not in file:
        files.remove(file)

print(len(files))

hyp_np = np.load(path + '/' + files[0])
print(hyp_np.max(), hyp_np.min())
print(hyp_np.shape)

in_channels = 136
height = 130
width = 42

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(device)
else:
    device = torch.device("cpu")
    print(device)

# init model:
ae_model = HyperspectralAE(in_channels, height, width)
ae_model = ae_model.to(device)

if torch.backends.mps.is_available():
    device = torch.device("mps")

training_data = FeaturesDataset(field = 'hips_2021', train=True, test=False, load_individual=True, load_series = False, debug=False)
training_dataloader = torch.utils.data.DataLoader(training_data, batch_size=1, num_workers = 0, drop_last=False)

for n, batch_data in enumerate(training_dataloader):
    # unpack:
    img, GT = batch_data
    img = img.to(torch.float32)
    print(img.dtype)
    print(img.shape)
    img = img.to(device)
    img = torch.squeeze(img, 0)
    print(img.shape)

    out = ae_model(img)
    break



#ae_model(torch.tensor(hyp_np).float