import torch
import torch.nn as nn
from torch import optim

import sys
sys.path.append('..')
from dataloading_scripts.feature_dataloader import FeaturesDataset
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
height = 136
width = 20

ae_model = HyperspectralAE(in_channels, height, width)
ae_model(torch.tensor(hyp_np).float())