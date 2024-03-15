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

import time

from sklearn.metrics import r2_score, mean_squared_error

# # load up sample image - will need to be part of dataloader at some point..
# path = '/Users/alim/Documents/prototyping/research_lab/HIPS_Hyperspectral/20210727'
# files = os.listdir(path)
# for file in files:
#     if '.npy' not in file:
#         files.remove(file)

# print(len(files))

# hyp_np = np.load(path + '/' + files[0])
# print(hyp_np.max(), hyp_np.min())
# print(hyp_np.shape)

in_channels = 136
height = 130
width = 42

# init model:
ae_model = HyperspectralAE(in_channels, height, width, debug=False)
#ae_model = ae_model.to(device)

# init optimizer:
optimizer = optim.Adam(ae_model.parameters(), lr = 1e-3, betas = (0.9, 0.99))


cpu_override = False
epochs = 5
field = 'hips_both_years'

# get number of model params:
total_params = sum(p.numel() for p in ae_model.parameters())
print("Total number of parameters: {}".format(total_params))

if torch.backends.mps.is_available():
    device = torch.device("mps")
    ae_model = ae_model.to(device)
    print('USING DEVICE:', device)
else:
    device = torch.device("cpu")
    ae_model = ae_model.to(device)
    print('USING DEVICE:', device)

if cpu_override:
    print('CUSTOM OVERRIDE TO CPU EVEN THOUGH GPU IS AVAILABLE')
    device = torch.device("cpu")
    ae_model = ae_model.to("cpu")

training_data = FeaturesDataset(field = field, train=True, test=False, load_individual=True, load_series = False, debug=False)
training_dataloader = torch.utils.data.DataLoader(training_data, batch_size=1, num_workers = 0, drop_last=False)

criterion = nn.MSELoss()
running_loss = []
total_loss = 0

def plot_input_and_reconstruction(in_img, reconstructed_img, freq, epoch, batch):
    # rgb_visual_display function expects h x w x c format.
    in_img = in_img.transpose(1,2,0)
    reconstructed_img = reconstructed_img.transpose(1,2,0)
    # get orig and reconstructed visual data:
    rgb_orig = get_visual_hyperspectral_rgb(in_img, freq)
    rgb_reconstructed = get_visual_hyperspectral_rgb(reconstructed_img, freq)

    fig, ax = plt.subplots(1,2)
    fig.suptitle('Result for epoch ' + str(epoch) + ' and batch ' + str(batch))
    ax[0].imshow(rgb_orig)
    ax[0].set_title('Original')
    ax[1].imshow(rgb_reconstructed)
    ax[1].set_title('Reconstruction')
    fig.savefig('/Users/alim/Documents/prototyping/research_lab/research_code/visualizations/ae_reconstructions/InputvReconstructed'
                + str(epoch) + '_' + str(batch) + '.jpg')
    plt.cla() # clear axis after saving..
    plt.close(fig)

    return 0

for epoch in range(epochs):
    for n, batch_data in enumerate(training_dataloader):
        optimizer.zero_grad() # reset gradients.

        # unpack from dataloader:
        img, GT, freq, _, GDD, PREC = batch_data # _ is the point cloud. Since we are training a representation of hyperspectral data, 
        # we ignore the point cloud
        img = img.to(torch.float32)
        #print(img.dtype)
        #print(img.shape)
        img = img.to(device)
        img = torch.squeeze(img, 0)
        #print(img.shape)
        out = ae_model(img)

        loss = criterion(out, img) # format is criterion(network_output, target)
        #print('loss on current batch is', loss)
        loss.backward() # compute gradient of loss wrt each parameter
        optimizer.step() # take a step based on optimizer learning rate and hyper-parameters.
        total_loss += loss.item()

        if (n+1) % 101 == 0:
            avg_loss = total_loss / 101
            running_loss.append(avg_loss)
            print("Loss in epoch", epoch, " is ", avg_loss)
            total_loss = 0

            
            np_orig_img = img.cpu().detach().numpy()
            np_reconstructed_img = out.cpu().detach().numpy()
            np_freq = freq.cpu().detach().numpy()

            plot_input_and_reconstruction(in_img = np_orig_img, reconstructed_img = np_reconstructed_img, freq = np_freq.T.squeeze(1),
                                         epoch = epoch, batch = n)
        


# plot loss:
plt.plot(running_loss)
plt.title('Loss over Training')
plt.xlabel('Training batch * 50')
plt.ylabel('Loss')
plt.savefig('Hyperspectral_AE_loss.jpg')

# save model:
torch.save(ae_model.state_dict(), 'trained_hyperspectral_autoencoder_model_' + field + '.pth')
print('model saved')
