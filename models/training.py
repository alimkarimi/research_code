from models import RNN

import torch
import torch.nn as nn
from torch import optim

import sys
sys.path.append('..')
from dataloading_scripts.feature_dataloader import FeaturesDataset

from matplotlib import pyplot as plt

import numpy as np

from sklearn.metrics import r2_score, mean_squared_error


epochs = 20
criterion = nn.MSELoss()
batch_size = 1

# instantiate model
rnn  = RNN(batch_size = batch_size, concat_based_LSTM = True, addition_based_LSTM = False,
           hidden_size = 100, cell_size = 100) # lstm gets instantiated inside RNN class.
rnn = rnn.double()



# instantiate optimizer
optimizer = torch.optim.Adam(rnn.parameters(), lr = 1e-3, betas = (0.9, 0.99))

# instantiate dataset
training_data = FeaturesDataset(field = 'hips_2021', train=True, test=False)
testing_data     = FeaturesDataset(field = 'hips_2021', train=False, test=True)

# instantiate dataloaders for train/test
training_dataloader = torch.utils.data.DataLoader(training_data, batch_size=1, num_workers = 0, drop_last=False, shuffle=True)
testing_datalodaer  = torch.utils.data.DataLoader(testing_data,  batch_size=1, num_workers = 0, drop_last=False)

running_loss = []
r_2_list = [] # used in function test_after_epoch
rmse_list = [] # used in function test_after_epoch
total_loss = 0

def test_after_epoch():
    y_pred = [] # we reset this to empty for every epoch
    y_true = [] # we reset this to empty for every epoch

    for n, testing_sample in enumerate(testing_datalodaer):
        features, GT = testing_sample

        features = torch.squeeze(features, 0)
        features = features.double()

        GT = torch.squeeze(GT, 0)
        GT = GT.to(torch.float64)
        
        out = rnn(features)
        _, _, final_pred, all_pred = out
        # print(final_pred.shape, all_pred.shape)
        # print(GT.shape)
        for x in range(GT.shape[0]):
            y_true.append(float(GT[x].detach().numpy())) # append y_true to list.
            y_pred.append(float(all_pred[x].detach().numpy())) # append y_pred to list

        # the lists above will be needed to compute r_2_avgs and rmse_avgs after the entire testing sample is iterated through.

        # print('y_true is', y_true)
        # print('y_pred is', y_pred)

        #loss = criterion(all_pred, GT)
        
    r_2 = r2_score(y_true, y_pred) # compute r_2 
    rmse = mean_squared_error(y_true, y_pred, squared=False) 
    r_2_list.append(r_2)
    rmse_list.append(rmse)
    print('r_2, rmse testing after epoch' , epoch, ': ', r_2, rmse)

            

for epoch in range(epochs):
    for i, timeseries in enumerate(training_dataloader):
        optimizer.zero_grad()

        features, GT = timeseries

        features = torch.squeeze(features, 0)
        features = features.double()

        GT = torch.squeeze(GT, 0)
        GT = GT.to(torch.float64)
        
        out = rnn(features)
        _, _, final_pred, all_pred = out

        loss = criterion(all_pred, GT)
        #print('loss is', loss)

        loss.backward() # compute gradient of loss wrt each parameter
        optimizer.step() # take a step based on optimizer learning rate and hyper-parameters.
        total_loss += loss.item()

        if (i + 1) % 20 == 0:
            avg_loss = total_loss / 20
            running_loss.append(avg_loss)
            #print("Loss is ", avg_loss)
            total_loss = 0

    print('KICKING OFF TEST AFTER EPOCH')
    test_after_epoch() 


# print training loss curve after training run:
plt.plot(running_loss)
plt.xlabel('iter * 20')
plt.ylabel('loss')
plt.title('loss over training')
plt.savefig('training_running_loss.jpg')
plt.clf() # close figure so we can save r_2, rmse values later.

# print rmse and r_2 after each epoch:
plt.plot(r_2_list, label='R^2 values')
plt.plot(rmse_list, label = 'RMSE values')
plt.title('R^2 and RMSE on Test Data -' + testing_data.field)
#plt.ylim(bottom = -1, top = 1)
plt.xlabel('Epoch')
plt.ylabel('RMSE or R_2')
plt.legend()
plt.savefig('r_2_and_rmse_over_training.jpg')

# save the model:
torch.save(rnn.state_dict(), 'trained_rnn_model.pth')

total_params = sum(p.numel() for p in rnn.parameters())
print("Total number of parameters: {}".format(total_params))


# create empty list for y_true and y_pred. Will need for computing RMSE and R^2




# compute RMSE and R^2 values:
# RMSE is the square root of the squared difference of the sum of each prediction and observation divided by the number of observations.

