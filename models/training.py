from models import LSTM_concat_based, RNN

import torch
import torch.nn as nn
from torch import optim

import sys
sys.path.append('..')
from dataloading_scripts.feature_dataloader import FeaturesDataset

# instantiate model
lstm = LSTM_concat_based(input_size = 17, hidden_size=100, cell_size=100)
rnn  = RNN(concat_based_LSTM = True, addition_based_LSTM = False)

epochs = 1
criterion = nn.MSELoss()

# instantiate optimizer
optimizer = torch.optim.Adam(rnn.parameters(), lr = 1e-3, betas = (0.9, 0.99))

# instantiate dataset
training_data = FeaturesDataset(field = 'hips_2021', train=True, test=False)
testing_data     = FeaturesDataset(field = 'hips_2021', train=False, test=True)

# instantiate dataloaders for train/test
training_dataloader = torch.utils.data.DataLoader(training_data, batch_size=1, num_workers = 0, drop_last=False)
testing_datalodaer  = torch.utils.data.DataLoader(testing_data,  batch_size=1, num_workers = 0, drop_last=False)

for epoch in range(epochs):
    for n, timeseries in enumerate(training_dataloader):
        features, GT = timeseries

        features = torch.squeeze(features, 0)
        features = features

        GT = torch.squeeze(GT, 0)
        GT = GT
        
        out = rnn(features)
        _, _, final_pred, all_pred = out
        print(final_pred)
        print(final_pred.shape)
        print(all_pred.shape)
        print()
        print('GT:', GT.shape)

        loss = criterion(all_pred, GT)
        print('loss is', loss)

        loss.backward() # compute gradient of loss wrt each parameter
        optimizer.step() # take a step based on optimizer learning rate and hyper-parameters.


