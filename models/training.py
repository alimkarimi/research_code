from models import RNN

import torch
import torch.nn as nn
from torch import optim

import sys
sys.path.append('..')
from dataloading_scripts.feature_dataloader import FeaturesDataset

epochs = 1
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

for epoch in range(epochs):
    for n, timeseries in enumerate(training_dataloader):
        optimizer.zero_grad()

        features, GT = timeseries

        features = torch.squeeze(features, 0)
        features = features.double()

        GT = torch.squeeze(GT, 0)
        GT = GT.to(torch.float64)
        
        out = rnn(features)
        _, _, final_pred, all_pred = out

        loss = criterion(all_pred, GT)
        print('loss is', loss)

        loss.backward() # compute gradient of loss wrt each parameter
        optimizer.step() # take a step based on optimizer learning rate and hyper-parameters.


# save the model:
torch.save(rnn.state_dict(), 'trained_rnn_model.pth')

total_params = sum(p.numel() for p in rnn.parameters())
print("Total number of parameters: {}".format(total_params))


for n, testing_sample in enumerate(testing_datalodaer):
    features, GT = testing_sample

    features = torch.squeeze(features, 0)
    features = features.double()

    GT = torch.squeeze(GT, 0)
    GT = GT.to(torch.float64)
    
    out = rnn(features)
    _, _, final_pred, all_pred = out

    loss = criterion(all_pred, GT)
    print('loss is', loss)


