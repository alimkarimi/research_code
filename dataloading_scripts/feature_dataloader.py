import pandas as pd
import numpy as np

from read_purnima_features import get_svr_features

import torch

class FeaturesDataset(torch.utils.data.Dataset):
    def __init__(self, field : str):
        super(FeaturesDataset).__init__()

        # get df of features from the field provided in the constructor
        self.df = get_svr_features(debug=False, data_path=field)
        self.num_plots = self.df['Plot'].unique().shape[0] # number of unique plots. We will need to pass features from each of these in a time series
        # to the model.
        self.plots = self.df['Plot'].unique()

        # stratify:
        

    def __len__(self):
        return self.num_plots

    def __getitem__(self, index):
        # to do: add stratification logic into this __getitem__ method.
        # normalize using standardscalar
        # split into train and test
        # update to get nitrogen based features

        print('querying for these plots:', self.plots[index])
        data = self.df[self.df['Plot'] == self.plots[index]]
        ground_truth_LAI = data[['LAI']].values
        training_features = data[['height_95p', 'canopy_cover_50p', 'canopy_cover_75p',
       'plot_volume2', 'VCI', 'CAP', 'LPI', 'NDVI705', 'GNDVI', 'EVI', 'VOG2',
       'MCARI2', 'MTVI', 'PBI', 'EVI2', 'GDD', 'PREC', ]].values
        metadata = data[['Plot', 'date', 'hybrid_or_inbred', 'pedigree', 'nitrogen_treatment']]

        # convert to torch tensors
        ground_truth_LAI = torch.tensor(ground_truth_LAI)
        training_features = torch.tensor(training_features)
        
        return training_features, ground_truth_LAI

# template below to load dataloader..
# my_train_dataset = MyDataset(root_train, catNms)
# my_val_dataset = MyDataset(root_val, catNms)



# my_val_dataloader = torch.utils.data.DataLoader(my_val_dataset, batch_size = 12, num_workers = 4, drop_last = 

if __name__ == "__main__":
    dataset = FeaturesDataset(field = 'hips_2021')
    print(dataset[4])
    print('this is the TYPE')
    print(type(dataset[87]))
    print('done w type')

    my_train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers = 0, drop_last=False)
    print('created dataloader... starting enumerating.')
    for n, (features, GT) in enumerate(my_train_dataloader):
        print(n)
        print(features.shape)
        print(GT.shape)