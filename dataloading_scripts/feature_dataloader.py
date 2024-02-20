import pandas as pd
import numpy as np

import sys
sys.path.append('..') # need this to access get_svr_features from training.py
from dataloading_scripts.read_purnima_features import get_svr_features

import torch

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

def train_test_split_for_dataloading(debug=False, field = 'hips_2021'):

    df = get_svr_features(debug=False, data_path=field)

    if field != '2022_f54': 
        groups = df['pedigree']
        y = df['hybrid_or_inbred'] # used for stratification
    else:
        groups = df['nitrogen_treatment']
        y = df['date'] 

    # make a train and test split
    train_indices = None
    test_indices = None
    sss = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=1)
    for i, (train_indices, test_indices) in enumerate(sss.split(df.iloc[:, 1:-5], y, groups)):
    # train_indices and test_indices are arrays of the train and test indices for a split (fold).
    # they will not necessarily be the same length, untless the train/test sizes are 0.5/0/5.
    # stratification is done based on the y labels (sss.split(X, y, group))

        # save train/test indices for later
        train_indices = train_indices
        test_indices = test_indices

        # do scalar transform on train_indices:
        scaler_k_fold = StandardScaler()
        transformed_training = scaler_k_fold.fit_transform(df.iloc[train_indices, 1:-5])
        transformed_testing  = scaler_k_fold.transform(df.iloc[test_indices, 1:-5])
 
        # convert back to a df:
        transformed_training = pd.DataFrame(transformed_training, columns = df.columns[1:-5])
        transformed_testing  = pd.DataFrame(transformed_testing, columns = df.columns[1:-5])

        # insert transformed df into "right part" of original df to preserve all the metadata.
        df.iloc[train_indices, 1 : -5] = transformed_training
        df.iloc[test_indices,  1 : -5] = transformed_testing

        break # just get one stratified split for now. Deep learning models will take longer to optimize 
        # than statistical models. 

    if debug:
        print(train_indices)
        print(test_indices)
        print('train,test idx')

    return df, train_indices, test_indices


class FeaturesDataset(torch.utils.data.Dataset):
    """
    This class builds a dataset from a specific field of hyperspectral and LiDAR data that is usable inside a torch
    dataloader. 
    """
    def __init__(self, field : str, train : bool, test : bool):
        super(FeaturesDataset).__init__()

        # get df of features from the field provided in the constructor
        self.df, train_indices, test_indices = train_test_split_for_dataloading(field = field)
        if train:
            self.df = self.df.iloc[train_indices, :]
        if test:
            self.df = self.df.iloc[test_indices, :]

        if train == test:
            print('Warning!!! Train and test are both true. Will not recieve a meaningful split...')

        self.num_plots = self.df['Plot'].unique().shape[0] # number of unique plots. We will need to pass features from each of these in a time series
        # to the model.
        self.plots = self.df['Plot'].unique()
        

    def __len__(self):
        return self.num_plots

    def __getitem__(self, index, debug=False):
        # to do: add stratification logic into this __getitem__ method.
        # normalize using standardscalar
        # split into train and test
        # update to get nitrogen based features

        if debug:
            print('querying for these plots:', self.plots[index])
        data = self.df[self.df['Plot'] == self.plots[index]]
        ground_truth_LAI = data[['LAI']].values
        train_or_test_features = data[['height_95p', 'canopy_cover_50p', 'canopy_cover_75p',
       'plot_volume2', 'VCI', 'CAP', 'LPI', 'NDVI705', 'GNDVI', 'EVI', 'VOG2',
       'MCARI2', 'MTVI', 'PBI', 'EVI2', 'GDD', 'PREC', ]].values
        metadata = data[['Plot', 'date', 'hybrid_or_inbred', 'pedigree', 'nitrogen_treatment']]

        # convert to torch tensors
        ground_truth_LAI = np.array(ground_truth_LAI, dtype = np.float64) # for compatibility with pytorch model
        train_or_test_features = np.array(train_or_test_features, dtype = np.float64) # for compatibility with pytorch model

        ground_truth_LAI = torch.tensor(ground_truth_LAI, dtype=torch.float64)
        train_or_test_features = torch.tensor(train_or_test_features, dtype=torch.float64)

        # returns data in the format batch_size x num_dates x num_features.
        # for example, if batch size is 1, and 2021 HIPS has 4 observations with 17 features, the returned 
        # train_or_test_features are 1 x 4 x 17

        # the GT is 1 x 4 x 1, where there are 4 different ground truth LAIs (one for each observation)
        
        return train_or_test_features, ground_truth_LAI

if __name__ == "__main__":
    training_data = FeaturesDataset(field = 'hips_2021', train=True, test=False)
    testing_data  = FeaturesDataset(field = 'hips_2022', train=True, test=False)

    training_dataloader = torch.utils.data.DataLoader(training_data, batch_size=1, num_workers = 0, drop_last=False)
    testing_datalodaer  = torch.utils.data.DataLoader(testing_data,  batch_size=1, num_workers = 0, drop_last=False)
    print('created dataloader... starting enumerating.')
    for n, (features, GT) in enumerate(training_dataloader):
        print(n)
        print(features.shape)
        print(GT.shape)

    for n, (features, GT) in enumerate(testing_datalodaer):
        print(n)
        print(features.shape)
        print(GT.shape)