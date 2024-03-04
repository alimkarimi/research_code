import numpy as np
import torch
import sys
sys.path.append('..') # need this to access get_svr_features from training.py
from dataloading_scripts.read_purnima_features import get_svr_features
from dataloading_scripts.load_plots import (load_plots_coords_for_field, load_individual_plot_xyxy, plot_path_2021_hips, 
plot_path_2022_hips, plot_path_root)
from dataloading_scripts.append_hyper_lidar_to_df import append_hyperspectral_paths
from skimage.transform import rescale, resize

from PIL import Image
import skimage

from sklearn.model_selection import StratifiedGroupKFold

# build data loader that reads in hyperspectral data from local disk
# then transforms that data to be mean centered and puts it into a nerual network.

# we should also return LAI for the given plot. 

# would be useful to have:
# dataframe with path of hyperspectral image, path of lidar data, the plot id, plot row, and genotype, weather, etc.

# best to use logic from feature dataloader and append image paths. 



def train_test_split_for_dataloading(debug=False, field = 'hips_2021'):

    df = append_hyperspectral_paths(field=field)
    print(df.shape, 'output from append func!!')

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
    def __init__(self, field : str, train : bool, test : bool, debug : bool, load_series : bool, load_individual : bool):
        super(FeaturesDataset).__init__()

        # get df of features from the field provided in the constructor
        self.load_series = load_series
        self.load_individual = load_individual
        self.df, train_indices, test_indices = train_test_split_for_dataloading(field = field, debug=debug)
        if train:
            self.df = self.df.iloc[train_indices, :]
        if test:
            self.df = self.df.iloc[test_indices, :]

        if train == test:
            print('Warning!!! Train and test are both true. Will not recieve a meaningful split...')

        self.num_plots = self.df['Plot'].unique().shape[0] # number of unique plots. We will need to pass features from each of these in a time series
        # to the model.
        self.plots = self.df['Plot'].unique()
        print('unique plot are', self.plots)
        self.field = field
        

    def __len__(self):
        if self.load_individual:
            return self.df.shape[0] # number of training rows
        else:
            return self.num_plots

    def __getitem__(self, index, debug=False):
        # to do:
        # load up an image from the index, turn it into the format torch expects.
        # compute mean of each channel in training data, use that to subtract from the mean of the data loaded. 
        if self.load_series:
            print(self.df['Plot'], 'this is self.df[Plot]')
            print('\n\n\n')
            print('index is', index)
            print(self.plots[index], 'this is self.plots[index]')
            data_row_1 = self.df[self.df['Plot'] == self.plots[index]]['hyp_path_p1']
            print('this is data:', data_row_1.values)
            print(type(data_row_1))
            print(data_row_1.index)

            data_row_2 = self.df[self.df['Plot'] == self.plots[index]]['hyp_path_p2']
            print('GOT INDEX')
            list_images_row_1 = []
            list_images_row_2 = []
            for i in data_row_1.index:
                # open data: 
                data_np = np.load(data_row_1[i])
                list_images_row_1.append(data_np)
                print(data_row_2[i])
                print(data_np.shape)

            for i in data_row_2.index:
                data_np = np.load(data_row_2[i])
                list_images_row_2.append(data_np)
                print(data_np.shape)

            # create empty tensor to hold result of the dataloader output:
            timeseries_tensor = torch.empty((len(list_images_row_1), 136, 130, 42))


            ground_truth_LAI = self.df[self.df['Plot'] == self.plots[index]]['LAI'].values
            #print('ground truth:', ground_truth_LAI)
            dates = self.df[self.df['Plot'] == self.plots[index]]['date']
            #print(dates)

            # combine data from row 1 and row 2 into an array that is 136 x 130 x 42
            # first, resize each plot into 136 x 130 x 21. This is done in the loop below. 
            # we iterate through each row in list_images_row_1 and list_images_row_2 and 
            # reshape each to 136 x 130 x 21 and then combine into 136 x 130 x 42.
            # We then add that combined image (the "mega-image" that contains both rows) into
            # the correct index of the timeseries tensor. This is the hyperspectral image we want to return
            # from the dataloader! 
            for t, img in enumerate(zip(list_images_row_1, list_images_row_2)):

                ### Normalize to by dividing by max pixel value found in dataset:
                max_HIPS_px = 8470.0
                min_HIPS_px = -555.0

                img_row_1 = img[0] / max_HIPS_px
                img_row_2 = img[1] / max_HIPS_px

                # clip so that negative values do not exist and so that we get pixels in the range 0 - 1:
                img_row_1 = np.clip(img_row_1, 0, 1)
                img_row_2 = np.clip(img_row_2, 0, 1)

                print(img_row_1.shape)
                print('max after clipping between 0 and 1:', img_row_1.max())

                # resize using skimage:
                resized_img_row_1 = skimage.transform.resize(img_row_1, output_shape = (136, 130, 21))
                resized_img_row_2 = skimage.transform.resize(img_row_2, output_shape = (136, 130, 21))
                print(resized_img_row_1.shape, resized_img_row_2.shape)
                
                # stack row plots together:
                stacked_rows = np.concatenate((resized_img_row_1, resized_img_row_2), axis=2)
                print(stacked_rows.shape)
                print('HERE!')
                print(stacked_rows.max())
                timeseries_tensor[t] = torch.tensor(stacked_rows)
            
            return timeseries_tensor, ground_truth_LAI

        if self.load_individual: # load individual images instead of in a series.
            # the intuition here is to just learn good features of the image, regardless of the 
            # point in the time series.

            path_row_1 = self.df.iloc[index]['hyp_path_p1']
            path_row_2 = self.df.iloc[index]['hyp_path_p2']

            ground_truth_LAI = self.df.iloc[index]['LAI']

            # load imgs of each row:
            img_row_1 = np.load(path_row_1)
            img_row_2 = np.load(path_row_2)

            ### Normalize to by dividing by max pixel value found in dataset:
            max_HIPS_px = 8470.0
            min_HIPS_px = -555.0

            img_row_1 = img_row_1 / max_HIPS_px
            img_row_2 = img_row_2 / max_HIPS_px

            # clip so that negative values do not exist and so that we get pixels in the range 0 - 1:
            img_row_1 = np.clip(img_row_1, 0, 1)
            img_row_2 = np.clip(img_row_2, 0, 1)

            # resize to 136 x 130 x 21 using skimage:
            resized_img_row_1 = skimage.transform.resize(img_row_1, output_shape = (136, 130, 21))
            resized_img_row_2 = skimage.transform.resize(img_row_2, output_shape = (136, 130, 21))
            
            # stack row plots together:
            stacked_rows = np.concatenate((resized_img_row_1, resized_img_row_2), axis=2) 

            return stacked_rows, ground_truth_LAI # stacked rows is the hyperspectral data.


if __name__ == '__main__':
    #train_test_split_for_dataloading(field='hips_2021')
    training_data = FeaturesDataset(field = 'hips_2021', train=True, test=False, debug=True, load_individual=True, load_series=False)
    training_dataloader = torch.utils.data.DataLoader(training_data, batch_size=1, num_workers = 0, drop_last=False)
    print('len is', training_data.__len__())
    for n, i in enumerate(training_dataloader):
        print(i[0].shape, i[1].shape)
        print(n)