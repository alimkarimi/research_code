import numpy as np
import torch
import sys
sys.path.append('..') # need this to access get_svr_features from training.py
from dataloading_scripts.read_purnima_features import get_svr_features
from dataloading_scripts.load_plots import (load_plots_coords_for_field, load_individual_plot_xyxy, plot_path_2021_hips, 
plot_path_2022_hips, plot_path_root)
from dataloading_scripts.append_hyper_lidar_to_df import append_hyperspectral_lidar_paths
from skimage.transform import rescale, resize
from torchvision import transforms as tvt

from PIL import Image
import skimage

from sklearn.model_selection import StratifiedGroupKFold

torch.manual_seed(0)
np.random.seed(0)

# build data loader that reads in hyperspectral data from local disk
# then transforms that data to be mean centered and puts it into a nerual network.

# we should also return LAI for the given plot. 

# would be useful to have:
# dataframe with path of hyperspectral image, path of lidar data, the plot id, plot row, and genotype, weather, etc.

# best to use logic from feature dataloader and append image paths. 

channel_means = np.load('/Users/alim/Documents/prototyping/research_lab/research_code/analysis/channel_means.npy')[:136]

def train_test_split_for_dataloading(debug=False, field = 'hips_2021'):

    df = append_hyperspectral_lidar_paths(field=field)
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
        #print('unique plot are', self.plots)
        self.field = field

        # compute the mean for each channel in the training dataset. This will be used to do find the mean centering transoform.
        

    def __len__(self):
        if self.load_individual:
            return self.df.shape[0] # number of training rows
        if self.load_series:
            return self.num_plots

    def __getitem__(self, index, debug=False):
        # to do:
        # load up an image from the index, turn it into the format torch expects.
        # compute mean of each channel in training data, use that to subtract from the mean of the data loaded. 
        if self.load_series:
            # series loader workflow:
            # index into plot ids
            # get all the hyperspectral paths, lidar paths, weather readings, and LAI readings for the plot over the year
            # combine hyperspectral (with normalization)
            # combine lidar (with normalization)
            # return LAI
            # return weather
            # return hyp combined
            # return lidar combined
            #print('index is', index)
            #print(self.plots[index], 'this is self.plots[index]')
            data_row_1 = self.df[self.df['Plot'] == self.plots[index]][['hyp_path_p1', 'lidar_path_p1']] 
            # access hyperspectral data path row 1 and lidar data path row 1
            # print('this is data:', data_row_1.values.shape)
            # print(type(data_row_1))
            # print(data_row_1.index)
            # print(type(data_row_1.values))

            data_row_2 = self.df[self.df['Plot'] == self.plots[index]][['hyp_path_p2', 'lidar_path_p2']] 

            data_weather = self.df[self.df['Plot'] == self.plots[index]][['GDD', 'PREC']]
            # access hyperspectral data path row 2 and lidar data path row 2.
            #print(self.plots[index])
            #print('GOT INDEX')
            list_images_row_1 = []
            list_images_row_2 = []
            list_lidar_row_1 = []
            list_lidar_row_2 = []

            for n, i in enumerate(data_row_1.index):
                # open hyp and lidar data in row 1: 
                data_hyp_np = np.load(data_row_1.iloc[n, 0])
                data_lidar_np = np.load(data_row_1.iloc[n, 1])
                list_images_row_1.append(data_hyp_np)
                list_lidar_row_1.append(data_lidar_np)
                #print(data_hyp_np.shape, 'shape hyp!', data_lidar_np.shape, 'lidar shp' )
                

            for n, i in enumerate(data_row_2.index):
                # open hyp and lidar data in row 2:
                data_hyp_np = np.load(data_row_2.iloc[n, 0])
                data_lidar_np = np.load(data_row_2.iloc[n, 1])
                list_images_row_2.append(data_hyp_np)
                list_lidar_row_2.append(data_lidar_np)
                #print(data_hyp_np.shape, 'shape again', data_lidar_np.shape, 'lidar shape')

            GDDs = torch.empty((len(list_images_row_1)))
            PRECs = torch.empty((len(list_images_row_1)))

            for t, i in enumerate(data_weather.index):
                # add GDD and PREC data to tensors. 
                GDDs[t] = data_weather.iloc[n, 0]
                PRECs[t] = data_weather.iloc[n, 1]

            # create empty tensor to hold result of the dataloader output:

            timeseries_hyp_tensor = torch.empty((len(list_images_row_1), 136, 130, 42))
            timeseries_lidar_list = [] # since each point cloud is not a fixed number of points, use a list to hold as
            # a list can hold variable data types / lengths.


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
            for t, (hyp_r1, hyp_r2, lid_r1, lid_r2) in enumerate(zip(list_images_row_1, list_images_row_2, list_lidar_row_1, list_lidar_row_2)):
                # First, work with hyperspectral data:
                ### Normalize to by dividing by max pixel value found in dataset:
                max_HIPS_px = 8470.0
                min_HIPS_px = -555.0

                img_row_1 = hyp_r1 / max_HIPS_px
                img_row_2 = hyp_r2 / max_HIPS_px

                # clip so that negative values do not exist and so that we get pixels in the range 0 - 1:
                img_row_1 = np.clip(img_row_1, 0, 1)
                img_row_2 = np.clip(img_row_2, 0, 1)

                # print(img_row_1.shape)
                # print('max after clipping between 0 and 1:', img_row_1.max())

                # resize using skimage:
                resized_img_row_1 = skimage.transform.resize(img_row_1, output_shape = (136, 130, 21))
                resized_img_row_2 = skimage.transform.resize(img_row_2, output_shape = (136, 130, 21))
                #print(resized_img_row_1.shape, resized_img_row_2.shape)
                
                # stack row plots together:
                stacked_rows = np.concatenate((resized_img_row_1, resized_img_row_2), axis=2)
                # print(stacked_rows.shape)
                # print('HERE!')
                # print(stacked_rows.max())
                timeseries_hyp_tensor[t] = torch.tensor(stacked_rows)

                # now, process lidar data:
                # merge point clouds from each time based observation (i.e, row1 and row2 from t1)
                plot_point_cloud = np.concatenate([lid_r1, lid_r2])
                #print(plot_point_cloud.shape, 'shape of point cloud concatenated')

                # Normalization of LiDAR data: Normalize each batch. I believe taking the mean of the entire batch introduces
                # spatial bias.
                x_mean = np.mean(plot_point_cloud[:,0])
                y_mean = np.mean(plot_point_cloud[:,1])
                z_mean = np.mean(plot_point_cloud[:,2])

                # rescale using min-max normalization:
                plot_point_cloud[:,0] = (plot_point_cloud[:,0] - np.min(plot_point_cloud[:,0])) / (np.max(plot_point_cloud[:,0]) - np.min(plot_point_cloud[:,0]))
                plot_point_cloud[:,1] = (plot_point_cloud[:,1] - np.min(plot_point_cloud[:,1])) / (np.max(plot_point_cloud[:,1]) - np.min(plot_point_cloud[:,1]))
                plot_point_cloud[:,2] = (plot_point_cloud[:,2] - np.min(plot_point_cloud[:,2])) / (np.max(plot_point_cloud[:,2]) - np.min(plot_point_cloud[:,2]))

                timeseries_lidar_list.append(plot_point_cloud)
            
            return timeseries_hyp_tensor, ground_truth_LAI, timeseries_lidar_list, GDDs, PRECs

        if self.load_individual: # load individual images instead of in a series.
            # the intuition here is to just learn good features of the image, regardless of the 
            # point in the time series.

            # First, get the Hyperspectral data:
            path_row_1 = self.df.iloc[index]['hyp_path_p1']
            path_row_2 = self.df.iloc[index]['hyp_path_p2']
            #test_row_1 = self.df.loc[index, 'hyp_path_p2']
            #print(test_row_1)
            #print('path_row_2', path_row_2)
            
            freq_path = path_row_2[0:-10] + 'numpy_freq.npy'

            freq_data = np.load(freq_path)

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

            # We are done with Hyperspectral, now get LiDAR data:
            # grab lidar paths:
            lidar_path_row_1 = self.df.iloc[index]['lidar_path_p1']
            lidar_path_row_2 = self.df.iloc[index]['lidar_path_p2']

            # load lidar from numpy file:
            point_cloud_row1 = np.load(lidar_path_row_1)
            point_cloud_row2 = np.load(lidar_path_row_2)

            # merge point clouds:
            plot_point_cloud = np.concatenate([point_cloud_row1, point_cloud_row2])

            # Normalization of LiDAR data: Normalize each batch. I believe taking the mean of the entire batch introduces
            # spatial bias.
            x_mean = np.mean(plot_point_cloud[:,0])
            y_mean = np.mean(plot_point_cloud[:,1])
            z_mean = np.mean(plot_point_cloud[:,2])

            # rescale using min-max normalization:
            plot_point_cloud[:,0] = (plot_point_cloud[:,0] - np.min(plot_point_cloud[:,0])) / (np.max(plot_point_cloud[:,0]) - np.min(plot_point_cloud[:,0]))
            plot_point_cloud[:,1] = (plot_point_cloud[:,1] - np.min(plot_point_cloud[:,1])) / (np.max(plot_point_cloud[:,1]) - np.min(plot_point_cloud[:,1]))
            plot_point_cloud[:,2] = (plot_point_cloud[:,2] - np.min(plot_point_cloud[:,2])) / (np.max(plot_point_cloud[:,2]) - np.min(plot_point_cloud[:,2]))


            # get weather data from original dataframe:
            GDD = self.df.iloc[index]['GDD']
            PREC = self.df.iloc[index]['PREC']

            # returns the hyperspectral data, LAI, band centers, point cloud, growing degree days, precipitation
            return stacked_rows, ground_truth_LAI, freq_data[:136], plot_point_cloud, GDD, PREC 

            

if __name__ == '__main__':
    #train_test_split_for_dataloading(field='hips_2021')
    training_data = FeaturesDataset(field = 'hips_both_years', train=True, test=False, debug=True, load_individual=False, 
    load_series=True)
    training_dataloader = torch.utils.data.DataLoader(training_data, batch_size=1, num_workers = 0, drop_last=False,
                                                        shuffle = True)
    print('len is', training_data.__len__())
    for n, i in enumerate(training_dataloader):
        print(i[0].shape, i[1].shape, i[3].shape, i[4].shape)
        for x in i[2]:
            print(x.shape)
        print(n)