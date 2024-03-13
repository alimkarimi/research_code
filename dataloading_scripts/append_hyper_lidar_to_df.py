import numpy as np
import os
import sys
sys.path.append('..') # need this to access get_svr_features from training.py
from dataloading_scripts.read_purnima_features import get_svr_features
from dataloading_scripts.load_plots import (load_plots_coords_for_field, load_individual_plot_xyxy, plot_path_2021_hips, 
plot_path_2022_hips, plot_path_root)

"""
This code builds a dataframe with traditional remote sensing features and paths to hyperspectral data plots. This hyperspectral data 
is saved as a .npy file.
"""

def append_hyperspectral_lidar_paths(field = 'hips_2021'):

    df = get_svr_features(data_path= field)

    df['hyp_path_p1'] = None
    df['hyp_path_p2'] = None
    df['lidar_path_p1'] = None
    df['lidar_path_p2'] = None

    hyp_data_path = '/Users/alim/Documents/prototyping/research_lab/HIPS_Hyperspectral/'
    lidar_data_path = '/Users/alim/Documents/prototyping/research_lab/HIPS_LiDAR/'

    # create a clean list of hyperspectral data folders which we want to iterate through:
    folders_hyp = os.listdir(hyp_data_path)
    if '.DS_Store' in folders_hyp:
        folders_hyp.remove('.DS_Store')
    folders_hyp.sort()

    # create a clean list of lidar data folders which we want to iterate through:
    folders_lidar = os.listdir(lidar_data_path)
    if '.DS_Store' in folders_lidar:
        folders_lidar.remove('.DS_Store')
    folders_lidar.sort()


    for date in zip(folders_hyp, folders_lidar):
        for i in range(df.shape[0]): # iterate through dataframe, so that every row with the date we are on gets updated.
            # get plot id:
            #print('iterating through each row of df - date in df is', df.loc[i,'date'], ' at the same time, date var contains', date)
            if df.loc[i,'date'] == date[0] or df.loc[i,'date'] == date[1] or df.loc[i, 'date'] == '20220714': # a hacky condition to ensure we update the path of the lidar data when necessary.
                #print(date)
                temp_plot = int(df.loc[i, 'Plot'])
                fp_row1 = str(temp_plot) + '_1.npy'
                fp_row2 = str(temp_plot) + '_2.npy'
                temp_date = df.loc[i, 'date']
                if temp_date == '20220624':
                    temp_date = '20220623' # handle date inconsistency so that images can load from directory struct on my computer.
                if temp_date == '20220714':
                    temp_date = '20220710'
                df.loc[i, 'hyp_path_p1'] = hyp_data_path + temp_date + '/' + fp_row1
                df.loc[i, 'hyp_path_p2'] = hyp_data_path + temp_date + '/' + fp_row2

                df.loc[i, 'lidar_path_p1'] = lidar_data_path + date[1] + '/lidar_xyz_' + fp_row1
                df.loc[i, 'lidar_path_p2'] = lidar_data_path + date[1] + '/lidar_xyz_' + fp_row2                

    return df

if __name__ == '__main__':
    # append_hyperspectral_paths(field='hips_2021')
    # append_hyperspectral_paths(field='hips_2022')
    df = append_hyperspectral_lidar_paths(field='hips_2022')
    print(df['hyp_path_p2'][0:20])


