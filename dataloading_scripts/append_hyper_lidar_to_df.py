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

def append_hyperspectral_paths(field = 'hips_2021'):

    df = get_svr_features(data_path= field)

    df['hyp_path_p1'] = None
    df['hyp_path_p2'] = None

    hips_data_path = '/Users/alim/Documents/prototyping/research_lab/HIPS_Hyperspectral/'
    folders = os.listdir(hips_data_path)
    if '.DS_Store' in folders:
        folders.remove('.DS_Store')
    folders.sort()

    for date in folders:
        for i in range(df.shape[0]):
            # get plot id:
            temp_plot = int(df.loc[i, 'Plot'])
            fp_row1 = str(temp_plot) + '_1.npy'
            fp_row2 = str(temp_plot) + '_2.npy'
            temp_date = df.loc[i, 'date']
            if temp_date == '20220624':
                temp_date = '20220623' # handle date inconsistency so that images can load from directory struct on my computer.
            if temp_date == '20220714':
                temp_date = '20220710'
            df.loc[i, 'hyp_path_p1'] = hips_data_path + temp_date + '/' + fp_row1
            df.loc[i, 'hyp_path_p2'] = hips_data_path + temp_date + '/' + fp_row2
            #print('added', hips_data_path + temp_date + '/' + fp_row1)

    return df

if __name__ == '__main__':
    # append_hyperspectral_paths(field='hips_2021')
    # append_hyperspectral_paths(field='hips_2022')
    df = append_hyperspectral_paths(field='hips_both_years')
    print(df['hyp_path_p2'][600])



