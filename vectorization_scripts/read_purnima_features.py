import os
import pandas as pd


def get_svr_features(debug=False, data_path='2022_f54'):
    """
    Put extracted features from HIPS or Field 54 into a Pandas dataframe.

    For Field 54:
     - input 2022_f54 as the arg to data_path. 
     - Dataframe shape: 480 observations, 21 features (date col gives another column)
     - Unique dates of observations are:
      - '20220624' '20220714' '20220721' '20220812' '20220830'
    - This is used for Nitrogen experiments on the same genotype


    For HIPS in 2021 only:
     - input 'hips_2021' as the arg to data_path:
     - Dataframe shape: 352 observations, 19 features (date col adds another column)
     - Unique dates of observations are:
      - '20210727' '20210802' '20210808' '20210816
    - HIPS 2021 is field 42
    - HIPS 2022 is field 78
    - Used for different genotype experiments

    For HIPS in 2021 and 2022:
     - input 'hips_both_years' as the arg to data_path:
     - Dateframe shape: 617 observations, 43 features.
     - With a visual check, it looks like all the features in 2021 are present in 2022.
     - 2021 features are a subset of the 2022 features, which is why there are so many features
     - unique dates of observations are:
        - '20210727' '20210802' '20210808' '20210816' '20220624' '20220714' '20220831']
    - HIPS 2021 is field 42
    - HIPS 2022 is field 78
    - Used for different genotype experiments with the same nitrogen treatment

    """
    base_path = '/Volumes/iot4agrs/data/students/Alim/data_from_Purnima/Features/Final_Features'
    # for some reason, sometimes I need to have base_path be /Volumes/depot/iot4agrs/ ...

    if data_path == 'hips_both_years':
        path_w_env = '/2021_2022_HIPS/LiDAR_and_spectral_with_envi_variables/'
    if data_path == 'hips_2021':
        path_w_env = '/2021_HIPS/LiDAR_and_spectral_with_GDD_Precipitation/'
    if data_path == '2022_f54':
        path_w_env = '/Field54_2022/LiDAR_and_spectral_with_GDD_precipitation/'

    dates = os.listdir(base_path + path_w_env)
    if '.DS_Store' in dates:
        dates.remove('.DS_Store')
    dates.sort()

    df = pd.DataFrame()
    
    # get data from Purnima's extracted features into data frames
    for n, date in enumerate(dates):
        file_to_open = base_path + path_w_env  + date + '/features3.csv'
        temp_df = pd.read_csv(file_to_open)
        temp_df['date'] = date
        print(date)
        df = pd.concat([df, temp_df], ignore_index=True)
        if debug:
            print(temp_df.shape)
            print(df.shape)
            print(df.head())
            print(df.tail())
            print(df.describe())
    if debug:
        print("FINAL SUMMARY")
        print("Dimensions of returned df:", df.shape)
        print(df.head())
        print(df.tail())
        print(df.describe())
        print("Unique Dates in df:", df['date'].unique())
    return df

def read_hyperspectral_per_plot_reflectance():
    """
    This function reads data that was originally on Purnima's computer.
    Hyperspectral reflectance per band and per plot, along with metadata about how many
    pixels per plot are present. 

    There are also vegetation indices in this data. 

    The reflectance is averaged over the number of pixels in the
    plot. 
    """
    
    base_path = ('/Volumes/depot/iot4agrs/data/students/Alim/data_from_Purnima/Features/Final_Features/'
                 'Intermediate Files/Hyperspectral_Feature_Extraction/2022_Field_54/54mn/')

    dates = os.listdir(base_path)
    dates.remove('.DS_Store') # remove .DS_Store file
    dates.sort() # sort in alphabetical order

    folders = [date for date in dates if 'india_44m' not in date] # removes india_44m from relevant folders
    # because only india44m has hyperspectral data we want to read. 
    

    df = pd.DataFrame()
    print(folders)
    for folder in folders:
        if folder[0:8] == '20220810':
            file = 'mean_plots_10trimmed_mn.csv'
        elif folder[0:8] == '20220830':
            file = 'mean_plots_10trimmed_s.csv'
        else:
            file = 'mean_plots_10trimmed.csv'

        temp_path = base_path + folder + '/hyper_' + folder[0:8] + '_' + file
        temp_df = pd.read_csv(temp_path)
        temp_df['date'] = folder[0:8]
        print('opening file:', temp_path)

        print("printing col names!!")
        
        for column_name in temp_df.columns:
            print(column_name)
        df = pd.concat([df, temp_df], ignore_index = True, axis=0)
        print('temp_shp',temp_df.shape)
        print(df.shape)
    
    print(df.head())
    print(df.tail())
    
    return df


def read_lidar_per_plot(debug=False):
    """
    Read data from LiDAR into dataframe for accessiblity during model training.
    This is on a per plot basis. 
    
    Retruned dataframe contains plot id, number of points in that plot, percentiles, and 
    other extracted features like LPI (Laser Penetration Index) by date.

    There are 6 unique in this dataframe:
    20220624, '20220710, 20220714, 20220721, 20220810, 20220830

    The total size of the returned dataframe:
     - 288 observations, 37 features
    """

    base_path = ('/Volumes/depot/iot4agrs/data/students/Alim/data_from_Purnima/Features/'
                'Final_Features/Intermediate Files/LiDAR_Feature_Extraction/'
                '2022_F54/54mn/result/')
    dates = os.listdir(base_path)
    dates.remove('.DS_Store')
    dates.sort()   
    print(dates) 
    field_and_alt = '/f54/44m/'


    # instantiate df:
    df = pd.DataFrame()
    for date in dates:
        temp_file_path = base_path + date + field_and_alt + \
        'LiDAR_' + date + 'mean_Hist_features_meanrows_10trimmed_23_16cm.csv'
        temp_df = pd.read_csv(temp_file_path)
        print(temp_df.shape)
        temp_df['date'] = date
        df = pd.concat([df, temp_df], ignore_index=True)
    if debug:
        for i in df['Plot']:
            print(i)
        print("Final DF stats:")
        print("Shape of dataframe:", df.shape)
        print(df.head())
        print(df.tail())
        print(df.describe())
    return df

def read_ground_truth(debug=False, field="f54"):
    """
    Reads data from LAI ground truth files for Field 54. Ground truth from HIPS 2021 (Field 42) and
    HIPS 2022 (Field 78) is read in the function read_purnima_features()
    
    The LAI values are the average LAI per plot.
    Readings done by LAI-2200c machine. No destructive sampling. 

    The LAI GT is over Field 54.
    Output has 552 observations,
    Plot id, LAI of that plot on a certain date.
    Field 54 data for Nitrogen variations only has data for 2022. 
    
    """

    base_path = '/Volumes/depot/iot4agrs/data/students/Alim/data_from_Purnima/Features/LAI_Data/Field_54/'
    LAI_folder_names = os.listdir(base_path)
    LAI_folder_names.remove('.DS_Store')

    LAI_folder_names.sort()
    LAI_folder_names = LAI_folder_names[0:8]

    df = pd.DataFrame()
    print(LAI_folder_names)

    for folder_name in LAI_folder_names:

        full_path = base_path + folder_name + '/LAI_' + folder_name[:8] + '.csv'
        temp_df = pd.read_csv(full_path)
        temp_df.rename(columns={'Unnamed: 0': 'Plot', folder_name[:8] : 'LAI'},inplace=True)
        temp_df['date'] = folder_name[:8]

        df = pd.concat([df, temp_df], ignore_index=True)

    if debug:
        print("Final df shape:", df.shape)
        print(df.head())
        print(df.tail())
        print(df.describe())
        print(df['date'].unique())
        

    return df



if __name__ == "__main__":
    df = get_svr_features(debug=True)
    # total_nan_count = df.isna().sum().sum()
    # print(total_nan_count)

    # read_hyperspectral_per_plot_reflectance()

    #df = read_lidar_per_plot(debug=True)


   #read_ground_truth(debug=True)


