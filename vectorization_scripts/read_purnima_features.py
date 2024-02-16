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
    - Used for different genotype experiments with the same nitrogen treatment.
    - Plot 4351 - 4394 are hybrid in 2021, rest are inbred
    - Pedigree information is in HIPS_YS_2021.xlsx

    For HIPS in 2022 only:
     - input 'hips_2022' as the arg to data_path
     - Plot 6351 - 6494 are hybrid in 2022, rest are inbred. 
     - Pedigree information is in '2022 YS_HIPS_BCS all data.xlsx'

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
    - Note that ground truth data for 2022 is in another directory (Features/LAI_Data/HIPS22)


    """
    base_path = '/Volumes/iot4agrs/data/students/Alim/data_from_Purnima/Features/'
    # for some reason, sometimes I need to have base_path be /Volumes/depot/iot4agrs/ ...

    if data_path == 'hips_both_years' or data_path == 'hips_2022':
        path_w_env = 'Final_Features/2021_2022_HIPS/LiDAR_and_spectral_with_envi_variables/'
        path_w_2022_gt = 'LAI_Data/HIPS_22/'
    if data_path == 'hips_2021':
        path_w_env = 'Final_Features/2021_HIPS/LiDAR_and_spectral_with_GDD_Precipitation/'
    if data_path == '2022_f54':
        path_w_env = 'Final_Features/Field54_2022/LiDAR_and_spectral_with_GDD_precipitation/'


    dates = os.listdir(base_path + path_w_env)
    if '.DS_Store' in dates:
        dates.remove('.DS_Store')
    dates.sort()

    df = pd.DataFrame()
    
    # get data from Purnima's extracted features into data frames
    for n, date in enumerate(dates):
        file_to_open = base_path + path_w_env  + date + '/features3.csv'
        temp_df = pd.read_csv(file_to_open)
        temp_df.columns = temp_df.columns.str.strip() # strip whitespace from column names
        temp_df['date'] = date
        df = pd.concat([df, temp_df], ignore_index=True)
        if debug:
            print('processing date: ', date)
            print(temp_df.shape)
            print(df.shape)
            print(df.head())
            print(df.tail())
            print(df.describe())
        
        # only will need to go into the if statement below if we want data from 2022 and 2021 from
        # HIPS
        if (n == len(dates) - 1) and (data_path == 'hips_both_years' or data_path == 'hips_2022'):
            # The features in 2021 are a subset of the features in 2022.
            # This if statement removes the ones that are not in both sets (i.e, finds 
            # the intersection).
            # Create a set from 2021 and 2022. Then apply set intersection and remove those that are not in 
            # the 2021 set.
            df_2021 = get_svr_features(data_path='hips_2021')
            df_2021_cols = set(df_2021.columns.values)
            df_2022_cols = set(df.columns.values)
            set_intersect = df_2022_cols.intersection(df_2021_cols)
            set_intersect = list(set_intersect)
            df = df[df.columns.intersection(set_intersect)]
            if debug:
                print(df_2021_cols)
                print('this is the set_intersect', set_intersect)
                print('done cleaning column names...')
                print('about to append 2022 LAI GT data...')
            
            # append LAI ground truth to 2021 and 2022 dataframe

            ground_truth_2022_HIPS = base_path + path_w_2022_gt
            df_2022_HIPS_GT = pd.read_excel(ground_truth_2022_HIPS + 'LAI.xlsx')
            df_2022_HIPS_GT.columns = df_2022_HIPS_GT.columns.map(str)
            if debug:
                print('shape of 2022 GT data:', df_2022_HIPS_GT.shape)


            # this for loop below goes through every row of the dataframe.
            # for every LAI ground truth that is blank in df, we get the plot id and date
            # that the blank LAI GT value corresponds to, find the LAI for that plot id
            # and date from the df_2022_HIPS_GT dataframe, and insert that value into 
            # the previously blank LAI value.
            # if the plot id does not exist, we delete that row.
            for i in range(df.shape[0]):
                if pd.isna(df.loc[i, 'LAI']):
                    temp_date = df.loc[i, 'date']
                    # get plot id for the current date:
                    if pd.notna(df['Plot'][i]):
                        temp_plot_id = df['Plot'][i]
                        #print('plot id is', temp_plot_id)
                        # get the plot ids LAI value from df_2022_HIPS_GT:
                        temp_filtered = df_2022_HIPS_GT[df_2022_HIPS_GT['plot'] == temp_plot_id]
                        df.loc[i, 'LAI'] = temp_filtered[temp_date].values
                    if pd.isna(df['Plot'][i]):
                        print('dropping row!!!')
                        # remove this row from dataframe. Data cannot be used.
                        df = df.drop(i) 
            df = df.reset_index(drop=True) # reset indexing for the dropped rows. We do not want to change in the loop above
            # as that can cause issues with going through the range/shape in the for i in range(df.shape[0]) command.

    
    if data_path == 'hips_2022':
        # for creating 2022, we loaded 2021 and 2022 data. We need to drop 2021 from df
        df = df[df['date'].str.contains('2022')]
        df = df.reset_index(drop=True) # subsetting only the 2022 data gives us indices that start from 352. Not ideal for
        # future operations / slicing. drop=True means that we will not get a new column called 'index' with the original
        # indices. 
        

    if data_path == 'hips_2022' or data_path == 'hips_2021' or data_path == 'hips_both_years':
        # below, we add data about hybrid vs inbred and pedigree (i.e, variant). This is only relevant for HIPS.

        # open pedigree spreadsheets:
        pedigree_df_2021 = pd.read_excel('/Volumes/depot/iot4agrs/data/students/Alim/pedigree_data/' + 'HIPS_YS_2021.xlsx')
        pedigree_df_2022 = pd.read_excel('/Volumes/depot/iot4agrs/data/students/Alim/pedigree_data/' + '2022 YS_HIPS_BCS all data.xlsx')
        
        # initialize columns for hybrid/inbred and pedigree
        df['hybrid_or_inbred'] = None
        df['pedigree'] = None
        df['nitrogen_treatment'] = 'no_nitrogen_variants_for_genotyping'       
        
        for row in range(df.shape[0]):
            # get plot id:
            temp_plot_id = df.loc[row, 'Plot']
            if (temp_plot_id >= 6351.0) and (temp_plot_id <= 6494.0): # hybrid vs inbred for 2022                
                df.loc[row, 'hybrid_or_inbred'] = 'hybrid'
            elif (temp_plot_id >= 4351.0) and (temp_plot_id <= 4394.0): # hybrid vs inbred for 2021
                df.loc[row, 'hybrid_or_inbred'] = 'hybrid'
            else:
                df.loc[row, 'hybrid_or_inbred'] = 'inbred'

        # add data for pedigree / variant:
        for row in range(df.shape[0]):
            # get plot id:
            temp_plot_id = df.loc[row, 'Plot']
            # get the pedigree of that corresponding plot id:
            if df.loc[row, 'date'][0:4] == '2022': # go to 2022 df (pedigree_df_2022)
                idx = pedigree_df_2022['Plot'].tolist().index(temp_plot_id)
                
                # add temp pedigree to the df:
                temp_pedigree = pedigree_df_2022.loc[idx, 'Pedigree']
                
                # add temp pedigree to df:
                df.loc[row, 'pedigree'] = temp_pedigree
            if df.loc[row, 'date'][0:4] == '2021': # go to 2021 df (pedigree_df_2021)
                idx = pedigree_df_2021['Plot'].tolist().index(temp_plot_id)
                temp_pedigree = pedigree_df_2021.loc[idx, 'Pedigree']

                # add temp pedigree to the df:
                df.loc[row, 'pedigree'] = temp_pedigree                

    if data_path == '2022_f54':
        # to append hybrid/inbred, pedigree, and nitorgen treatment type for field 54.
        df['hybrid_or_inbred'] = 'no_variants_for_nitrogen_treatment'
        df['pedigree'] = 'no_variants_for_nitrogen_treatment'
        df['nitrogen_treatment'] = None
        nitrogen_treatment_file = '/Volumes/depot/iot4agrs/data/students/Alim/nitrogen_data/nitrogen_rate.xlsx'
        df_nitrogen_treatment = pd.read_excel(nitrogen_treatment_file)

        for row in range(df.shape[0]):
            # get plot id:
            temp_plot_id = df.loc[row, 'Plot']
            
            # get nitrogen treatment for that plot from df_nitrogen_treatment
            temp_nitrogen_rate = df_nitrogen_treatment[df_nitrogen_treatment['Plot'] == temp_plot_id]['Nit_rate']
            
            # add the nitrogen tretment to the dataframe we want to return
            df.loc[row, 'nitrogen_treatment'] = int(temp_nitrogen_rate.values)
            

            
    if debug:
        print("FINAL SUMMARY")
        print("Dimensions of returned df:", df.shape)
        print(df.head())
        print(df.tail())
        print(df.describe())
        print("Unique Dates in df:", df['date'].unique())
        print("Columns that contain NaN values:", df.columns[df.isna().any()].tolist())
        print("creating csv of df for debugging...")
        df.to_csv('debugging_df_' + data_path + '.csv')
        print("Final shape of data:", df.shape)
        print("Columns in df:", df.columns)

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
        print("Final df shape:", df.shape)
        

    return df



if __name__ == "__main__":
    df = get_svr_features(debug=True, data_path='2022_f54')
    total_nan_count = df.isna().sum().sum()
    print('nan count is', total_nan_count)

    # read_hyperspectral_per_plot_reflectance()

    #df = read_lidar_per_plot(debug=True)


   #read_ground_truth(debug=True)


