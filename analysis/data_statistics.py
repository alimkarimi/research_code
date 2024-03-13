# goal of this script is to understand the average plot dimensions and be able to come up with a reasonable input image size and 
# kernel configuration for any convolution operations in the networks.

# iteratre through every folder, load each numpy file in the folder, put h / w into a list, and take the mean of the list and
# plot the values on a histogram.

# decision is to have input of 136 x 130 x 42

import os
import numpy as np
import matplotlib.pyplot as plt


base_path = '/Users/alim/Documents/prototyping/research_lab/HIPS_Hyperspectral/'

folders = os.listdir(base_path)

if '.DS_Store' in folders:
    folders.remove('.DS_Store')

folders.sort()

heights = []
widths = []
px_min = []
px_max = []
channel_mean_values = np.empty(136)

##### - USE THESE T/F Flags to set which statistics et computed
img_dims = False
channel_means = False
lidar_data = True
##### - USE THESE T/F Flags to set which statistics et computed
mean_count = 0

def compute_hyperspectral_statistics(img_dims : bool, channel_means : bool, mean_count, channel_mean_values):
    for folder in folders:
        files_in_folder = os.listdir(base_path + folder)
        #print(files_in_folder)
        #filtered_list = [item for item in my_list if '.npy' in item]
        for file in files_in_folder:
            if '.npy' in file and 'mosaic' not in file:
                np_img = np.load(base_path + folder + '/' + file)
                #print(np_img.shape)
                if np_img.ndim == 3: # sanity check not to get the entire field
                    if np_img.shape[1] < 500:
                        if img_dims:
                            print(np_img.shape)
                            heights.append(np_img.shape[1])
                            widths.append(np_img.shape[2])
                            px_min.append(np_img.min())
                            px_max.append(np_img.max())

                        if channel_means:
                            temp_channel_mean_values = np.mean(np_img[0:136], axis = (1,2))
                            mean_count  = mean_count + 1 # update number of images we have computed the mean for
                            if mean_count == 1:
                                channel_mean_values = temp_channel_mean_values
                            if mean_count > 1: 
                                channel_mean_values = channel_mean_values + ((temp_channel_mean_values - channel_mean_values)/ mean_count)
                                # assume mean is 3. 
                                # 3 + 1 / 2. Mean is now 2 (4 /2 )
                                # New number is 1. 
                                # (2 + 1) * 2 / 3 is 

    if channel_means:
        np.save('channel_means.npy', channel_mean_values)
        return channel_mean_values, mean_count
                            
                            # online update:
                            # mu_k+1 = mu_k + (1/k+1) * (mu_k+)

def compute_lidar_statistics():
    num_samples = 0
    x_mean = y_mean = z_mean = 0
    
    # open lidar folders:
    lidar_path_local_all = '/Users/alim/Documents/prototyping/research_lab/HIPS_LiDAR/'
    local_dirs = os.listdir(lidar_path_local_all)
    local_dirs.remove('.DS_Store')
    local_dirs.sort()
    print(local_dirs)
    for n, folder in enumerate(local_dirs):
        file_list = os.listdir(lidar_path_local_all + folder)

        # remove .las files:
        for m, file in enumerate(file_list):
            if '.npy' not in file:
                continue # move to next file
            else:
                temp_point_cloud = np.load(lidar_path_local_all + folder + '/' + file)
                print(temp_point_cloud.shape)
                if (n == 0) and (m == 0):
                    x_mean = np.mean(temp_point_cloud[:,0])
                    y_mean = np.mean(temp_point_cloud[:,1])
                    z_mean = np.mean(temp_point_cloud[:,2])
                    num_samples = temp_point_cloud.shape[0]

                else:
                    # compute online mean update: (current_mean * number of samples so far + sum of new batch) / (number of samples so far + new batch size)
                    x_mean = (x_mean * num_samples + np.sum(temp_point_cloud[:,0])) / (num_samples + temp_point_cloud.shape[0])
                    y_mean = (y_mean * num_samples + np.sum(temp_point_cloud[:,1])) / (num_samples + temp_point_cloud.shape[0])
                    z_mean = (z_mean * num_samples + np.sum(temp_point_cloud[:,2])) / (num_samples + temp_point_cloud.shape[0])
                    num_samples = num_samples + temp_point_cloud.shape[0]

    return x_mean, y_mean, z_mean, num_samples







        


if lidar_data:
    x_mean, y_mean, z_mean, num_samples = compute_lidar_statistics()
    print("x_mean:", x_mean, "y_mean:", y_mean, "z_mean", z_mean, "num_samples", num_samples)

if channel_means:
    channel_mean_values = np.empty(136)
    mean_count = 0
    channel_mean_values, mean_count = compute_hyperspectral_statistics(img_dims = False, channel_means= True, mean_count=mean_count,
     channel_mean_values = channel_mean_values)
    print(mean_count)
    print(channel_mean_values)






if img_dims:
    compute_hyperspectral_statistics(img_dims = True, channel_means=False)

    ### MIN/MAX VALUE ACROSS ALL HIPS:
    print(max(px_max), 'is the max px value!')
    print(min(px_min), 'is the min px value!')

    max_pixel_value_HIPS = max(px_max) # 8470.0
    min_pixel_value_HIPS = min(px_min) # -555.0

    ### MIN/MAX PLOT SIZES ACROSS ALL HIPS



    # Plot histogram
    plt.hist(heights, bins=100, edgecolor='black')
    plt.xlabel('Plot height (y) size')
    plt.ylabel('Frequency')
    plt.title('Histogram of Plot Height (y) Size')
    #plt.grid(True)
    plt.savefig('Heights_hist.jpg')

    # Clear the current plot
    plt.clf()

    plt.hist(widths, bins=100, edgecolor='black')
    plt.xlabel('Plot width (x) size')
    plt.ylabel('Frequency')
    plt.title('Histogram of Plot Width (x) Size')
    #plt.grid(True)
    plt.savefig('Widths_hist.jpg')


    plt.clf()

    plt.hist(px_min, bins=100, edgecolor='black')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram of Pixel Min Vals')
    #plt.grid(True)
    plt.savefig('Pixel_mins.jpg')


    plt.clf()

    plt.hist(px_max, bins=100, edgecolor='black')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram of Pixel Max Vals')
    #plt.grid(True)
    plt.savefig('Pixel_maxes.jpg')


