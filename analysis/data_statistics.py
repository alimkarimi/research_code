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
img_dims = False
channel_means = True
mean_count = 0

def compute_statistics(img_dims : bool, channel_means : bool, mean_count, channel_mean_values):
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


if channel_means:
    channel_mean_values = np.empty(136)
    mean_count = 0
    channel_mean_values, mean_count = compute_statistics(img_dims = False, channel_means= True, mean_count=mean_count,
     channel_mean_values = channel_mean_values)
    print(mean_count)
    print(channel_mean_values)






if img_dims:
    compute_statistics(img_dims = True, channel_means=False)

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


