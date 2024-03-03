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
                    heights.append(np_img.shape[1])
                    widths.append(np_img.shape[2])
                    px_min.append(np_img.min())
                    px_max.append(np_img.max())
                    if np_img.shape[1] < 70:
                        print(file)
                        print(np_img.shape)

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


