import os
from osgeo import gdal
from osgeo import ogr, osr
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import spectral
import spectral.io.envi as envi
from spectral import open_image

import laspy
from laspy.file import File
import numpy as np

import rasterio

print('imported')

hyp_path = '/Volumes/depot/iot4agrs/data/sensor_data/2021_field72/20210727_f72e_india_44m/vnir/processed/elm_mosaic/'
hyp_file = hyp_path + 'seam_mosaic'
hyp_path_local = '/Users/alim/Documents/prototyping/research_lab/hyperspectral_data/2021_field72/20210727_f72e_india_44m/vnir/elm_mosaic/'
hyp_file_local = hyp_path_local + 'seam_mosaic'


data = gdal.Open(hyp_file_local)
print('file opened')
x = data.RasterXSize # width
y = data.RasterYSize # height
dim = data.RasterCount
print('getting basic info...')
print('shape of raster:', dim, y, x)

band_index = 0  # Change this to the index of the band you want to analyze

# Read the specific band
img = np.empty([dim, y, x])

# get projection of hyperspectral data:
spatialRef = data.GetSpatialRef()
print((spatialRef.SetProjection(4326)))
print(spatialRef)
print(type(spatialRef))

for i in range(1): # update range to dim to loop through entire raster
    band = data.GetRasterBand(i + 1)  # Band indexing is 1-based in GDAL
    #print(band.GetMinimum()) # this implies that min/max is already in metadata
    #print(band.GetMaximum()) # this implies that min/max is already in metadata
    #print(band.ComputeRasterMinMax())
    band_data = band.ReadAsArray()
    img[i,:,:] = band_data
    print('max value of band:', band_data.max())
    print('min value of band:', band_data.min())

# # Display pixel values
# plt.imshow(band_data, cmap='gray')
# plt.colorbar(label='Pixel Value')
# plt.title('Pixel Values in Band {}'.format(band_index + 1))
# plt.xlabel('Column')
# plt.ylabel('Row')
#plt.show()



# open hdr file
# Path to the .hdr file

def read_hdr(hdr_file = None):
    if hdr_file == None:
        hdr_file = hyp_file + '.hdr'
    else: 
        hdr_file = hdr_file
    # Open the .hdr file
    hdr_img = spectral.open_image(hdr_file)
    # Access metadata
    metadata = hdr_img.metadata
    # Print metadata
    print(metadata)




# Open the hyperspectral image
# print('using rasterio...')
# with rasterio.open(hyp_file_local) as src:
#     # Read the image data (bands)
#     hyp_data = src.read()

# # # Plot the image
# plt.imshow(hyp_data.transpose((1, 2, 0)))  # Transpose to match matplotlib's ordering
# plt.colorbar()

# # # Show the plot
# plt.show()

def visualize_lidar():

    lidar_path = '/Volumes/depot/iot4agrs/data/sensor_data/2021_field72/20210727_f72e_india_44m/lidar/final/'
    lidar_path = '/Volumes/depot/iot4agrs/data/sensor_data/2021_field72/20210802_f72w_india_44m/lidar/processed/final/'

    lidar_file = lidar_path + 'DSM.las'
    lidar_opened = laspy.read(lidar_file)

    print(dir(lidar_opened))
    print(lidar_opened.x.min(), lidar_opened.x.max())
    print('above was x, now printing y')
    print(lidar_opened.y.min(), lidar_opened.y.max())
    print('now printing z')
    print(lidar_opened.z.min(), lidar_opened.z.max())

    print(dir(lidar_opened.header))
    # Access header information
    header = lidar_opened.header
    print("Header information:")
    print("Number of points:", header.point_count)
    print("Point format:", header.point_format)
    print("scales:", header.scales, header.y_scale, header.z_offset)

    points = lidar_opened.points
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    ax.scatter(points['x'], points['y'], points['z'], s=0.1, c=points['z'], cmap='viridis')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('LiDAR Point Cloud Visualization')

    fig.savefig('lidar.jpg')
