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

import time

import rasterio

print('imported')

hyp_path = '/Volumes/depot/iot4agrs/data/sensor_data/2021_field72/20210727_f72e_india_44m/vnir/processed/elm_mosaic/'
hyp_file = hyp_path + 'seam_mosaic'
hyp_path_local = '/Users/alim/Documents/prototyping/research_lab/hyperspectral_data/2021_field72/20210727_f72e_india_44m/vnir/elm_mosaic/'
hyp_file_local = hyp_path_local + 'seam_mosaic'

ds = gdal.Open(hyp_file_local)


def read_hyperspectral_data(path = hyp_file_local, num_bands = 135, debug=True, save=True):

    data = gdal.Open(path)
    
    if debug: print('file opened')

    x = data.RasterXSize # width
    y = data.RasterYSize # height
    dim = data.RasterCount

    if debug:
        print('getting basic info...')
        print('shape of raster:', dim, y, x)

    # initialize empty array to hold image
    img = np.empty([data.RasterCount, data.RasterYSize, data.RasterXSize])

    # initialize empty array to hold band location in EM spectrum
    freq = np.empty(data.RasterCount)

    # get projection of hyperspectral data:
    spatialRef = data.GetSpatialRef()
    if debug:
        print('spatial info:', spatialRef)
        print(type(spatialRef))
    t1 = time.time()
    for n, i in enumerate(range(num_bands)): # update range to dim to loop through entire raster
        band = data.GetRasterBand(i+1)  # Band indexing is 1-based in GDAL
        band_data = band.ReadAsArray()
        img[n,:,:] = band_data # note, this is not the ordering for displaying and image,
        # but it is must faster to write the object with n, :, : vs :, :, n. Not sure why. 
        freq[n]    = band.GetDescription()
        if debug:
            print('max value of band:', band_data.max())
            print('min value of band:', band_data.min())
            print('added raster of shape', band_data.shape, 'to array')
            print('it was at', freq[n], 'nm...')
    t2 = time.time()
    print('took ', np.abs(t1 - t2), 'seconds to go open and hold all bands in memory') 

    if save:
        print('saving files as .npy...')
        t1 = time.time()
        np.save('/Users/alim/Documents/prototyping/research_lab/hyperspectral_data/numpy_hyp.npy', img)
        np.save('/Users/alim/Documents/prototyping/research_lab/hyperspectral_data/numpy_freq.npy', freq)
        t2 = time.time()
        if debug:
            print('done saving files... it took', np.abs(t1-t2), 'seconds.')

    return img, freq

def get_visual_hyperspectral_rgb(freq, hyp_im):
    """
    Pass in a numpy array in format h x w x c and the frequency data.
    
    This function will visualize rgb channels from those two objects.

    Returns visualizable image
    """
    if hyp_im.shape[2] != 3:
        hyp_im = hyp_im.transpose(1,2,0)

    r = np.mean(hyp_im[:,:,np.logical_and(freq>640,freq<670)],axis=2)
    g = np.mean(hyp_im[:,:,np.logical_and(freq>530,freq<590)],axis=2)
    b = np.mean(hyp_im[:,:,np.logical_and(freq>450,freq<520)],axis=2)
    img = 1.0 * np.dstack((r,g,b))
    img = (img-img.min())/(img.max()-img.min())
    return img


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


def open_and_visualize_lidar(visualize=False):

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
    print(header.parse_crs()) # doesn't exist as an attribute
    print(dir(header))

    points = lidar_opened.points

    if visualize:
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

if __name__ == "__main__":
    # #hyp_img, freq = read_hyperspectral_data()
    # path_hyp = '/Users/alim/Documents/prototyping/research_lab/hyperspectral_data/numpy_hyp.npy'
    # path_freq = '/Users/alim/Documents/prototyping/research_lab/hyperspectral_data/numpy_freq.npy'
    # t1 = time.time()
    # hyp_im = np.load(path_hyp)
    # freq = np.load(path_freq)
    # t2 = time.time()
    # print('took', np.abs(t1-t2), 'to load data')

    # t1 = time.time()
    # img_rgb = get_visual_hyperspectral_rgb(freq, hyp_im)
    # t2 = time.time()
    # print('took', np.abs(t1-t2), 'to get visual rgb')
    # plt.imshow(img_rgb)
    # plt.savefig('visualize_hyperspectral.jpg')

    open_and_visualize_lidar()

