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

from PIL import Image

import sys
sys.path.append('..')

from dataloading_scripts.load_plots import load_plots_coords_for_field, load_individual_plot_xyxy

print('imported')

hyp_path = '/Volumes/depot/iot4agrs/data/sensor_data/2021_field72/20210727_f72e_india_44m/vnir/processed/elm_mosaic/'
hyp_file = hyp_path + 'seam_mosaic'
hyp_path_local = '/Users/alim/Documents/prototyping/research_lab/hyperspectral_data/2021_field72/20210727_f72e_india_44m/vnir/elm_mosaic/'
hyp_file_local = hyp_path_local + 'seam_mosaic'


ds = gdal.Open(hyp_file_local)


def read_hyperspectral_data(folder, path = hyp_file_local, num_bands = 135, debug=True, save=True):

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
        np.save('/Users/alim/Documents/prototyping/research_lab/HIPS_Hyperspectral/' + folder + '/numpy_hyp.npy', img)
        np.save('/Users/alim/Documents/prototyping/research_lab/HIPS_Hyperspectral/' + folder + '/numpy_freq.npy', freq)
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

def read_hyp_np(file):
    """
    Read a .npy file, return it loaded.
    """
    print('about to load file...')
    loaded_file = np.load(file)   
    return loaded_file

def read_freq_np(file):
    """
    Read the frequency/wavelength data associated with a hyperspectral file in a .npy
    """
    print('about to load freq')
    loaded_file = np.load(file)
    return loaded_file

def get_transformation_matrix(ds = ds):
    """
    The function expects an opened ds object to be passed in. ds object can be created via ds = gdal.Open(fp)
    Transform as shown in Geotransform tutorial on GDAL website (https://gdal.org/tutorials/geotransforms_tut.html):
    X_geo = GT(0) + X_pixel * GT(1) + Y_line * GT(2)
    Y_geo = GT(3) + X_pixel * GT(4) + Y_line * GT(5)

    This is essentially an Ax + b transform

    A is a 2 x 2 matrix in the format np.array([[GT1, GT2], 
                                                [GT4, GT5]])
    x = [X_pixel, Y_line].T
    b = [GT0, GT3].T
    """
    # get transformation matrix parametres:

    transform_params = ds.GetGeoTransform()

    # arrange parameters into a matrix and offset:
    A = np.array([[transform_params[1], transform_params[2]], 
                    [transform_params[4], transform_params[5]]])
    b = np.array([transform_params[0], transform_params[3]]).T

    return A, b

def transform_image_and_extract_plot(A, b, np_img_coords, plot_json, index):
    """
    This function expects a np object with hyperspectral data to generate an empty np array with the right x/y shape.
    Each index in the numpy array will be linearly transformed using the A and b parameters. This transform represents 
    the pixel coordinate in EPSG 26916 (NAD 83 UTM 16). Then, using the plot_json object, we can understand the plot boundaries
    for each plot id. We can then add a TRUE or FALSE into np_geo_coords, which represents whether a plot is contained within 
    a certain region of a hyperspectral image. 
    to a new numpy array. The values in 


    These coordinates can then be used with the plot boundaries to reference which indices to "crop" from the original 
    numpy array.

    The final image can be converted to a torch tensor for use in a deep learning algorithm!
    """
    np_geo_coords = np.empty((np_img_coords.shape[1], np_img_coords.shape[2])) # first axis of the numpy object is the number of channels. This is 
    # irrelevant for this transformation.
    plot_json = load_plots_coords_for_field(field='hips_2021') # load plot json for the image.
    x0, y0, x1, y1, plot_id, plot_row = load_individual_plot_xyxy(plot_json=plot_json, index=index) # get the boundaries for the plot

    # below, we iterate through each pixel, figure out if pixel is in the boundary for the plot.
    for row in range(np_img_coords.shape[1]):
        for col in range(np_img_coords.shape[2]):
            x = np.array([row, col]).T
            transformed_index = A @ x + b # a 2d vector with the [x,y] in NAD83 UTM 16
            
            if transformed_index[0] >= x0 and transformed_index[0] <= x1 \
            and transformed_index[1] >= y0 and transformed_index[1] <= y1: # we are inside the plot boundary.
                np_geo_coords[col, row] = True # need to understand why [col, row] extracts plots in the correct orientation!!
    
    print('number of pixels extracted for plot id and row', plot_id, plot_row, ':', np.count_nonzero(np_geo_coords))

    mask = np_geo_coords.astype(bool)
    print(mask)
    print(mask.shape)
    cropped_plot = np_img_coords[:, mask]
    print('this is cropped plot', cropped_plot.shape)

    return np_geo_coords


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

def open_rgb_orthos():
    hyp_path_local_all = '/Users/alim/Documents/prototyping/research_lab/HIPS_Hyperspectral/'
    local_dirs = os.listdir(hyp_path_local_all)
    local_dirs.remove('.DS_Store')
    local_dirs.sort()
    for folder in local_dirs:
        jpg_file_path = hyp_path_local_all + folder + '/seam_mosaic.jpg'
        PIL_img = Image.open(jpg_file_path)
        np_img = np.uint8(PIL_img)
        print(np_img.shape)

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

    #open_rgb_orthos()

    #open_and_visualize_lidar()
    hyp_path_local_all = '/Users/alim/Documents/prototyping/research_lab/HIPS_Hyperspectral/'

    #plot level directories:
    plot_path_root = '/Users/alim/Documents/prototyping/research_lab/HIPS_grid/'
    plot_path_2022_hips = 'hips_2022_plot_extraction_Alim_modified_on_gsheets_20220609_f78_hips_manshrink.csv'
    plot_path_2021_hips = '20210617_india_f42mYS_HIPS_1cm_noshrinkWei.txt'
    ####

    local_dirs = os.listdir(hyp_path_local_all)
    local_dirs.remove('.DS_Store')
    local_dirs.sort()
    print(local_dirs)
    hyp_plot_li = []
    for folder in local_dirs:
        print('in', folder)
        if folder[0:4] == '2021':
            path = hyp_path_local_all + folder + '/seam_mosaic'
            #read_hyperspectral_data(folder, path = path, save=False)
            hyp = read_hyp_np(hyp_path_local_all + folder + '/numpy_hyp.npy')
            #freq = read_freq_np(hyp_path_local_all + folder + '/numpy_freq.npy')
            plot_data = load_plots_coords_for_field(path = plot_path_root + plot_path_2021_hips, field='hips_2021', img_coords = False, geo_coords=True)

            # get transformation matrix:

            ds = gdal.Open(path)
            A, b = get_transformation_matrix(ds)
            # do the transform on an image coordinate:
            print('doing transform!!!')
            for i in range(len(plot_data['features'])):
                output = transform_image_and_extract_plot(A, b, np_img_coords=hyp, plot_json=plot_data, index=i)
                print(output)
                print(output.shape)
                hyp_plot_li.append(output)
                if i == 2:
                    break
            break
    
    plt.imshow(np.uint8(hyp_plot_li[0][2000:2500, 1200:1500 ]))
    plt.savefig('test_plot0.jpg')
    plt.imshow(np.uint8(hyp_plot_li[1][2000:2500, 1200:1500 ]))
    plt.savefig('test_plot1.jpg')
    plt.imshow(np.uint8(hyp_plot_li[2][2000:2500, 1200:1500 ]))
    plt.savefig('test_plot2.jpg')


            # # the images spatial reference is:
            # # Get the image's spatial reference
            # print('images srs')
            # image_srs = osr.SpatialReference()
            # image_srs.ImportFromWkt(ds.GetProjection())

            # # Define the target NAD83 UTM spatial reference
            # target_srs = osr.SpatialReference()
            # target_srs.SetWellKnownGeogCS("NAD83")
            # target_srs.SetUTM(16, True)  # Set UTM zone and hemisphere 

            # # Create a coordinate transformation object
            # transform = osr.CoordinateTransformation(image_srs, target_srs)

            # # Convert image coordinates to NAD83 UTM coordinates
            # x_image, y_image = 500164, 4480666  # Example image coordinates
            # x_utm, y_utm, _ = transform.TransformPoint(x_image, y_image)
            # print(x_utm, y_utm, _)
            # print('heRE!!!!')
            






        # if folder[0:4] == '2022' and folder[-4:] != '0831':
        #     hyp = read_hyp_np(hyp_path_local_all + folder + '/numpy_hyp.npy')
        #     freq = read_freq_np(hyp_path_local_all + folder + '/numpy_freq.npy')
        #     plot_data = load_plots_coords_for_field(path = plot_path_root + plot_path_2022_hips)
        #     print(plot_data.shape)
        #     print(hyp.shape)
        


        # if folder == '20220623':
        #     path = hyp_path_local_all + folder + '/20220623_vnir_60m_1110_vnirMosaic_4cm'
        # if folder == '20220710':
        #     path = hyp_path_local_all + folder + '/20220710_rgb_lidar_vnir_44m_1255_vnirMosaic_4cm'
        # if folder == '20220831':
        #     print('debugging this file... skipping opening for now...')


