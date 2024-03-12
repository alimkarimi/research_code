import os
from osgeo import gdal
from osgeo import ogr, osr
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import ScalarFormatter


import spectral
import spectral.io.envi as envi
from spectral import open_image

import laspy
from laspy.file import File
import numpy as np

import time

import rasterio
import re

from PIL import Image

import sys
sys.path.append('..')

from dataloading_scripts.load_plots import (load_plots_coords_for_field, 
load_individual_plot_xyxy, plot_path_2022_hips, plot_path_2021_hips, load_entire_plot_xyxy)

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
        if folder != '20220831':
            freq[n]    = band.GetDescription()
        else:
            temp_band_desc = band.GetDescription()
            print(type(temp_band_desc))
            print('this is temp_band_desc' , temp_band_desc)
            #pattern = r'(\d+\.\d+)'
            pattern = r'(?<!\d\.)(?<!\d)(?:[2-9]\d{2,}\.\d+|[2-9]\d{2,}|[1-9]\d{3,}\.\d+|[1-9]\d{3,})'


            # Extract numerical value using regex
            numerical_band_value = re.search(pattern, temp_band_desc)
            print(numerical_band_value.group())
            
            freq[n] = numerical_band_value.group()
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
        if os.path.exists('/Users/alim/Documents/prototyping/research_lab/HIPS_Hyperspectral/' + folder + '/numpy_hyp.npy'):
            print('numpy file already saved...')
        else:
            np.save('/Users/alim/Documents/prototyping/research_lab/HIPS_Hyperspectral/' + folder + '/numpy_hyp.npy', img)
        np.save('/Users/alim/Documents/prototyping/research_lab/HIPS_Hyperspectral/' + folder + '/numpy_freq.npy', freq)
        t2 = time.time()
        if debug:
            print('done saving files... it took', np.abs(t1-t2), 'seconds.')

    return img, freq

def get_visual_hyperspectral_rgb(hyp_im, freq):
    """
    Pass in a numpy array in format h x w x c and the frequency data. c can be many channels.
    
    This function will visualize rgb channels from those two objects.

    Returns visualizable image
    """

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

def save_extracted_plot(base_path, folder, filename, extracted_plot, extension='.npy'):
    np.save(file = base_path + folder + filename + extension, arr = extracted_plot)
    print('saved file as', base_path + folder + filename + extension)

def save_rgb(base_path, folder, filename, get_visual_hyperspectral_rgb, hyp_im, freq, extension='.jpg'):
    """
    Note that get_visual_hyperspectral_rgb() expects data in h x w x c
    """
    if hyp_im.shape[2] != 136:
        hyp_im = hyp_im.transpose(1,2,0)
    rgb_data = get_visual_hyperspectral_rgb(hyp_im, freq)
    # rgb data is returned between 0 and 1
    rgb_data = (rgb_data * 255).astype(np.uint8)

    im = Image.fromarray(rgb_data)
    im.save(base_path + folder + filename + extension)

def get_inverse_transformation_matrix(ds = ds):
    """
    Get the inverse affine transform to go from projected coordinate to pixel coordinate:

    How to use: A_inv @ [x_utm, y_utm] + b_inv = [x_img, y_img]
    """
    A, b = get_transformation_matrix(ds)
    A_inv = np.linalg.inv(A)
    b_inv = -A_inv @ b
    return A_inv, b_inv

def get_transformation_matrix(ds = ds):
    """
    Goes from pixel coordinates to projected coordinates.
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
    print(transform_params)

    A = np.array([[transform_params[1], transform_params[2]], 
                     [transform_params[4], transform_params[5]]])
    b = np.array([transform_params[0], transform_params[3]])

    return A, b

def transform_image_and_extract_plot(A, b, np_img_coords, plot_json, index, folder, freq, field, loop=False, direct=False, 
                                    use_px_coords = True, extract_both_rows = True, debug=False, save=False):
    """
    This function expects a np object with hyperspectral data to generate an empty np array with the right rows/cols shape.
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
    plot_json = load_plots_coords_for_field(field=field) # load plot json for the image.

    if extract_both_rows:
        x0, y0, x1, y1, queried_plot_id = load_entire_plot_xyxy(plot_json, plot_id_query, field)
        # update so that we iterate through plot_ids instead of index??
        # need to think about this. 
    else:
        x0, y0, x1, y1, plot_id, plot_row = load_individual_plot_xyxy(plot_json=plot_json, index=index, field=field) # get the boundaries for the plot
        print(x0, y0,' this is x0 and y0')
        print(A, b)
        b = b.reshape(2,1)
        x0y0 = A @ np.array([[x0], [y0]]).reshape(2,1) + b
        x1y1 = A @ np.array([[x1], [y1]]).reshape(2,1) + b
        print(x0y0, '\ninverted to pixel coords!!')
        print(x1y1)

    if use_px_coords:
        # this option is to use the inverse transform, and appy operations directly on pixel coordiantes. 
        x0 = int(np.floor(x0y0[0])) # column 1501 
        y0 = int(np.ceil(x0y0[1])) # row 1417
        x1 = int(np.ceil(x1y1[0])) # column 1521
        y1 = int(np.floor(x1y1[1])) # row 1287
        print(x0, y0, x1, y1)
        print(np_img_coords.shape[1], np_img_coords.shape[2]) # 2676 rows, 3352 columns
        # Create a boolean mask array initialized with False

        mask = np.zeros((136, np_img_coords.shape[1], np_img_coords.shape[2]), dtype=bool)
        # print(mask.shape)

        # Set pixels within the boundary to True
        mask[:, y1:y0+1, x0:x1+1] = True
        cropped_plot = np_img_coords[:, y1:y0 + 1, x0:x1 + 1]
        print('cropped_plot', cropped_plot.shape)

        if save:
            save_extracted_plot(base_path = '/Users/alim/Documents/prototyping/research_lab/HIPS_Hyperspectral/', 
                folder= folder + '/', filename=str(plot_id) + '_' + str(plot_row), extracted_plot=cropped_plot)

            save_rgb(base_path = '/Users/alim/Documents/prototyping/research_lab/HIPS_Hyperspectral/', 
                folder= folder + '/', filename=str(plot_id) + '_' + str(plot_row), hyp_im = cropped_plot, 
                get_visual_hyperspectral_rgb=get_visual_hyperspectral_rgb, freq=freq)


        return mask, cropped_plot, 'placeholder'
        

    if direct:

        x = np.indices((np_img_coords.shape[1], np_img_coords.shape[2])) # generate indices for the original image.
        print('original x shape:', x.shape)
        x = x.reshape(2, -1)
        # For a 2 x 2 image, the indices are (0,0), (0,1), (1,0), (1,1).
        # These indices can all be transformed at once using the A matrix operator - this is more efficient than a for loop! 
        print(x)

        print('b is', b, b.shape)
        print(b.reshape(2,1))
        
        transformed_coordinates = A @ x + b.reshape(2,1) 
        print('applied transform!')
        print('transformed result shape:', transformed_coordinates.shape)


        # create mask on transformed coordinates. 
        mask = (transformed_coordinates[0] >= x0) & (transformed_coordinates[0] <= x1) & (transformed_coordinates[1] >= y0)  & \
            (transformed_coordinates[1] <= y1)
        
        print('created mask', mask, mask.shape)
        # reshape mask back to the original hyperspectral image shape:
        mask = mask.reshape(np_img_coords.shape[1], np_img_coords.shape[2])
        print('unique vals in mask:', np.unique(mask))

        cropped_plot = np_img_coords * mask
        print(cropped_plot.shape)
        print('number of pixels extracted for plot id and row', plot_id, plot_row, ':', np.count_nonzero(cropped_plot))

        return np_geo_coords, cropped_plot, 'placeholder'

    if loop: # note that this loop is not efficient...
    #below, we iterate through each pixel, figure out if pixel is in the boundary for the plot.
        for row in range(np_img_coords.shape[1]):
            for col in range(np_img_coords.shape[2]):
                x = np.array([row, col]).T


                transformed_index = A @ x + b # result is 2d vector with the [x,y] in NAD83 UTM 16. Note, x is the column, y is the row.

                
                if transformed_index[0] >= x0 and transformed_index[0] <= x1 \
                and transformed_index[1] >= y0 and transformed_index[1] <= y1: # we are inside the plot boundary.
                    np_geo_coords[col, row] = True # Note, this has to be col, row to keep the correct orientation. 
                    # when we perform the if statement above, x0/x1 are the COLUMNS, y0/y1 are the ROWS. 
                    print('found px in boundary at row/col', row, col)
                           

        mask = np_geo_coords.astype(bool)

        plot_in_field = np_img_coords * mask
        print('this is cropped plot', plot_in_field.shape)
        print('number of pixels extracted for plot id and row', plot_id, plot_row, ':', np.count_nonzero(plot_in_field))

        # now, we actually drop the pixels where the mask is false.

        boundary = np.where(mask == True) # boundary returns a list of x and y coordiantes. boundary[0] are the x's, boundary[1] are the 
        # corresponding y's
        if debug:
            print(boundary)
            print(boundary[0].shape)
            print(boundary[1].shape)
            print('boundary for top left:')
            print('boundary for bottom right??')
            print(boundary[0][-1], boundary[1][-1])
            print(boundary[0][0], boundary[1][0])

        top_left_coordinates = (boundary[0][0], boundary[1][0] + 1)
        bottom_right_coordinates = (boundary[0][-1], boundary[1][-1] + 1)
        cropped_plot = plot_in_field[:, top_left_coordinates[0] : bottom_right_coordinates[0], top_left_coordinates[1] : bottom_right_coordinates[1]]
        print('extracted the following shape from the original hyperspectral image:', cropped_plot.shape)

        if save:
            save_extracted_plot(base_path = '/Users/alim/Documents/prototyping/research_lab/HIPS_Hyperspectral/', 
                            folder= folder + '/', filename=str(plot_id) + '_' + str(plot_row), extension='.npy',
                            extracted_plot=cropped_plot)
        
        return np_geo_coords, cropped_plot, plot_in_field 


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
    """
    This function prints out the dimensions of the seam_mosaic.jpg files. It is not needed in the processing, just a sanity checker. 
    """
    hyp_path_local_all = '/Users/alim/Documents/prototyping/research_lab/HIPS_Hyperspectral/'
    local_dirs = os.listdir(hyp_path_local_all)
    local_dirs.remove('.DS_Store')
    local_dirs.sort()
    for folder in local_dirs:
        jpg_file_path = hyp_path_local_all + folder + '/seam_mosaic.jpg'
        PIL_img = Image.open(jpg_file_path)
        np_img = np.uint8(PIL_img)
        print(np_img.shape)


def main_hyperspectral_orchestrator():
    plot_path_root = '/Users/alim/Documents/prototyping/research_lab/HIPS_grid/'
    hyp_path_local_all = '/Users/alim/Documents/prototyping/research_lab/HIPS_Hyperspectral/'


    local_dirs = os.listdir(hyp_path_local_all)
    local_dirs.remove('.DS_Store')
    local_dirs.sort()
    print(local_dirs)
    hyp_plot_li = []
    for folder in local_dirs:
        print('in', folder)

        if folder[0:4] == '2021':
            # open the hyperspectral data, corresponding frequencies, and plot data. This will be the same for each folder (i.e, timestep)            
            path = hyp_path_local_all + folder + '/seam_mosaic'
            hyp = read_hyp_np(hyp_path_local_all + folder + '/numpy_hyp.npy')
            freq = read_freq_np(hyp_path_local_all + folder + '/numpy_freq.npy')
            plot_data = load_plots_coords_for_field(path = plot_path_root + plot_path_2021_hips, field='hips_2021', 
                                                    img_coords = False, geo_coords=True)
            print('shape of loaded numpy:', hyp.shape)

            ds = gdal.Open(path)
            
            A, b = get_transformation_matrix(ds)
            A_inv, b_inv = get_inverse_transformation_matrix(ds)
            # do the transform on an image coordinate:

            for i in range(100, 277): # update this range to get all fields.
                output_bools, cropped_plot, entire_field = transform_image_and_extract_plot(A_inv, b_inv, np_img_coords=hyp, 
                plot_json=plot_data, index=i, folder=folder, freq=freq, field='hips_2021', loop=False, direct=False, use_px_coords = True, save=True)
                hyp_plot_li.append(cropped_plot)


        if folder[0:4] == '2022':
            if folder == '20220623':
                path = hyp_path_local_all + folder + '/20220623_vnir_60m_1110_vnirMosaic_4cm'
            if folder == '20220710':
                path = hyp_path_local_all + folder + '/20220710_rgb_lidar_vnir_44m_1255_vnirMosaic_4cm'
            if folder == '20220831':
                path = hyp_path_local_all + folder + '/20220831_rgb_lidar_vnir_44m_1202_vnirMosaic_4cm'
            
            hyp = read_hyp_np(hyp_path_local_all + folder + '/numpy_hyp.npy')
            freq = read_freq_np(hyp_path_local_all + folder + '/numpy_freq.npy')
            plot_data = load_plots_coords_for_field(path = plot_path_root + plot_path_2022_hips, field='hips_2022', img_coords = False, geo_coords=True)
            print(hyp.shape)

            ds = gdal.Open(path)
            
            A, b = get_transformation_matrix(ds)
            A_inv, b_inv = get_inverse_transformation_matrix(ds)
            print('got transformations')
            # do the transform on an image coordinate:

            for i in range(100, 277): # update this range to get all fields.
                output_bools, cropped_plot, entire_field = transform_image_and_extract_plot(A_inv, b_inv, np_img_coords=hyp, 
                plot_json=plot_data, index=i, folder=folder, freq=freq, field='hips_2022', loop=False, direct=False, use_px_coords = True, save=True)
                hyp_plot_li.append(cropped_plot)

def open_and_visualize_lidar(lidar_file_path, display_metadata=False):

    # lidar_path = '/Volumes/depot/iot4agrs/data/sensor_data/2021_field72/20210727_f72e_india_44m/lidar/final/'
    # lidar_path = '/Volumes/depot/iot4agrs/data/sensor_data/2021_field72/20210802_f72w_india_44m/lidar/processed/final/'

    # lidar_file = lidar_path + 'DSM.las'
    lidar_data = laspy.read(lidar_file_path)


    #print(dir(lidar_opened))
    if display_metadata:
        print(lidar_data.x.min(), lidar_data.x.max())
        print('above was x, now printing y')
        print(lidar_data.y.min(), lidar_data.y.max())
        print('now printing z')
        print(lidar_data.z.min(), lidar_data.z.max())
        #print(dir(lidar_opened.header))
        # Access header information
        header = lidar_data.header
        print("Header information:")
        print("Number of points:", header.point_count)
        print("Point format:", header.point_format)
        print("scales:", header.scales, header.y_scale, header.z_offset)
        print('crs????')
        print(header.parse_crs()) # doesn't exist as an attribute
        #print(dir(header))
    
    return lidar_data

def transform_pointcloud_and_extract_plot(A, b, lidar_data, plot_json, index, field, folder=None, debug=False):
    """
    A and b are transformation matrices. Not needed, since points are already in NAD83 UTM16. 

    lidar_data is a .las file opened.

    plot_json is a json with information about NAD 83 UTM 16 coordinates 
    """
    x_pts = np.array(lidar_data.x)
    y_pts = np.array(lidar_data.y)
    z_pts = np.array(lidar_data.z)
    if debug:
        print(x_pts, y_pts)
        print(x_pts.shape, y_pts.shape)
        print(lidar_data.header.point_count)

    x0, y0, x1, y1, plot_id, row= load_individual_plot_xyxy(plot_json, index, field = field)
    if debug:
        print(x0, y0, x1, y1, plot_id, row)
    indices_within_boundary = (x_pts >= x0) & (x_pts <= x1) & (y_pts >= y0) & (y_pts <= y1) & (z_pts > -9999)  
    cropped_plot_points = lidar_data.points[indices_within_boundary]

    return cropped_plot_points

def visualize_lidar_points(lidar_data, filename='lidar.jpg', save=False):
    x = lidar_data.x
    y = lidar_data.y
    z = lidar_data.z

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    sc = ax.scatter(x, y, z, s=0.5, c=z, cmap='viridis') # s is the size of the point in the plot. c controls which point to base cmap on.
    ax.set_box_aspect([1, (np.max(y) - np.min(y))/(np.max(x)-np.min(x)), 1])

    # Set labels and title
    ax.set_xlabel('X', fontsize=6)
    ax.set_ylabel('Y', fontsize=6)
    ax.set_zlabel('Z', fontsize=6)
    ax.set_title('LiDAR Point Cloud Visualization')
    # Set tick label size
    ax.tick_params(axis='x', labelsize=6)
    ax.tick_params(axis='y', labelsize=6)
    ax.tick_params(axis='z', labelsize=6)
    # Disable scientific notation for tick labels
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.zaxis.set_major_formatter(ScalarFormatter(useMathText=False))

    # Show elevation scale:
    cbar = fig.colorbar(sc, shrink=0.5)
    cbar.set_label('Z Value', rotation=270, labelpad=10)

    if save:
        fig.savefig(filename)

def main_lidar_orchestrator():

    # Flow to extract LiDAR pt cloud around a plot # 
    # get local directories for iteration
    lidar_path_local_all = '/Users/alim/Documents/prototyping/research_lab/HIPS_LiDAR/'
    local_dirs = os.listdir(lidar_path_local_all)
    local_dirs.remove('.DS_Store')
    local_dirs.sort()
    print(local_dirs)

    for folder in local_dirs:
        print('in folder', folder)
        if folder[0:4] == '2021':
            field = 'hips_2021'
        if folder[0:4] == '2022':
            field = 'hips_2022'
        plot_json = load_plots_coords_for_field(field=field, geo_coords=True) # load plot json for all of 
        # hips 2021 or 2022.
        file_in_folder = os.listdir(lidar_path_local_all + folder) # list files for particular flight. 
        print(file_in_folder)

        # get .las file from that file_in_folder list.
        las_file_idx = 0
        for n, file in enumerate(file_in_folder):
            if '.las' in file:
                las_file_idx = n

        full_lidar_fp = lidar_path_local_all + folder + '/' + file_in_folder[las_file_idx] 
        lidar_data = open_and_visualize_lidar(full_lidar_fp)
        A = b = 0
        for i in range(100,277): # 100 to 277 are the idxs of the plots in the experiment, as represented in the plot_json
            cropped_points = transform_pointcloud_and_extract_plot(A, b, lidar_data, plot_json, field = field, index = i)
            # cropped points contain the x,y, and z points for the lidar file. 
            x_pts = np.array(cropped_points.x)
            y_pts = np.array(cropped_points.y)
            z_pts = np.array(cropped_points.z)

            xyz_pts_stacked = np.stack([x_pts, y_pts, z_pts], axis = 1) # create a stacked 
            
            # the plot and row info gathered here is for file naming purposes.
            out = load_individual_plot_xyxy(plot_json=plot_json, field=field, index=i)
            x0, y0, x1, y1, plot, row = out
            print('saving pts in .npy for plot', plot, 'and row',row)
            np.save(lidar_path_local_all + folder + '/lidar_xyz_' + str(plot) + '_' + str(row) + '.npy', xyz_pts_stacked)
            #visualize_lidar_points(cropped_points, 'cropped_lidar_test.jpg', save=False)

if __name__ == "__main__":
    # code to save .npy hyp and freq for 20220831:
    if not os.path.exists('/Users/alim/Documents/prototyping/research_lab/HIPS_Hyperspectral/20220831/' + 'numpy_hyp.npy'):
        base_path = '/Users/alim/Documents/prototyping/research_lab/HIPS_Hyperspectral/20220831/'
        read_hyperspectral_data(folder='20220831', path = base_path + '20220831_rgb_lidar_vnir_44m_1202_vnirMosaic_4cm')

    do_hyp = False
    do_lidar=True

    if do_lidar:
        main_lidar_orchestrator()

    
    if do_hyp:
        main_hyperspectral_orchestrator()

    


