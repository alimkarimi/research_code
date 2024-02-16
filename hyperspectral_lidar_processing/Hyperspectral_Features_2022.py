# Import modules
import os
import numpy as np
import sys
import time
from copy import deepcopy
from scipy.io import loadmat, savemat
from matplotlib.image import imread
from osgeo import gdal
import re
from matplotlib import pyplot as plt
import matplotlib
#from ipywidgets import *
#%matplotlib inline
from sklearn.decomposition import PCA
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
from sklearn import manifold
from datetime import datetime
import matplotlib.patches as patches
import spectral.io.envi as envi
import spectral
import pywt
import scipy.ndimage
import gc
from pylab import nbytes
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
from scipy.interpolate import griddata, interp2d
from skimage.color import rgb2hsv


#from matplotlib.mlab import PCA

import copy
   
#import grid_ortho2hyper
#reload(grid_ortho2hyper)
#from grid_ortho2hyperUTM import grid_ortho2hyperUTM
from shutil import copyfile, copytree
from spectral import open_image

def show_rgb_from_hyperspec(hyp_im, freq):
    """ Show the RGB component of hyperspectral data"""
    r = np.mean(hyp_im[:,:,np.logical_and(freq>640,freq<670)],axis=2)
    g = np.mean(hyp_im[:,:,np.logical_and(freq>530,freq<590)],axis=2)
    b = np.mean(hyp_im[:,:,np.logical_and(freq>450,freq<520)],axis=2)
    img = 1.0 * np.dstack((r,g,b))
    img = (img-img.min())/(img.max()-img.min())
    return img
    #pp.figure(figsize=(7,4))
    #pp.imshow(img,cmap=None,interpolation=None)
    
def showImBand(band, hyp_im, freq, ax):
    """ Show the image from a particular band of hyperspectral data"""
    #pp.figure(figsize=(5,5))
    y = 1.0*hyp_im[:,:,band]
    y = (y-y.min())/(y.max()-y.min())
    ax.imshow(y,cmap=matplotlib.cm.Greys_r,interpolation=None)
    ax.set_title(str(freq[band]))
    
def convert_hyperspec_image_to_3D_array(ds):
    """ Convert the hyperspectral image read using GDAL to 3D array """
    x = np.zeros((ds.RasterYSize,ds.RasterXSize,ds.RasterCount))
    freq = np.zeros((ds.RasterCount,))
    freq_regexp_nanometers = re.compile(r'\((\d+\.?\d*)\s*Nanometers\)')
    freq_regexp = re.compile(r'\((\d+\.?\d*)\s*.*\)')
    for b in range(ds.RasterCount):
        #print(b)
        band = ds.GetRasterBand(b+1)
        arr = band.ReadAsArray()
        x[:,:,b] = arr
        '''band_desc = band.GetDescription()
        frn = freq_regexp_nanometers.search(band_desc)
        if frn:
            freq[b] = float(frn.group(1))
        else:
            freq_match = freq_regexp.search(band_desc)
            freq[b] = float(freq_match.group(1))'''
            
    return x#, freq

def grid_ortho2hyperUTM(grid,ds,top_x,top_y,res):
    x1 = grid[:,0]
    y1 = grid[:,1]
    x2 = grid[:,2]
    y2 = grid[:,3]
    plot = grid[:,4]
    row = grid[:,5]
    s = len(x1)
    UTM_x1 = np.zeros(s)
    UTM_y1 = np.zeros(s)
    UTM_x2 = np.zeros(s)
    UTM_y2 = np.zeros(s)
    col1 = np.zeros(s)
    col2 = np.zeros(s)
    row1 = np.zeros(s)
    row2 = np.zeros(s)
             

    transform = ds.GetGeoTransform()

    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = -transform[5]

    #data = band.ReadAsArray(0, 0, cols, rows)



    for a in range(0,s):
        UTM_x1[a] = top_x + x1[a] * res
        UTM_x2[a] = top_x + x2[a] * res
        UTM_y1[a] = top_y - y1[a] * res
        UTM_y2[a] = top_y - y2[a] * res

        col1[a] = int((UTM_x1[a] - xOrigin) / pixelWidth)
        col2[a] = int((UTM_x2[a] - xOrigin) / pixelWidth)
        row1[a] = int((yOrigin - UTM_y1[a] ) / pixelHeight)
        row2[a] = int((yOrigin - UTM_y2[a] ) / pixelHeight)
    #a = to_latlon(UTM_x1[1],UTM_y1[1],16,'N')


    grid_hyper = np.zeros((grid.shape[0],grid.shape[1]))
    grid_hyper[:,0] = col1
    grid_hyper[:,1] = row1
    grid_hyper[:,2] = col2
    grid_hyper[:,3] = row2
    grid_hyper[:,4] = plot
    grid_hyper[:,5] = row
    return grid_hyper

def extract_row_plot_from_field_rgb_image(h, grid, plotnum, rownum):
    """ Extract plot from the RGB image of the field using the values
        specified in the grid"""
    ind = grid[:,4] == plotnum
    if not any(ind):
        print("Unable to find plot %d in the grid" % (plotnum))
        return None
    else:
        ind = np.where(ind)[0]
        #print(ind)
        ind2 = np.where(grid[ind,5] == rownum)[0]
        #print(ind2)
        plotbound = grid[ind[ind2], 0:][0]
        #print(plotbound)
        return h[(int(plotbound[1])-1):int(plotbound[3]), (int(plotbound[0])-1):int(plotbound[2])], plotbound
    
def extract_row_from_field_rgb_image(h, grid, rownum):
    """ Extract plot from the RGB image of the field using the values
        specified in the grid"""
    if rownum>len(grid):
        print("Unable to find plot %d in the grid" % (rownum))
        return None
    else:
        plotbound = grid[rownum, 0:4]
        # print(plotbound)
        return h[(int(plotbound[1])-1):int(plotbound[3]), (int(plotbound[0])-1):int(plotbound[2]), :], plotbound

  

datadir = "D:/Processing/Hyperspectral/f54_2022/"
#heightdates = ["20210617","20210622","20210703","20210719","20210727","20210802","20210808","20210816","20210822","20210828","20210906","20210913","20210917","20210924"]
#heightdates = ["20220614", "20220628","20220705","20220710","20220714","20220721","20220729","20220806", "20220810", "20220817", "20220824", "20220830", "20220908"]
heightdates = ["20220730"]
field_number = "f54mn"
platform = 'india'
#platform = 'vnirswir'
field_panel  = "44m"
# North
top_x = 500829.179
top_y = 4480958.863
res = 0.01
#South
#top_x = 500829.152
#top_y = 4480740.918
#res = 0.01
rows_to_extract = [3,4]
trim_ax = 'y'
for heightdate in heightdates:
    outdir = datadir+heightdate+"_"+field_number+"_"+platform+field_panel+"/"
    if os.path.isdir(outdir)==False:
        os.makedirs(outdir)
    SWIR = False
    wavelet_ex = False
    bin_number = 2
    fieldim = os.path.join(datadir+heightdate+"_"+field_number+"_"+platform+"_"+field_panel+"/seam_mosaic")
    print(fieldim)
    ds_nano = gdal.Open(fieldim,gdal.GA_ReadOnly)
    print("size of image: %dx%d, number of bands: %d" % (ds_nano.RasterXSize,ds_nano.RasterYSize,ds_nano.RasterCount))
    geotransform = ds_nano.GetGeoTransform()
    print("resolution %.2fm x %.2f m" % (geotransform[1],np.abs(geotransform[5])))
    if SWIR:
        fieldim_swir = os.path.join(datadir+heightdate+"/"+field_number+"/"+"Mosaic_F41Cali_SWIR_"+heightdate+"_ref")

    grid = np.loadtxt("D:/Processing/Hyperspectral/f54_2022/f54mn_rows.txt", skiprows=1)
    grid_org = copy.deepcopy(grid)
    #grid[:,0] = grid_org[:,1]
    #grid[:,1] = grid_org[:,0]
    #grid[:,2] = grid_org[:,3]
    #grid[:,3] = grid_org[:,2]
    grid_org = copy.deepcopy(grid)
    #exec(open("cigars_ectraction_script_2022.py").read())
    def band_finder(hyper_data, wavelengthes, wave):
        #print(np.argmin(abs(wavelengthes-wave)))
        return np.atleast_3d(hyper_data[:, :, np.argmin(abs(wavelengthes-wave))])
    
    def index_correction(index, desire_min, desire_max):
        index[index<desire_min]=desire_min
        index[index>desire_max]=desire_max
        index[np.isnan(index)]=desire_max
        return index
    
    def index_check_max(index, desire_max):
        index[index<=desire_max]=0
        index[index>desire_max]=1
        return index
    
    def index_check_min(index, desire_min):
        index[index<desire_min]=1
        index[index>=desire_min]=0
        return index
    
    def plot_index(index, outdir, index_name):
        f = plt.figure(figsize = (45,int(25/np.divide(np.float(index.shape[1]),np.float(index.shape[0])))))
        ax1 = f.add_subplot(2,3,1)
        a = ax1.imshow(index[:,:,0])
        f.colorbar(a, ax = ax1)
        plt.savefig(os.path.join(outdir+index_name+'.png'), bbox_inches='tight', pad_inches=0.2)
        plt.close()
    
    grid_org = copy.deepcopy(grid)
    plots = np.unique(grid[:,4])
    rows = np.unique(grid[:,5])
    print(plots)
    print(rows)
    if os.path.isdir(outdir)==False:
        os.makedirs(outdir)
        
    
    grid_hyper = grid_ortho2hyperUTM(grid, ds_nano,top_x,top_y,res)
    border_xy = 150
    min_x_hyper = int(np.min(grid_hyper[:,[0, 2]])-border_xy)
    max_x_hyper = int(np.max(grid_hyper[:,[0, 2]])+border_xy)
    min_y_hyper = int(np.min(grid_hyper[:,[1, 3]])-border_xy)
    max_y_hyper = int(np.max(grid_hyper[:,[1, 3]])+border_xy)
    grid_hyper_subset = grid_hyper - [min_x_hyper, min_y_hyper, min_x_hyper, min_y_hyper, 0, 0]
    
    plots = np.unique(grid_hyper_subset[:,4])
    rows = np.unique(grid_hyper_subset[:,5])
    grid = copy.deepcopy(grid_hyper_subset)
    grid_org = copy.deepcopy(grid_hyper_subset)
    
    def read_hyperspec_image_wavelengths(ds):
        """ Convert the hyperspectral image read using GDAL to 3D array """
        freq = np.zeros((ds.RasterCount,))
        for b in range(ds.RasterCount):
            band = ds.GetRasterBand(b+1)
            band_desc = band.GetDescription()
            if band_desc.find(" nm")!=-1:
                bands_list4[b] = float(band_desc[:band_desc.find(" nm")])
            elif band_desc.find(" Nanometers")!=-1:
                if band_desc.find(") (")==-1:
                    freq[b] = float(band_desc[:band_desc.find(" Nanometers")])
                else:
                    freq[b] = float(band_desc[band_desc.find(") (")+3:band_desc.find(" Nanometers")])
            else:
                freq[b] = float(band_desc)
        return freq
    
    Nano_wavelength = read_hyperspec_image_wavelengths(ds_nano)
    
    x = open_image(fieldim+'.hdr')
    if x.scale_factor ==1:
        scale_ref = 1.0 ###################### Careful
    elif x.scale_factor ==10000:
        scale_ref = 1.0
    
    ''''
    ds_nano2 = np.array(ds_nano)
    x = convert_hyperspec_image_to_3D_array(ds_nano)
    scale_ref = 10000
    '''
    
    x = x[min_y_hyper:max_y_hyper,min_x_hyper:max_x_hyper,:] / scale_ref
    x[:, :, 0] = x[:, :, 1]
    x[:, :, -1] = x[:, :, -2]
    print(len(x<0))
    x[x<0] = 0 
    print(x.shape, np.max(grid, axis = 0))
    wavelength = Nano_wavelength
    print(wavelength.shape)  
    
    OSAVI = 1.16 * (band_finder(x, wavelength, 800) - band_finder(x, wavelength, 670)) / ( 0.16 + band_finder(x, wavelength, 800) + band_finder(x, wavelength, 670))
    OSAVI = index_correction(OSAVI, 0, 1)
    plot_index(OSAVI, outdir, 'OSAVI')
    
    ##### Shado-Soil Removal:
    
    Shiny_thereshold = 0.25
    Shadw_thereshold = 0.02
    OSAVI_thereshold = 0.30
    
    ortho_mask = np.zeros((x.shape[0],x.shape[1]))
    hyper_im_rgb = show_rgb_from_hyperspec(x, wavelength)
    visible_ave = np.mean(hyper_im_rgb, axis=2)
    non_veg_mask = (OSAVI[:,:,0]<OSAVI_thereshold)
    shiny_mask = (visible_ave>Shiny_thereshold)
    shadw_mask = (hyper_im_rgb[:,:,1]<Shadw_thereshold)
    Mask = non_veg_mask + shadw_mask + ortho_mask + shiny_mask> 0
    
    envi.save_image(outdir+'OSAVI.hdr', OSAVI, dtype=np.float32, interleave='bsq', force=True, ext='')
    #envi.save_image(outdir+'non_veg_mask.hdr', non_veg_mask, dtype=np.float32, interleave='bsq', force=True, ext='')
    #envi.save_image(outdir+'shiny_mask.hdr', shiny_mask, dtype=np.float32, interleave='bsq', force=True, ext='')
    envi.save_image(outdir+'shadw_mask.hdr', shadw_mask, dtype=np.float32, interleave='bsq', force=True, ext='')
    envi.save_image(outdir+'Mask.hdr', Mask, dtype=np.float32, interleave='bsq', force=True, ext='')
    envi.save_image(outdir+'hyper_im_rgb.hdr', hyper_im_rgb, dtype=np.float32, interleave='bsq', force=True, ext='')
    
    f = plt.figure(figsize = (100,int(30/np.divide(np.float(x.shape[1]),np.float(x.shape[0])))))
    ax1 = f.add_subplot(1,4,1)
    a = ax1.imshow(2*hyper_im_rgb)
    plt.title('Hyperspectral Data', fontsize=22)
    plt.axis('off')
    
    ax2 = f.add_subplot(1,4,2)
    b = ax2.imshow(OSAVI[:,:,0])
    #f.colorbar(b, ax = ax2)
    plt.title('OSAVI', fontsize=22)
    plt.axis('off')
    
    ax4 = f.add_subplot(1,4,3)
    c = ax4.imshow(shadw_mask)
    #f.colorbar(c, ax = ax4)
    plt.title('Mask Image (shadow pixels)', fontsize=22)
    plt.axis('off')
    
    ax5 = f.add_subplot(1,4,4)
    c = ax5.imshow(Mask)
    plt.title('Mask Image', fontsize=22)
    plt.axis('off')
    
    np.save(os.path.join(outdir + heightdate+"_Mask"), Mask)
    plt.savefig(os.path.join(outdir+heightdate+'_NonVegMask.png'), bbox_inches='tight', pad_inches=0.2)
    plt.close()
    ##################################################
    
    Indices_features = True # True or False
    NW = Nano_wavelength
    
    if Indices_features:
    
        NDVI705 = (band_finder(x, NW, 750) - band_finder(x, NW, 705)) / (band_finder(x, NW, 750) + band_finder(x, NW, 705))
        NDVI705 = index_correction(NDVI705, 0, 1)
        plot_index(NDVI705, outdir, 'NDVI705')
            
        mNDVI705 = (band_finder(x, NW, 750) - band_finder(x, NW, 705)) / (band_finder(x, NW, 750) + band_finder(x, NW, 705) - 2*band_finder(x, NW, 445))
        mNDVI705 = index_correction(mNDVI705, 0, 1)
        plot_index(mNDVI705, outdir, 'mNDVI705')
        
        mSR705 = (band_finder(x, NW, 750) - band_finder(x, NW, 445)) / (band_finder(x, NW, 705) - band_finder(x, NW, 445))
        mSR705 = index_correction(mSR705, 0, 15)
        plot_index(mSR705, outdir, 'mSR705')
        
        GNDVI = (band_finder(x, NW, 750) - band_finder(x, NW, 550)) / (band_finder(x, NW, 750) + band_finder(x, NW, 550))
        GNDVI = index_correction(GNDVI, 0, 1)
        plot_index(GNDVI, outdir, 'GNDVI')
    
        RNDVI = (band_finder(x, NW, 800) - band_finder(x, NW, 670)) / np.sqrt(band_finder(x, NW, 800) + band_finder(x, NW, 670))
        RNDVI = index_correction(RNDVI, 0, 1)
        plot_index(RNDVI, outdir, 'RNDVI')
    
        NDCI = (band_finder(x, NW, 762) - band_finder(x, NW, 527)) / (band_finder(x, NW, 762) + band_finder(x, NW, 527))
        NDCI = index_correction(NDCI, 0, 1)
        plot_index(NDCI, outdir, 'NDCI')
    
        Datt1 = (band_finder(x, NW, 850) - band_finder(x, NW, 710)) / (band_finder(x, NW, 850) - band_finder(x, NW, 680))
        Datt1 = index_correction(Datt1, 0, 1)
        plot_index(Datt1, outdir, 'Datt1')
    
        Datt2 = band_finder(x, NW, 850) / band_finder(x, NW, 710) 
        Datt2 = index_correction(Datt2, 0, 15)
        plot_index(Datt2, outdir, 'Datt2')
    
        Datt3 = band_finder(x, NW, 754) / band_finder(x, NW, 704) 
        Datt3 = index_correction(Datt3, 0, 15)
        plot_index(Datt3, outdir, 'Datt3')
    
        Carte1 = band_finder(x, NW, 695) / band_finder(x, NW, 420) 
        Carte1 = index_correction(Carte1, 0, 5)
        plot_index(Carte1, outdir, 'Carte1')
    
        Carte2 = band_finder(x, NW, 695) / band_finder(x, NW, 760) 
        Carte2 = index_correction(Carte2, 0, 1)
        plot_index(Carte2, outdir, 'Carte2')
    
        Carte3 = band_finder(x, NW, 605) / band_finder(x, NW, 760) 
        Carte3 = index_correction(Carte3, 0, 1)
        plot_index(Carte3, outdir, 'Carte3')
    
        Carte4 = band_finder(x, NW, 710) / band_finder(x, NW, 760) 
        Carte4 = index_correction(Carte4, 0, 1)
        plot_index(Carte4, outdir, 'Carte4')
    
        Carte5 = band_finder(x, NW, 695) / band_finder(x, NW, 670) 
        Carte5 = index_correction(Carte5, 0, 5)
        plot_index(Carte5, outdir, 'Carte5')
        
        SR800680 = band_finder(x, NW, 800) / band_finder(x, NW, 680) 
        SR800680 = index_correction(SR800680, 0, 100)
        plot_index(SR800680, outdir, 'SR800680')
        
        SR675700 = band_finder(x, NW, 675) / band_finder(x, NW, 700) 
        SR675700 = index_correction(SR675700, 0, 1)
        plot_index(SR675700, outdir, 'SR675700')
        
        SR700670 = band_finder(x, NW, 700) / band_finder(x, NW, 670) 
        SR700670 = index_correction(SR700670, 0, 5)
        plot_index(SR700670, outdir, 'SR700670')
        
        SR750700 = band_finder(x, NW, 750) / band_finder(x, NW, 700) 
        SR750700 = index_correction(SR750700, 0, 25)
        plot_index(SR750700, outdir, 'SR750700')
    
        SR752690 = band_finder(x, NW, 752) / band_finder(x, NW, 690) 
        SR752690 = index_correction(SR752690, 0, 50)
        plot_index(SR752690, outdir, 'SR752690')
        
        SR750550 = band_finder(x, NW, 750) / band_finder(x, NW, 550) 
        SR750550 = index_correction(SR750550, 0, 50)
        plot_index(SR750550, outdir, 'SR750550')
    
        SR750710 = band_finder(x, NW, 750) / band_finder(x, NW, 710) 
        SR750710 = index_correction(SR750710, 0, 10)
        plot_index(SR750710, outdir, 'SR750710')
        
    #     New Vegetation Index (NVI)
        NVI = (band_finder(x, NW, 777) - band_finder(x, NW, 747)) / band_finder(x, NW, 673) 
        NVI = index_correction(NVI, 0, 40)
        plot_index(NVI, outdir, 'NVI')
    
        EVI = 2.5*(band_finder(x, NW, 800) - band_finder(x, NW, 670)) / (band_finder(x, NW, 800) + 6*band_finder(x, NW, 670) + 7.5*band_finder(x, NW, 475) + 1) 
        EVI = index_correction(EVI, 0, 2)
        plot_index(EVI, outdir, 'EVI')
        
        OSAVI = 1.16 * (band_finder(x, NW, 800) - band_finder(x, NW, 670)) / ( 0.16 + band_finder(x, NW, 800) + band_finder(x, NW, 670))
        OSAVI = index_correction(OSAVI, 0, 1)
        plot_index(OSAVI, outdir, 'OSAVI')
        
        OSAVI2 = 1.16 * (band_finder(x, NW, 750) - band_finder(x, NW, 705)) / ( 0.16 + band_finder(x, NW, 750) + band_finder(x, NW, 705))
        OSAVI2 = index_correction(OSAVI2, 0, 1)
        plot_index(OSAVI2, outdir, 'OSAVI2')
        
        TCARI = 0.5 + 3*( band_finder(x, NW, 700) - band_finder(x, NW, 670) - 0.2 * (band_finder(x, NW, 700) - band_finder(x, NW, 550)) * (band_finder(x, NW, 700) / band_finder(x, NW, 670))) 
        TCARI = index_correction(TCARI, 0, 1)
        plot_index(TCARI, outdir, 'TCARI')
        
        TCARI2 = 3*( band_finder(x, NW, 750) - band_finder(x, NW, 705) - 0.2 * (band_finder(x, NW, 750) - band_finder(x, NW, 550)) * (band_finder(x, NW, 750) / band_finder(x, NW, 705))) 
        TCARI2 = index_correction(TCARI2, -1, 0.5)
        plot_index(TCARI2, outdir, 'TCARI2')
        
        MCARI = (band_finder(x, NW, 700) - band_finder(x, NW, 670) - 0.2 * (band_finder(x, NW, 700) - band_finder(x, NW, 550))) * (band_finder(x, NW, 700) / band_finder(x, NW, 670)) 
        MCARI = index_correction(MCARI, 0, 1)
        plot_index(MCARI, outdir, 'MCARI')
        
        TVI =  0.5 * (120*(band_finder(x, NW, 750)-band_finder(x, NW, 550)) - 2.5 * (band_finder(x, NW, 670)-band_finder(x, NW, 550)) )
        TVI = index_correction(TVI, -100, 100)
        plot_index(TVI, outdir, 'TVI')
        
    #     SPVI = 0.4* 3.7*(band_finder(x, NW, 800)-band_finder(x, NW, 670)) - 1.2*np.abs(band_finder(x, NW, 530)-band_finder(x, NW, 670))
    #     SPVI = index_correction(SPVI, -100, 100)
    #     plot_index(SPVI, outdir, 'SPVI')
        SPVI = 0.4*3.7*(band_finder(x, NW, 800)-band_finder(x, NW, 670))-1.2*np.sqrt((band_finder(x, NW, 530)-band_finder(x, NW, 670))**2)
        SPVI = index_correction(SPVI, -100, 100)
        plot_index(SPVI, outdir, 'SPVI') 
        
        
        REP = 700 + 40*( (band_finder(x, NW, 670)+band_finder(x, NW, 780))/2 - band_finder(x, NW, 700) ) / (band_finder(x, NW, 740)-band_finder(x, NW, 700))
        REP = index_correction(REP, 700, 750)
        plot_index(REP, outdir, 'REP')
        
        PRI = (band_finder(x, NW, 531) - band_finder(x, NW, 570)) / (band_finder(x, NW, 531) + band_finder(x, NW, 570))
        PRI = index_correction(PRI, -1, 1)
        plot_index(PRI, outdir, 'PRI')
        
        RI1db = band_finder(x, NW, 735) / band_finder(x, NW, 720) 
        RI1db = index_correction(RI1db, 0.5, 2.5)
        plot_index(RI1db, outdir, 'RI1db')
        
        VOG1 = band_finder(x, NW, 740) / band_finder(x, NW, 720) 
        VOG1 = index_correction(VOG1, 0.5, 3)
        plot_index(VOG1, outdir, 'VOG1')
        
        VOG2 = (band_finder(x, NW, 734) - band_finder(x, NW, 747)) / (band_finder(x, NW, 715) + band_finder(x, NW, 726))
        VOG2 = index_correction(VOG2, -1.2, 0.3)
        plot_index(VOG2, outdir, 'VOG2')
        
        VOG3 = (band_finder(x, NW, 734) - band_finder(x, NW, 747)) / (band_finder(x, NW, 715) + band_finder(x, NW, 720))
        VOG3 = index_correction(VOG3, -1.2, 0.3)
        plot_index(VOG3, outdir, 'VOG3')
        
        RDVI = (band_finder(x, NW, 800)-band_finder(x, NW, 670))/ (np.sqrt(band_finder(x, NW, 800)+band_finder(x, NW, 670)))
        RDVI = index_correction(RDVI, 0, 1)
        plot_index(RDVI, outdir, 'RDVI')
        
        MSAVI = 0.5*(2*band_finder(x, NW, 800)+1-np.sqrt(((2*band_finder(x, NW, 800)+1)**2)-8*(band_finder(x, NW, 800)-band_finder(x, NW, 670))))
        MSAVI = index_correction(MSAVI, -1.0, 1.0)
        plot_index(RDVI, outdir, 'MSAVI')
        
        MCARI2 = (band_finder(x, NW, 750) - band_finder(x, NW, 705) - 0.2 * (band_finder(x, NW, 750) - band_finder(x, NW, 550))) * (band_finder(x, NW, 750) / band_finder(x, NW, 705)) 
        MCARI2 = index_correction(MCARI2, 0, 1)
        plot_index(MCARI2, outdir, 'MCARI2') 
        
        MCARI2_OSAVI2 = MCARI2/OSAVI2 
        MCARI2_OSAVI2 = index_correction(MCARI2_OSAVI2, 0, 1)
        plot_index(MCARI2_OSAVI2, outdir, 'MCARI2_OSAVI2') 
        
        PSRI =  (band_finder(x, NW, 678)-band_finder(x, NW, 500))/band_finder(x, NW, 750)
        PSRI = index_correction(PSRI, -1, 1)
        plot_index(PSRI, outdir, 'PSRI')
        
        HBSI1 =  (band_finder(x, NW, 855)-band_finder(x, NW, 682))/(band_finder(x, NW, 855)+band_finder(x, NW, 682))
        HBSI1 = index_correction(HBSI1, -1, 1)
        plot_index(HBSI1, outdir, 'HBSI1')
        
        HBSI2 =  (band_finder(x, NW, 910)-band_finder(x, NW, 682))/(band_finder(x, NW, 910)+band_finder(x, NW, 682))
        HBSI2 = index_correction(HBSI2, -1, 1)
        plot_index(HBSI2, outdir, 'HBSI2')
        
        HBSI3=  (band_finder(x, NW, 550)-band_finder(x, NW, 682))/(band_finder(x, NW, 550)+band_finder(x, NW, 682))
        HBSI3 = index_correction(HBSI3, -1, 1)
        plot_index(HBSI3, outdir, 'HBSI3')
        
        DCNI = (band_finder(x, NW, 720)-band_finder(x, NW, 700))/(band_finder(x, NW, 700)-band_finder(x, NW, 670))/(band_finder(x, NW, 720)-band_finder(x, NW, 670)+0.03)
        DCNI = index_correction(DCNI, 0, 100)
        plot_index(DCNI, outdir, 'DCNI')
        
        HBCI8 = (band_finder(x, NW, 550)-band_finder(x, NW, 515))/(band_finder(x, NW, 550)+band_finder(x, NW, 515))
        HBCI8 = index_correction(HBCI8, -1, 1)
        plot_index(HBCI8, outdir, 'HBCI8')
        
        HBCI9 = (band_finder(x, NW, 550)-band_finder(x, NW, 490))/(band_finder(x, NW, 550)+band_finder(x, NW, 490))
        HBCI9 = index_correction(HBCI9, -1, 1)
        plot_index(HBCI9, outdir, 'HBCI9')
        
        HREI15 = (band_finder(x, NW, 855)-band_finder(x, NW, 720))/(band_finder(x, NW, 855)+band_finder(x, NW, 720))
        HREI15 = index_correction(HREI15, -1, 1)
        plot_index(HREI15, outdir, 'HREI15')
        
        HREI16 = (band_finder(x, NW, 910)-band_finder(x, NW, 705))/(band_finder(x, NW, 910)+band_finder(x, NW, 705))
        HREI16 = index_correction(HREI16, -1, 1)
        plot_index(HREI16, outdir, 'HREI16')
        
        NDRE = (band_finder(x, NW, 790)-band_finder(x, NW, 720))/(band_finder(x, NW, 790)+band_finder(x, NW, 720))
        NDRE = index_correction(NDRE, -1, 1)
        plot_index(NDRE, outdir, 'NDRE')
        
        Indices = np.concatenate((NDVI705, mNDVI705, mSR705, GNDVI, RNDVI, NDCI, Datt1, Datt2, Datt3, Carte1, Carte2, Carte3, Carte4, Carte5, SR800680, SR675700, SR700670, SR750700, SR752690, SR750550, SR750710, NVI, EVI, OSAVI, OSAVI2, TCARI, TCARI2, MCARI, TVI, SPVI, REP, PRI, RI1db, VOG1, VOG2, VOG3, RDVI, MSAVI, MCARI2, MCARI2_OSAVI2, PSRI, HBSI1, HBSI2, HBSI3, DCNI, HBCI8, HBCI9, HREI15, HREI16, NDRE), axis = 2)
        print(Indices.shape)
        
        Indices_names = 'NDVI705, mNDVI705, mSR705, GNDVI, RNDVI, NDCI, Datt1, Datt2, Datt3, Carte1, Carte2, Carte3, Carte4, Carte5, SR800680, SR675700, SR700670, SR750700, SR752690, SR750550, SR750710, NVI, EVI, OSAVI, OSAVI2, TCARI, TCARI2, MCARI, TVI, SPVI, REP, PRI, RI1db, VOG1, VOG2, VOG3, RDVI, MSAVI, MCARI2, MCARI2_OSAVI2, PSRI, HBSI1, HBSI2, HBSI3, DCNI, HBCI8, HBCI9, HREI15, HREI16, NDRE'
    
    del NDVI705, mNDVI705, mSR705, GNDVI, RNDVI, NDCI, Datt1, Datt2, Datt3, Carte1, Carte2, Carte3, Carte4, Carte5, SR800680, SR675700, SR700670, SR750700, SR752690, SR750550, SR750710, NVI, EVI, OSAVI, OSAVI2, TCARI, TCARI2, MCARI, TVI, SPVI, REP, PRI, RI1db, VOG1, VOG2, VOG3, RDVI, MSAVI, MCARI2, MCARI2_OSAVI2, PSRI, HBSI1, HBSI2, HBSI3, DCNI, HBCI8, HBCI9, HREI15, HREI16, NDRE
    gc.collect()
    
    def latent_feature_extraction(hyper2d, method, **karg):
        
        if 'n_components' in karg:
            n_components = karg['n_components']
        else:
            n_components = 30
            
        if 'n_neighbors' in karg:
            n_neighbors = karg['n_neighbors']
        else:
            n_neighbors = 10
    
        if method == 'PCA':
            pca = PCA(n_components=n_components)
            pca.fit(hyper2d)
            latent_features = pca.transform(hyper2d)
            latent_features_nzd = latent_features-np.min(latent_features, axis=0)
            latent_features_mean = np.mean(latent_features_nzd, axis=0)
    
        if method == 'Isomap':
            iso = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components)
            iso.fit(np.atleast_2d(hyper2d))
            latent_features = iso.transform(hyper2d)
            latent_features_nzd = latent_features-np.min(latent_features, axis=0)
            latent_features_mean = np.mean(latent_features_nzd, axis=0)
            
        if method == 'LLE':
            lle = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components)
            lle.fit(np.atleast_2d(hyper2d))
            latent_features = lle.transform(hyper2d)
            latent_features_nzd = latent_features-np.min(latent_features, axis=0)
            latent_features_mean = np.mean(latent_features_nzd, axis=0)
    
        return latent_features_mean, latent_features 
    
    def Integration_feature_extraction(x, wavelength, **karg):
        band400 = np.argmin(abs(wavelength-400))
        band500 = np.argmin(abs(wavelength-500))
        band600 = np.argmin(abs(wavelength-600))
        band685 = np.argmin(abs(wavelength-685))
        band745 = np.argmin(abs(wavelength-745))
        band770 = np.argmin(abs(wavelength-770))
        band910 = np.argmin(abs(wavelength-910))
        band1000 = np.argmin(abs(wavelength-1000))
    
        Integ_feature = np.zeros((x.shape[0],x.shape[1],6))
        
        Integ_feature[:, :, 0] = np.trapz(x[:, :, band400:band500], x=np.squeeze(wavelength[band400:band500]))
        Integ_feature[:, :, 1] = np.trapz(x[:, :, band500:band600], x=np.squeeze(wavelength[band500:band600]))
        Integ_feature[:, :, 2] = np.trapz(x[:, :, band600:band685], x=np.squeeze(wavelength[band600:band685]))
        Integ_feature[:, :, 3] = np.trapz(x[:, :, band685:band745], x=np.squeeze(wavelength[band685:band745]))
        Integ_feature[:, :, 4] = np.trapz(x[:, :, band770:band910], x=np.squeeze(wavelength[band770:band910]))
        Integ_feature[:, :, 5] = np.trapz(x[:, :, band910:band1000+1], x=np.squeeze(wavelength[band910:band1000+1]))
    
        return Integ_feature
    
    x_Integration = Integration_feature_extraction(x, wavelength)
        
    
    def plot_save_image(image_to_show,outdir,heightdate,label):
        f = plt.figure(figsize = (30,15))
        ax1 = f.add_subplot(1,5,1)
        a = ax1.imshow(image_to_show)
        tittle = label + ' - '+heightdate
        plt.title(tittle, fontsize=15)
        plt.axis('off')
        f.savefig(os.path.join(outdir+heightdate+'_'+label+'.png'), bbox_inches='tight', pad_inches=0.2)
        plt.close()
        
    def FSDR_feature_extraction(x_FDR, x_SDR, wavelength, **karg):
        min_green = np.argmin(x_FDR[:,:,np.argmin(abs(wavelength-500)):np.argmin(abs(wavelength-600))], 2) + np.argmin(abs(wavelength-500))
        max_red = np.argmax(x_FDR[:,:,np.argmin(abs(wavelength-500)):np.argmin(abs(wavelength-600))], 2)+ np.argmin(abs(wavelength-500))
        I,J = np.ogrid[:x_FDR.shape[0],:x_FDR.shape[1]]
        fdr_slope = (x_FDR[I,J,min_green]-x_FDR[I,J,max_red]) / (min_green-max_red)
        min_nir = np.argmin(x_FDR[:,:,np.argmin(abs(wavelength-680)):np.argmin(abs(wavelength-760))], 2)+np.argmin(abs(wavelength-680))
        min_fdr = x_FDR[I,J,min_nir]
        band670 = np.argmin(abs(wavelength-670))
        band780 = np.argmin(abs(wavelength-780))
        integration_NIR = np.trapz(x_FDR[:,:,band670:band780], x=np.squeeze(wavelength[band670:band780]))
        band910 = np.argmin(abs(wavelength-910))
        band1000 = np.argmin(abs(wavelength-1000))
        integration_FNIR = np.trapz(x_FDR[:,:,band910:band1000], x=np.squeeze(wavelength[band910:band1000]))
        min_nir_sdr = np.argmin(x_FDR[:,:,np.argmin(abs(wavelength-650)):np.argmin(abs(wavelength-800))], 2)+ np.argmin(abs(wavelength-650))
        max_nir_sdr = np.argmax(x_FDR[:,:,np.argmin(abs(wavelength-650)):np.argmin(abs(wavelength-800))], 2)+ np.argmin(abs(wavelength-650))
        sdr_slope = (x_SDR[I,J,min_nir_sdr]-x_SDR[I,J,max_nir_sdr]) / (min_nir_sdr-max_nir_sdr)
        integration_sdr = np.trapz(x_SDR[:,:,:], x=np.squeeze(wavelength[:-2]))
    
        FSDR_features = np.zeros((x.shape[0],x.shape[1],6))
        
        plot_save_image(fdr_slope,outdir,heightdate,'FDR_Slope')
        FSDR_features[:, :, 0] = fdr_slope
    
        plot_save_image(min_fdr,outdir,heightdate,'FDR_Min')
        FSDR_features[:, :, 1] = min_fdr
        
        plot_save_image(integration_NIR,outdir,heightdate,'FDR_Intgr_NIR')
        FSDR_features[:, :, 2] = integration_NIR
        
        plot_save_image(integration_FNIR,outdir,heightdate,'FDR_Intgr_FNIR')
        FSDR_features[:, :, 3] = integration_FNIR
        
        plot_save_image(sdr_slope,outdir,heightdate,'SDR_Slope')
        FSDR_features[:, :, 4] = sdr_slope
        
        plot_save_image(integration_sdr,outdir,heightdate,'SDR_Intgr')
        FSDR_features[:, :, 5] = integration_sdr
        
        print('number of Nan values in FSDR features:', np.where(np.isnan(FSDR_features))[0].shape, 'of: ', FSDR_features[:].shape)
        FSDR_features[np.isnan(FSDR_features)]=0
        return FSDR_features
    smoothed_x = signal.savgol_filter(x, 11, 3, axis=-1)
    x_FDR = np.divide(smoothed_x[:,:,:-1]-smoothed_x[:,:,1:], wavelength[1:]-wavelength[:-1]) 
    smoothed_x = []
    smoothed_x_FDR = signal.savgol_filter(x_FDR, 11, 3, axis=-1)
    x_SDR = np.divide(smoothed_x_FDR[:,:,:-1]-smoothed_x_FDR[:,:,1:], wavelength[1:-1]-wavelength[:-2])
    smoothed_x_SDR = signal.savgol_filter(x_SDR, 11, 3, axis=-1)
    FSDR_features = FSDR_feature_extraction(smoothed_x_FDR, smoothed_x_SDR, wavelength)
    smoothed_x_FDR = []
    smoothed_x_SDR = []
    
    ######### Wavelet
    if wavelet_ex:
        hyp_wavelet_01 = pywt.wavedec2(x[:,:,:], 'bior2.2',  mode='smooth', level=3, axes=(0,1))[0]
        print(hyp_wavelet_01.shape)
        hyp_wavelet_012 = pywt.wavedec(hyp_wavelet_01, 'bior2.2', mode='smooth', level=3, axis=2)[0][2:-2,2:-2,2:-2]
        print(hyp_wavelet_012.shape)
        hyp_wavelet_012_resample = scipy.ndimage.zoom(hyp_wavelet_012, (8, 8, 1), order=0)
        hyp_wavelet_012_resample.shape
        file_to_save = outdir + 'wavelet2.hdr'
        envi.save_image(file_to_save, hyp_wavelet_012_resample, dtype=np.float32, interleave='bil', force=True, ext='')
        for i in range(np.shape(hyp_wavelet_012_resample)[-1]):
            plot_index(hyp_wavelet_012_resample[:,:,i:i+1], outdir, 'Wavelet_'+str(i))
    
    #########################################################################
    #########################################################################
    #########################################################################
    #########################################################################
    #########################################################################
    #########################################################################
    n_p_trim = 0
    grid = copy.deepcopy(grid_org)
    Map_PlotNumber = np.zeros((x.shape[0],x.shape[1]))
    Map_ClassNumber = np.zeros((x.shape[0],x.shape[1]))
    for plotnum in plots:
        ind = grid[:,4] == plotnum
        rows = grid[np.where(ind)[0],5]
        for rownum in rows:
            rownum = int(rownum)
            if trim_ax == 'x':
                grid[np.where(ind)[0][rownum-1], 0:4] = grid[np.where(ind)[0][rownum-1], 0:4] + [n_p_trim, 0, -n_p_trim, 0]
            else:
                grid[np.where(ind)[0][rownum-1], 0:4] = grid[np.where(ind)[0][rownum-1], 0:4] + [0, n_p_trim, 0,-n_p_trim]
            plotbound = grid[np.where(ind)[0][rownum-1], 0:4]
            Map_PlotNumber[(int(plotbound[1])-1):int(plotbound[3]), (int(plotbound[0])-1):int(plotbound[2])] = plotnum    
                
    f = plt.figure(figsize = (15,int(14/np.divide(np.float(x.shape[1]),np.float(x.shape[0])))))
    ax1 = f.add_subplot(2,1,1)
    a = ax1.imshow(Map_PlotNumber, clim=(plots[0]-1, plots[-1]+1))
    f.colorbar(a, ax = ax1)
    np.save(os.path.join(outdir + 'Map_PlotNumber'), Map_PlotNumber)
    #plt.close()
    
    f = plt.figure(figsize = (300,100))
    ax1 = f.add_subplot(1,5,1)
    a = ax1.imshow(2*hyper_im_rgb)
    tittle = 'Hyperspectral (RGB) - '+heightdate+' - '+str(4*n_p_trim)+' CM Trimmed'
    plt.title(tittle, fontsize=100)
    plt.axis('off')
    for i in grid:
        rect = patches.Rectangle((i[0],i[1]),i[2]-i[0],i[3]-i[1],linewidth=8,edgecolor='r',facecolor='none')
        matplotlib.pyplot.text(i[0],i[1], str(i[4])[:4], fontsize=12, bbox=dict(facecolor='w', alpha=0.5))
        ax1.add_patch(rect)
    plt.show()
    f.savefig(os.path.join(outdir+heightdate+'_'+str(4*n_p_trim)+'cm_trimmed.png'), bbox_inches='tight', pad_inches=0.2)
    
    Manifold=False
    from extract_plot_features_pheno_traits import bin_edges_to_centers
    n_hyp_channels = x.shape[2]
    print('n_hyp_channels', n_hyp_channels)
    n_plots = len(plots)
    n_rows = len(rows)
    nbins = 1
    
    n_indices_features = 0
    
    if Indices_features:
        n_indices_features = Indices.shape[2]
        indx_hist_features = np.zeros((n_plots*n_rows, 25))
        print('n_indices_features', n_indices_features)
    
    Plots_n_pixels = np.zeros((n_plots*n_rows, 1))
    Plots_mean = np.zeros((n_plots*n_rows, 2 + n_hyp_channels))
    Plots_indices = np.zeros((n_plots*n_rows, n_indices_features))
    Plots_variability = np.zeros((n_plots*n_rows, n_hyp_channels))
    Plots_FDR = np.zeros((n_plots*n_rows, n_hyp_channels-1))
    Plots_SDR = np.zeros((n_plots*n_rows, n_hyp_channels-2))
    Plots_Intg = np.zeros((n_plots*n_rows, 6))
    Plots_Derivative = np.zeros((n_plots*n_rows, 6))
    Plots_Derivative_ave = np.zeros((n_plots*n_rows, 6))
    Plots_median = np.zeros((n_plots*n_rows, n_hyp_channels))
    Plots_Hist = np.zeros((n_plots*n_rows, nbins**2))
    Plots_wavelet = np.zeros((n_plots*n_rows, 17))
    
    n_comp = 20
    Plots_PCA = np.zeros((n_plots*n_rows, n_comp))
    Plots_Isomap10 = np.zeros((n_plots*n_rows, n_comp))
    Plots_Isomap20 = np.zeros((n_plots*n_rows, n_comp))
    Plots_Isomap30 = np.zeros((n_plots*n_rows, n_comp))
    Plots_Isomap40 = np.zeros((n_plots*n_rows, n_comp))
    Plots_Isomap50 = np.zeros((n_plots*n_rows, n_comp))
    Plots_LLE10 = np.zeros((n_plots*n_rows, n_comp))
    Plots_LLE20 = np.zeros((n_plots*n_rows, n_comp))
    Plots_LLE30 = np.zeros((n_plots*n_rows, n_comp))
    Plots_LLE40 = np.zeros((n_plots*n_rows, n_comp))
    Plots_LLE50 = np.zeros((n_plots*n_rows, n_comp))
    Latent_Features = {}
    
    Plots_mean_var = np.zeros((n_plots*n_rows, n_hyp_channels))
    Plots_indices_var = np.zeros((n_plots*n_rows, n_indices_features))
    Plots_Intg_var = np.zeros((n_plots*n_rows, 6))
    Plots_Derivative_var = np.zeros((n_plots*n_rows, 6))
    
    counter = 0
    for plotnum in plots:
        plotnum = int(plotnum)
        ind = grid[:,4] == plotnum
        rows = grid[np.where(ind)[0], 5]
        for rownum in rows:
            rownum = int(rownum)
            hyp_im_plot, plotbound = extract_row_plot_from_field_rgb_image(x, grid, plotnum, rownum)
            hyp_im_plot_FDR, plotbound = extract_row_plot_from_field_rgb_image(x_FDR, grid, plotnum, rownum)
            hyp_im_plot_SDR, plotbound = extract_row_plot_from_field_rgb_image(x_SDR, grid, plotnum, rownum)
            hyp_im_plot_Intg, plotbound = extract_row_plot_from_field_rgb_image(x_Integration, grid, plotnum, rownum)
            hyp_im_plot_FSDR, plotbound = extract_row_plot_from_field_rgb_image(FSDR_features, grid, plotnum, rownum)
    
            if Indices_features:
                indx_im_plot = extract_row_plot_from_field_rgb_image(Indices, grid, plotnum, rownum)[0]
                counts1, edges1 = np.histogram(indx_im_plot, bins=25)
                bin_centers = bin_edges_to_centers(edges1)
                indx_hist_features[counter, :] = counts1
            
            if plotnum==48010:
                fig = plt.figure(figsize = (5,10))
                opacity = 0.4
                error_config = {'ecolor': '0.3'}
                rects1 = plt.bar(bin_centers, counts1)      
                plt.show()
    
            mask_im_plot = extract_row_plot_from_field_rgb_image(np.atleast_3d(Mask), grid, plotnum, rownum)[0]
            mask_im_plot2d = mask_im_plot.reshape((-1, 1))
    
            hyp_im_plot2d = hyp_im_plot.reshape((-1, n_hyp_channels))
            indx_im_plot2d = indx_im_plot.reshape((-1, n_indices_features))
            hyp_im_plot_FDR2d = hyp_im_plot_FDR.reshape((-1, hyp_im_plot_FDR.shape[2]))
            hyp_im_plot_SDR2d = hyp_im_plot_SDR.reshape((-1, hyp_im_plot_SDR.shape[2]))
            hyp_im_plot_Intg2d = hyp_im_plot_Intg.reshape((-1, hyp_im_plot_Intg.shape[2]))
            hyp_im_plot_FSDR2d = hyp_im_plot_FSDR.reshape((-1, hyp_im_plot_FSDR.shape[2]))
    
            hyp_im_plot2d_masked = hyp_im_plot2d[np.squeeze(~mask_im_plot2d), :]
            indx_im_plot2d_masked = indx_im_plot2d[np.squeeze(~mask_im_plot2d), :]
            hyp_im_plot_FDR2d_masked = hyp_im_plot_FDR2d[np.squeeze(~mask_im_plot2d), :]
            hyp_im_plot_SDR2d_masked = hyp_im_plot_SDR2d[np.squeeze(~mask_im_plot2d), :]
            hyp_im_plot_Intg2d_masked = hyp_im_plot_Intg2d[np.squeeze(~mask_im_plot2d), :]
            hyp_im_plot_FSDR2d_masked = hyp_im_plot_FSDR2d[np.squeeze(~mask_im_plot2d), :]
            
            if wavelet_ex:
                hyp_im_plot_wave, plotbound = extract_row_plot_from_field_rgb_image(hyp_wavelet_012_resample, grid, plotnum, rownum)
                hyp_im_plot_wave2d = hyp_im_plot_wave.reshape((-1, hyp_im_plot_wave.shape[2]))
                Plots_wavelet[counter]= np.mean(hyp_im_plot_wave2d, axis = 0)
    
            mean_hyp = np.mean(hyp_im_plot2d_masked, axis = 0)
            mean_hyp_fdr = np.mean(hyp_im_plot_FDR2d_masked, axis = 0)
            mean_hyp_sdr = np.mean(hyp_im_plot_SDR2d_masked, axis = 0)
            
            Plots_n_pixels[counter] = float(len(np.where(mask_im_plot2d==0)[0]))
            Plots_mean[counter][0] = plotnum
            Plots_mean[counter][1] = rownum
            Plots_mean[counter][2:] = mean_hyp
            Plots_median[counter] = np.median(hyp_im_plot2d_masked, axis = 0)
            Plots_indices[counter] = np.mean(indx_im_plot2d_masked, axis = 0)
            Plots_FDR[counter] = mean_hyp_fdr
            Plots_SDR[counter] = mean_hyp_sdr
            Plots_Intg[counter] = np.mean(hyp_im_plot_Intg2d_masked, axis = 0)
            Plots_Derivative[counter] = np.mean(hyp_im_plot_FSDR2d_masked, axis = 0)
    
            min_green = np.argmin(mean_hyp_fdr[np.argmin(abs(wavelength-500)):np.argmin(abs(wavelength-600))])+ np.argmin(abs(wavelength-500))
            max_red = np.argmax(mean_hyp_fdr[np.argmin(abs(wavelength-500)):np.argmin(abs(wavelength-600))])+ np.argmin(abs(wavelength-500))
            fdr_slope = (mean_hyp_fdr[min_green]-mean_hyp_fdr[max_red]) / (min_green-max_red)
            min_nir = np.argmin(mean_hyp_fdr[np.argmin(abs(wavelength-680)):np.argmin(abs(wavelength-760))])+np.argmin(abs(wavelength-680))
            min_fdr = mean_hyp_fdr[min_nir]
            band670 = np.argmin(abs(wavelength-670))
            band780 = np.argmin(abs(wavelength-780))
            integration_NIR = np.trapz(mean_hyp_fdr[band670:band780], x=np.squeeze(wavelength[band670:band780]))
            band910 = np.argmin(abs(wavelength-910))
            band1000 = np.argmin(abs(wavelength-1000))
            integration_FNIR = np.trapz(mean_hyp_fdr[band910:band1000], x=np.squeeze(wavelength[band910:band1000]))
            min_nir_sdr = np.argmin(mean_hyp_sdr[np.argmin(abs(wavelength-650)):np.argmin(abs(wavelength-800))])+ np.argmin(abs(wavelength-650))
            max_nir_sdr = np.argmax(mean_hyp_sdr[np.argmin(abs(wavelength-650)):np.argmin(abs(wavelength-800))])+ np.argmin(abs(wavelength-650))
            sdr_slope = (mean_hyp_sdr[min_nir_sdr]-mean_hyp_sdr[max_nir_sdr]) / (min_nir_sdr-max_nir_sdr)
            integration_sdr = np.trapz(mean_hyp_sdr[:], x=np.squeeze(wavelength[:-2]))
            
            Plots_Derivative_ave[counter][0] = fdr_slope
            Plots_Derivative_ave[counter][1] = min_fdr
            Plots_Derivative_ave[counter][2] = integration_NIR
            Plots_Derivative_ave[counter][3] = integration_FNIR
            Plots_Derivative_ave[counter][4] = sdr_slope
            Plots_Derivative_ave[counter][5] = integration_sdr
            
    
            
            Plots_mean_var[counter] = np.var(hyp_im_plot2d_masked, axis = 0)
            Plots_indices_var[counter] = np.var(indx_im_plot2d_masked, axis = 0)
            Plots_Intg_var[counter] = np.var(hyp_im_plot_Intg2d_masked, axis = 0)
            Plots_Derivative_var[counter] = np.var(hyp_im_plot_FSDR2d_masked, axis = 0)
    
            if Manifold==True:
                if rownum>0 and rownum<3:
                    Plots_PCA[counter, :], Latent_Features['PCA_plot_'+str(plotnum)+'_row_'+str(1+rownum)]  = latent_feature_extraction(hyp_im_plot2d_masked[:100, :], 'PCA', n_components=n_comp)
                    Plots_Isomap10[counter, :], Latent_Features['Isomap10_plot_'+str(plotnum)+'_row_'+str(1+rownum)] = latent_feature_extraction(hyp_im_plot2d_masked[:100, :], 'Isomap', n_neighbors=10, n_components=n_comp)
                    Plots_Isomap20[counter, :], Latent_Features['Isomap20_plot_'+str(plotnum)+'_row_'+str(1+rownum)] = latent_feature_extraction(hyp_im_plot2d_masked[:100, :], 'Isomap', n_neighbors=20, n_components=n_comp)
                    Plots_Isomap30[counter, :], Latent_Features['Isomap30_plot_'+str(plotnum)+'_row_'+str(1+rownum)] = latent_feature_extraction(hyp_im_plot2d_masked[:100, :], 'Isomap', n_neighbors=30, n_components=n_comp)
                    Plots_Isomap40[counter, :], Latent_Features['Isomap40_plot_'+str(plotnum)+'_row_'+str(1+rownum)] = latent_feature_extraction(hyp_im_plot2d_masked[:100, :], 'Isomap', n_neighbors=40, n_components=n_comp)
                    Plots_Isomap50[counter, :], Latent_Features['Isomap50_plot_'+str(plotnum)+'_row_'+str(1+rownum)] = latent_feature_extraction(hyp_im_plot2d_masked[:100, :], 'Isomap', n_neighbors=50, n_components=n_comp)
                    Plots_LLE10[counter, :], Latent_Features['LLE10_plot_'+str(plotnum)+'_row_'+str(1+rownum)] = latent_feature_extraction(hyp_im_plot2d_masked[:100, :], 'LLE', n_neighbors=10, n_components=n_comp)
                    Plots_LLE20[counter, :], Latent_Features['LLE20_plot_'+str(plotnum)+'_row_'+str(1+rownum)] = latent_feature_extraction(hyp_im_plot2d_masked[:100, :], 'LLE', n_neighbors=20, n_components=n_comp)
                    Plots_LLE30[counter, :], Latent_Features['LLE30_plot_'+str(plotnum)+'_row_'+str(1+rownum)] = latent_feature_extraction(hyp_im_plot2d_masked[:100, :], 'LLE', n_neighbors=30, n_components=n_comp)
                    Plots_LLE40[counter, :], Latent_Features['LLE40_plot_'+str(plotnum)+'_row_'+str(1+rownum)] = latent_feature_extraction(hyp_im_plot2d_masked[:100, :], 'LLE', n_neighbors=40, n_components=n_comp)
                    Plots_LLE50[counter, :], Latent_Features['LLE50_plot_'+str(plotnum)+'_row_'+str(1+rownum)] = latent_feature_extraction(hyp_im_plot2d_masked[:100, :], 'LLE', n_neighbors=50, n_components=n_comp)
    
            if hyp_im_plot2d_masked.shape[0]>0:
                Plots_variability[counter] = np.percentile(hyp_im_plot2d_masked, 75, axis = 0) - np.percentile(hyp_im_plot2d_masked, 25, axis = 0)
            counter = counter + 1
    
    
    all_f = {}
    all_f['plots_number'] = Plots_mean[:,0:1]
    all_f['rows_number'] = Plots_mean[:,1:2]
    all_f['pixels_number'] = Plots_n_pixels
    all_f['spectral_features'] = Plots_mean[:,2:]
    all_f['spc_hist_features'] = Plots_Hist
    all_f['Indices_features'] = Plots_indices
    all_f['Variability_features'] = Plots_variability
    all_f['FDR_features'] = Plots_FDR
    all_f['SDR_features'] = Plots_SDR
    all_f['Integration_features'] = Plots_Intg
    all_f['Derivative_features'] = Plots_Derivative
    all_f['Derivative_features_ave'] = Plots_Derivative_ave
    all_f['wavelet_features'] = Plots_wavelet
    all_f['Indx_hist'] = indx_hist_features
    all_f['PCA'] = Plots_PCA
    all_f['Isomap10'] = Plots_Isomap10
    all_f['Isomap20'] = Plots_Isomap20
    all_f['Isomap30'] = Plots_Isomap30
    all_f['Isomap40'] = Plots_Isomap40
    all_f['Isomap50'] = Plots_Isomap50
    all_f['LLE10'] = Plots_LLE10
    all_f['LLE20'] = Plots_LLE20
    all_f['LLE30'] = Plots_LLE30
    all_f['LLE40'] = Plots_LLE40
    all_f['LLE50'] = Plots_LLE50
    all_f['spectral_features_var'] = Plots_mean_var
    all_f['Indices_features_var'] = Plots_indices_var
    all_f['Integration_features_var'] = Plots_Intg_var
    all_f['Derivative_features_var'] = Plots_Derivative_var
    
    
    
    print('size of the plots number: ', all_f['plots_number'].shape)
    print('size of the rows number: ', all_f['rows_number'].shape)
    print('size of the pixels number: ', all_f['pixels_number'].shape)
    print('size of the spectral features: ', all_f['spectral_features'].shape)
    print('size of the histogram features: ', all_f['spc_hist_features'].shape)
    print('size of the Indices features: ', all_f['Indices_features'].shape)
    print('size of the Variability features: ', all_f['Variability_features'].shape)
    print('size of the FDR features: ', all_f['FDR_features'].shape)
    print('size of the SDR features: ', all_f['SDR_features'].shape)
    print('size of the Integration features: ', all_f['Integration_features'].shape)
    print('size of the Derivative features: ', all_f['Derivative_features'].shape)
    print('size of the Derivative features ave: ', all_f['Derivative_features_ave'].shape)
    print('size of the Wavelet features: ', all_f['wavelet_features'].shape)
    print('size of the Indices hists: ', all_f['Indx_hist'].shape)
    print('size of the PCA: ', all_f['PCA'].shape)
    print('size of the Isomap10: ', all_f['Isomap10'].shape)
    print('size of the Isomap20: ', all_f['Isomap20'].shape)
    print('size of the Isomap30: ', all_f['Isomap30'].shape)
    print('size of the Isomap40: ', all_f['Isomap40'].shape)
    print('size of the Isomap50: ', all_f['Isomap50'].shape)
    print('size of the LLE10: ', all_f['LLE10'].shape)
    print('size of the LLE20: ', all_f['LLE20'].shape)
    print('size of the LLE30: ', all_f['LLE30'].shape)
    print('size of the LLE40: ', all_f['LLE40'].shape)
    print('size of the LLE50: ', all_f['LLE50'].shape)
    print('all features: ', [v for v in all_f.keys()])
    np.save(os.path.join(outdir,'hyper_'+heightdate+'_mean_Hist_features_allrowssep_'+str(n_p_trim)+'trimmed'), all_f)
    np.save(os.path.join(outdir,'hyper_'+heightdate+'_Latent_Features_'+str(n_p_trim)+'trimmed'), Latent_Features)
    ####################################################
    header_to_excel='Plot, # Row, N of pixels, '
    for i in wavelength:
        header_to_excel = header_to_excel + str(i)+','
    header_to_excel = header_to_excel + Indices_names
    data_to_excel = np.concatenate((all_f['plots_number'],all_f['rows_number'], all_f['pixels_number'], all_f['spectral_features'], all_f['Indices_features']), axis=1)
    np.savetxt(os.path.join(outdir,'hyper_'+heightdate+'_mean_plots_'+str(n_p_trim)+'trimmed.csv'), data_to_excel, delimiter=",", header=header_to_excel, fmt='%10.5f')
    
    
    all_f_plot = {}
    all_f_plot['plots_number'] = plots
    all_f_plot['spectral_features'] = np.zeros((n_plots, all_f['spectral_features'].shape[1]))
    all_f_plot['spc_hist_features'] = np.zeros((n_plots, all_f['spc_hist_features'].shape[1]))
    all_f_plot['Indices_features'] = np.zeros((n_plots, all_f['Indices_features'].shape[1]))
    all_f_plot['Variability_features'] = np.zeros((n_plots, all_f['Variability_features'].shape[1]))
    all_f_plot['FDR_features'] = np.zeros((n_plots, all_f['FDR_features'].shape[1]))
    all_f_plot['SDR_features'] = np.zeros((n_plots, all_f['SDR_features'].shape[1]))
    all_f_plot['Integration_features'] = np.zeros((n_plots, all_f['Integration_features'].shape[1]))
    all_f_plot['Derivative_features'] = np.zeros((n_plots, all_f['Derivative_features'].shape[1]))
    all_f_plot['Derivative_features_ave'] = np.zeros((n_plots, all_f['Derivative_features_ave'].shape[1]))
    all_f_plot['wavelet_features'] = np.zeros((n_plots, all_f['wavelet_features'].shape[1]))
    all_f_plot['Indx_hist'] = np.zeros((n_plots, all_f['Indx_hist'].shape[1]))
    all_f_plot['PCA'] = np.zeros((n_plots, all_f['PCA'].shape[1]))
    all_f_plot['Isomap10'] = np.zeros((n_plots, all_f['Isomap10'].shape[1]))
    all_f_plot['Isomap20'] = np.zeros((n_plots, all_f['Isomap20'].shape[1]))
    all_f_plot['Isomap30'] = np.zeros((n_plots, all_f['Isomap30'].shape[1]))
    all_f_plot['Isomap40'] = np.zeros((n_plots, all_f['Isomap40'].shape[1]))
    all_f_plot['Isomap50'] = np.zeros((n_plots, all_f['Isomap50'].shape[1]))
    all_f_plot['LLE10'] = np.zeros((n_plots, all_f['LLE10'].shape[1]))
    all_f_plot['LLE20'] = np.zeros((n_plots, all_f['LLE20'].shape[1]))
    all_f_plot['LLE30'] = np.zeros((n_plots, all_f['LLE30'].shape[1]))
    all_f_plot['LLE40'] = np.zeros((n_plots, all_f['LLE40'].shape[1]))
    all_f_plot['LLE50'] = np.zeros((n_plots, all_f['LLE50'].shape[1]))
    all_f_plot['spectral_features_var'] = np.zeros((n_plots, all_f['spectral_features_var'].shape[1]))
    all_f_plot['Indices_features_var'] = np.zeros((n_plots, all_f['Indices_features_var'].shape[1]))
    all_f_plot['Integration_features_var'] = np.zeros((n_plots, all_f['Integration_features_var'].shape[1]))
    all_f_plot['Derivative_features_var'] = np.zeros((n_plots, all_f['Derivative_features_var'].shape[1]))
    
    for i,plotnum in enumerate(plots):
        ind_plot = np.where(all_f['plots_number'] == plotnum)[0]
        ind = ind_plot[np.where(all_f['rows_number'][ind_plot] == rows_to_extract)[0]]
        Weights = np.squeeze(all_f['pixels_number'][ind])
        if np.sum(Weights)==0:
            Weights[0] = 1
        all_f_plot['spectral_features'][i] = np.average(all_f['spectral_features'][ind], weights=Weights, axis=0)
        all_f_plot['spc_hist_features'][i] = np.average(all_f['spc_hist_features'][ind], weights=Weights, axis=0)
        all_f_plot['Indices_features'][i] = np.average(all_f['Indices_features'][ind], weights=Weights, axis=0)
        all_f_plot['Variability_features'][i] = np.average(all_f['Variability_features'][ind], weights=Weights, axis=0)
        all_f_plot['FDR_features'][i] = np.average(all_f['FDR_features'][ind], weights=Weights, axis=0)
        all_f_plot['SDR_features'][i] = np.average(all_f['SDR_features'][ind], weights=Weights, axis=0)
        all_f_plot['Integration_features'][i] = np.average(all_f['Integration_features'][ind], weights=Weights, axis=0)
        all_f_plot['Derivative_features'][i] = np.average(all_f['Derivative_features'][ind], weights=Weights, axis=0)
        all_f_plot['Derivative_features_ave'][i] = np.average(all_f['Derivative_features_ave'][ind], weights=Weights, axis=0)
        all_f_plot['wavelet_features'][i] = np.average(all_f['wavelet_features'][ind], weights=Weights, axis=0)
        all_f_plot['Indx_hist'][i] = np.average(all_f['Indx_hist'][ind], weights=Weights, axis=0)
        all_f_plot['PCA'][i] = np.average(all_f['PCA'][ind], weights=Weights, axis=0)
        all_f_plot['Isomap10'][i] = np.average(all_f['Isomap10'][ind], weights=Weights, axis=0)
        all_f_plot['Isomap20'][i] = np.average(all_f['Isomap20'][ind], weights=Weights, axis=0)
        all_f_plot['Isomap30'][i] = np.average(all_f['Isomap30'][ind], weights=Weights, axis=0)
        all_f_plot['Isomap40'][i] = np.average(all_f['Isomap40'][ind], weights=Weights, axis=0)
        all_f_plot['Isomap50'][i] = np.average(all_f['Isomap50'][ind], weights=Weights, axis=0)
        all_f_plot['LLE10'][i] = np.average(all_f['LLE10'][ind], weights=Weights, axis=0)
        all_f_plot['LLE20'][i] = np.average(all_f['LLE20'][ind], weights=Weights, axis=0)
        all_f_plot['LLE30'][i] = np.average(all_f['LLE30'][ind], weights=Weights, axis=0)
        all_f_plot['LLE40'][i] = np.average(all_f['LLE40'][ind], weights=Weights, axis=0)
        all_f_plot['LLE50'][i] = np.average(all_f['LLE50'][ind], weights=Weights, axis=0)
        
        var_plus_mean2 = all_f['spectral_features'][ind]**2+all_f['spectral_features_var'][ind]
        all_f_plot['spectral_features_var'][i] = np.average(var_plus_mean2, weights=Weights, axis=0) - all_f_plot['spectral_features'][i]**2
    
        var_plus_mean2 = all_f['Indices_features'][ind]**2+all_f['Indices_features_var'][ind]
        all_f_plot['Indices_features_var'][i] = np.average(var_plus_mean2, weights=Weights, axis=0) - all_f_plot['Indices_features'][i]**2
    
        var_plus_mean2 = all_f['Integration_features'][ind]**2+all_f['Integration_features_var'][ind]
        all_f_plot['Integration_features_var'][i] = np.average(var_plus_mean2, weights=Weights, axis=0) - all_f_plot['Integration_features'][i]**2
    
        var_plus_mean2 = all_f['Derivative_features'][ind]**2+all_f['Derivative_features_var'][ind]
        all_f_plot['Derivative_features_var'][i] = np.average(var_plus_mean2, weights=Weights, axis=0) - all_f_plot['Derivative_features'][i]**2
    
    
    print('size of the plots number: ', all_f_plot['plots_number'].shape)
    print('size of the spectral features: ', all_f_plot['spectral_features'].shape)
    print('size of the histogram features: ', all_f_plot['spc_hist_features'].shape)
    print('size of the Indices features: ', all_f_plot['Indices_features'].shape)
    print('size of the Variability features: ', all_f_plot['Variability_features'].shape)
    print('size of the FDR features: ', all_f_plot['FDR_features'].shape)
    print('size of the SDR features: ', all_f_plot['SDR_features'].shape)
    print('size of the Integration features: ', all_f_plot['Integration_features'].shape)
    print('size of the Derivative features: ', all_f_plot['Derivative_features'].shape)
    print('size of the Derivative features ave: ', all_f_plot['Derivative_features_ave'].shape)
    print('size of the Wavelet features: ', all_f_plot['wavelet_features'].shape)
    print('size of the Indices hists: ', all_f_plot['Indx_hist'].shape)
    print('size of the PCA: ', all_f_plot['PCA'].shape)
    print('size of the Isomap10: ', all_f_plot['Isomap10'].shape)
    print('size of the Isomap20: ', all_f_plot['Isomap20'].shape)
    print('size of the Isomap30: ', all_f_plot['Isomap30'].shape)
    print('size of the Isomap40: ', all_f_plot['Isomap40'].shape)
    print('size of the Isomap50: ', all_f_plot['Isomap50'].shape)
    print('size of the LLE10: ', all_f_plot['LLE10'].shape)
    print('size of the LLE20: ', all_f_plot['LLE20'].shape)
    print('size of the LLE30: ', all_f_plot['LLE30'].shape)
    print('size of the LLE40: ', all_f_plot['LLE40'].shape)
    print('size of the LLE50: ', all_f_plot['LLE50'].shape)
    
    print('size of the spectral features variance: ', all_f_plot['spectral_features_var'].shape)
    print('size of the Indices features variance: ', all_f_plot['Indices_features_var'].shape)
    print('size of the Integration features variance: ', all_f_plot['Integration_features_var'].shape)
    print('size of the Derivative features variance: ', all_f_plot['Derivative_features_var'].shape)
    
    print('all features: ', [v for v in all_f_plot.keys()])
    np.save(os.path.join(outdir,'hyper_'+heightdate+'_mean_Hist_features_meanrows_'+str(n_p_trim)+'trimmed_'+''.join(str(e) for e in rows_to_extract)), all_f_plot)
    
    
    #########################################################################
    #########################################################################
    #########################################################################
    #########################################################################
    #########################################################################
    #########################################################################
    n_p_trim = 10
    grid = copy.deepcopy(grid_org)
    Map_PlotNumber = np.zeros((x.shape[0],x.shape[1]))
    Map_ClassNumber = np.zeros((x.shape[0],x.shape[1]))
    for plotnum in plots:
        ind = grid[:,4] == plotnum
        rows = grid[np.where(ind)[0],5]
        for rownum in rows:
            rownum = int(rownum)
            if trim_ax == 'x':
                grid[np.where(ind)[0][rownum-1], 0:4] = grid[np.where(ind)[0][rownum-1], 0:4] + [n_p_trim, 0, -n_p_trim, 0]
            else:
                grid[np.where(ind)[0][rownum-1], 0:4] = grid[np.where(ind)[0][rownum-1], 0:4] + [0, n_p_trim, 0,-n_p_trim]
            plotbound = grid[np.where(ind)[0][rownum-1], 0:4]
            Map_PlotNumber[(int(plotbound[1])-1):int(plotbound[3]), (int(plotbound[0])-1):int(plotbound[2])] = plotnum    
                
    f = plt.figure(figsize = (15,int(14/np.divide(np.float(x.shape[1]),np.float(x.shape[0])))))
    ax1 = f.add_subplot(2,1,1)
    a = ax1.imshow(Map_PlotNumber, clim=(plots[0]-1, plots[-1]+1))
    f.colorbar(a, ax = ax1)
    np.save(os.path.join(outdir + 'Map_PlotNumber'), Map_PlotNumber)
    #plt.close()
    
    f = plt.figure(figsize = (300,100))
    ax1 = f.add_subplot(1,5,1)
    a = ax1.imshow(2*hyper_im_rgb)
    tittle = 'Hyperspectral (RGB) - '+heightdate+' - '+str(4*n_p_trim)+' CM Trimmed'
    plt.title(tittle, fontsize=100)
    plt.axis('off')
    for i in grid:
        rect = patches.Rectangle((i[0],i[1]),i[2]-i[0],i[3]-i[1],linewidth=8,edgecolor='r',facecolor='none')
        matplotlib.pyplot.text(i[0],i[1], str(i[4])[:-2], fontsize=12, bbox=dict(facecolor='w', alpha=0.5))
        ax1.add_patch(rect)
    plt.show()
    f.savefig(os.path.join(outdir+heightdate+'_'+str(4*n_p_trim)+'cm_trimmed.png'), bbox_inches='tight', pad_inches=0.2)
    
    Manifold=False
    from extract_plot_features_pheno_traits import bin_edges_to_centers
    n_hyp_channels = x.shape[2]
    print('n_hyp_channels', n_hyp_channels)
    n_plots = len(plots)
    n_rows = len(rows)
    nbins = 1
    
    n_indices_features = 0
    
    if Indices_features:
        n_indices_features = Indices.shape[2]
        indx_hist_features = np.zeros((n_plots*n_rows, 25))
        print('n_indices_features', n_indices_features)
    
    Plots_n_pixels = np.zeros((n_plots*n_rows, 1))
    Plots_mean = np.zeros((n_plots*n_rows, 2 + n_hyp_channels))
    Plots_indices = np.zeros((n_plots*n_rows, n_indices_features))
    Plots_variability = np.zeros((n_plots*n_rows, n_hyp_channels))
    Plots_FDR = np.zeros((n_plots*n_rows, n_hyp_channels-1))
    Plots_SDR = np.zeros((n_plots*n_rows, n_hyp_channels-2))
    Plots_Intg = np.zeros((n_plots*n_rows, 6))
    Plots_Derivative = np.zeros((n_plots*n_rows, 6))
    Plots_Derivative_ave = np.zeros((n_plots*n_rows, 6))
    Plots_median = np.zeros((n_plots*n_rows, n_hyp_channels))
    Plots_Hist = np.zeros((n_plots*n_rows, nbins**2))
    Plots_wavelet = np.zeros((n_plots*n_rows, 17))
    
    n_comp = 20
    Plots_PCA = np.zeros((n_plots*n_rows, n_comp))
    Plots_Isomap10 = np.zeros((n_plots*n_rows, n_comp))
    Plots_Isomap20 = np.zeros((n_plots*n_rows, n_comp))
    Plots_Isomap30 = np.zeros((n_plots*n_rows, n_comp))
    Plots_Isomap40 = np.zeros((n_plots*n_rows, n_comp))
    Plots_Isomap50 = np.zeros((n_plots*n_rows, n_comp))
    Plots_LLE10 = np.zeros((n_plots*n_rows, n_comp))
    Plots_LLE20 = np.zeros((n_plots*n_rows, n_comp))
    Plots_LLE30 = np.zeros((n_plots*n_rows, n_comp))
    Plots_LLE40 = np.zeros((n_plots*n_rows, n_comp))
    Plots_LLE50 = np.zeros((n_plots*n_rows, n_comp))
    Latent_Features = {}
    
    Plots_mean_var = np.zeros((n_plots*n_rows, n_hyp_channels))
    Plots_indices_var = np.zeros((n_plots*n_rows, n_indices_features))
    Plots_Intg_var = np.zeros((n_plots*n_rows, 6))
    Plots_Derivative_var = np.zeros((n_plots*n_rows, 6))
    
    counter = 0
    for plotnum in plots:
        plotnum = int(plotnum)
        ind = grid[:,4] == plotnum
        rows = grid[np.where(ind)[0], 5]
        for rownum in rows:
            rownum = int(rownum)
            hyp_im_plot, plotbound = extract_row_plot_from_field_rgb_image(x, grid, plotnum, rownum)
            hyp_im_plot_FDR, plotbound = extract_row_plot_from_field_rgb_image(x_FDR, grid, plotnum, rownum)
            hyp_im_plot_SDR, plotbound = extract_row_plot_from_field_rgb_image(x_SDR, grid, plotnum, rownum)
            hyp_im_plot_Intg, plotbound = extract_row_plot_from_field_rgb_image(x_Integration, grid, plotnum, rownum)
            hyp_im_plot_FSDR, plotbound = extract_row_plot_from_field_rgb_image(FSDR_features, grid, plotnum, rownum)
    
            if Indices_features:
                indx_im_plot = extract_row_plot_from_field_rgb_image(Indices, grid, plotnum, rownum)[0]
                counts1, edges1 = np.histogram(indx_im_plot, bins=25)
                bin_centers = bin_edges_to_centers(edges1)
                indx_hist_features[counter, :] = counts1
            
            if plotnum==48010:
                fig = plt.figure(figsize = (5,10))
                opacity = 0.4
                error_config = {'ecolor': '0.3'}
                rects1 = plt.bar(bin_centers, counts1)      
                plt.show()
    
            mask_im_plot = extract_row_plot_from_field_rgb_image(np.atleast_3d(Mask), grid, plotnum, rownum)[0]
            mask_im_plot2d = mask_im_plot.reshape((-1, 1))
    
            hyp_im_plot2d = hyp_im_plot.reshape((-1, n_hyp_channels))
            indx_im_plot2d = indx_im_plot.reshape((-1, n_indices_features))
            hyp_im_plot_FDR2d = hyp_im_plot_FDR.reshape((-1, hyp_im_plot_FDR.shape[2]))
            hyp_im_plot_SDR2d = hyp_im_plot_SDR.reshape((-1, hyp_im_plot_SDR.shape[2]))
            hyp_im_plot_Intg2d = hyp_im_plot_Intg.reshape((-1, hyp_im_plot_Intg.shape[2]))
            hyp_im_plot_FSDR2d = hyp_im_plot_FSDR.reshape((-1, hyp_im_plot_FSDR.shape[2]))
    
            hyp_im_plot2d_masked = hyp_im_plot2d[np.squeeze(~mask_im_plot2d), :]
            indx_im_plot2d_masked = indx_im_plot2d[np.squeeze(~mask_im_plot2d), :]
            hyp_im_plot_FDR2d_masked = hyp_im_plot_FDR2d[np.squeeze(~mask_im_plot2d), :]
            hyp_im_plot_SDR2d_masked = hyp_im_plot_SDR2d[np.squeeze(~mask_im_plot2d), :]
            hyp_im_plot_Intg2d_masked = hyp_im_plot_Intg2d[np.squeeze(~mask_im_plot2d), :]
            hyp_im_plot_FSDR2d_masked = hyp_im_plot_FSDR2d[np.squeeze(~mask_im_plot2d), :]
            
            if wavelet_ex:
                hyp_im_plot_wave, plotbound = extract_row_plot_from_field_rgb_image(hyp_wavelet_012_resample, grid, plotnum, rownum)
                hyp_im_plot_wave2d = hyp_im_plot_wave.reshape((-1, hyp_im_plot_wave.shape[2]))
                Plots_wavelet[counter]= np.mean(hyp_im_plot_wave2d, axis = 0)
                
            mean_hyp = np.mean(hyp_im_plot2d_masked, axis = 0)
            mean_hyp_fdr = np.mean(hyp_im_plot_FDR2d_masked, axis = 0)
            mean_hyp_sdr = np.mean(hyp_im_plot_SDR2d_masked, axis = 0)
            
            Plots_n_pixels[counter] = float(len(np.where(mask_im_plot2d==0)[0]))
            Plots_mean[counter][0] = plotnum
            Plots_mean[counter][1] = rownum
            Plots_mean[counter][2:] = mean_hyp
            Plots_median[counter] = np.median(hyp_im_plot2d_masked, axis = 0)
            Plots_indices[counter] = np.mean(indx_im_plot2d_masked, axis = 0)
            Plots_FDR[counter] = mean_hyp_fdr
            Plots_SDR[counter] = mean_hyp_sdr
            Plots_Intg[counter] = np.mean(hyp_im_plot_Intg2d_masked, axis = 0)
            Plots_Derivative[counter] = np.mean(hyp_im_plot_FSDR2d_masked, axis = 0)
    
            min_green = np.argmin(mean_hyp_fdr[np.argmin(abs(wavelength-500)):np.argmin(abs(wavelength-600))])+ np.argmin(abs(wavelength-500))
            max_red = np.argmax(mean_hyp_fdr[np.argmin(abs(wavelength-500)):np.argmin(abs(wavelength-600))])+ np.argmin(abs(wavelength-500))
            fdr_slope = (mean_hyp_fdr[min_green]-mean_hyp_fdr[max_red]) / (min_green-max_red)
            min_nir = np.argmin(mean_hyp_fdr[np.argmin(abs(wavelength-680)):np.argmin(abs(wavelength-760))])+np.argmin(abs(wavelength-680))
            min_fdr = mean_hyp_fdr[min_nir]
            band670 = np.argmin(abs(wavelength-670))
            band780 = np.argmin(abs(wavelength-780))
            integration_NIR = np.trapz(mean_hyp_fdr[band670:band780], x=np.squeeze(wavelength[band670:band780]))
            band910 = np.argmin(abs(wavelength-910))
            band1000 = np.argmin(abs(wavelength-1000))
            integration_FNIR = np.trapz(mean_hyp_fdr[band910:band1000], x=np.squeeze(wavelength[band910:band1000]))
            min_nir_sdr = np.argmin(mean_hyp_sdr[np.argmin(abs(wavelength-650)):np.argmin(abs(wavelength-800))])+ np.argmin(abs(wavelength-650))
            max_nir_sdr = np.argmax(mean_hyp_sdr[np.argmin(abs(wavelength-650)):np.argmin(abs(wavelength-800))])+ np.argmin(abs(wavelength-650))
            sdr_slope = (mean_hyp_sdr[min_nir_sdr]-mean_hyp_sdr[max_nir_sdr]) / (min_nir_sdr-max_nir_sdr)
            integration_sdr = np.trapz(mean_hyp_sdr[:], x=np.squeeze(wavelength[:-2]))
            
            Plots_Derivative_ave[counter][0] = fdr_slope
            Plots_Derivative_ave[counter][1] = min_fdr
            Plots_Derivative_ave[counter][2] = integration_NIR
            Plots_Derivative_ave[counter][3] = integration_FNIR
            Plots_Derivative_ave[counter][4] = sdr_slope
            Plots_Derivative_ave[counter][5] = integration_sdr
            
    
            
            Plots_mean_var[counter] = np.var(hyp_im_plot2d_masked, axis = 0)
            Plots_indices_var[counter] = np.var(indx_im_plot2d_masked, axis = 0)
            Plots_Intg_var[counter] = np.var(hyp_im_plot_Intg2d_masked, axis = 0)
            Plots_Derivative_var[counter] = np.var(hyp_im_plot_FSDR2d_masked, axis = 0)
    
            if Manifold==True:
                if rownum>0 and rownum<3:
                    Plots_PCA[counter, :], Latent_Features['PCA_plot_'+str(plotnum)+'_row_'+str(1+rownum)]  = latent_feature_extraction(hyp_im_plot2d_masked[:100, :], 'PCA', n_components=n_comp)
                    Plots_Isomap10[counter, :], Latent_Features['Isomap10_plot_'+str(plotnum)+'_row_'+str(1+rownum)] = latent_feature_extraction(hyp_im_plot2d_masked[:100, :], 'Isomap', n_neighbors=10, n_components=n_comp)
                    Plots_Isomap20[counter, :], Latent_Features['Isomap20_plot_'+str(plotnum)+'_row_'+str(1+rownum)] = latent_feature_extraction(hyp_im_plot2d_masked[:100, :], 'Isomap', n_neighbors=20, n_components=n_comp)
                    Plots_Isomap30[counter, :], Latent_Features['Isomap30_plot_'+str(plotnum)+'_row_'+str(1+rownum)] = latent_feature_extraction(hyp_im_plot2d_masked[:100, :], 'Isomap', n_neighbors=30, n_components=n_comp)
                    Plots_Isomap40[counter, :], Latent_Features['Isomap40_plot_'+str(plotnum)+'_row_'+str(1+rownum)] = latent_feature_extraction(hyp_im_plot2d_masked[:100, :], 'Isomap', n_neighbors=40, n_components=n_comp)
                    Plots_Isomap50[counter, :], Latent_Features['Isomap50_plot_'+str(plotnum)+'_row_'+str(1+rownum)] = latent_feature_extraction(hyp_im_plot2d_masked[:100, :], 'Isomap', n_neighbors=50, n_components=n_comp)
                    Plots_LLE10[counter, :], Latent_Features['LLE10_plot_'+str(plotnum)+'_row_'+str(1+rownum)] = latent_feature_extraction(hyp_im_plot2d_masked[:100, :], 'LLE', n_neighbors=10, n_components=n_comp)
                    Plots_LLE20[counter, :], Latent_Features['LLE20_plot_'+str(plotnum)+'_row_'+str(1+rownum)] = latent_feature_extraction(hyp_im_plot2d_masked[:100, :], 'LLE', n_neighbors=20, n_components=n_comp)
                    Plots_LLE30[counter, :], Latent_Features['LLE30_plot_'+str(plotnum)+'_row_'+str(1+rownum)] = latent_feature_extraction(hyp_im_plot2d_masked[:100, :], 'LLE', n_neighbors=30, n_components=n_comp)
                    Plots_LLE40[counter, :], Latent_Features['LLE40_plot_'+str(plotnum)+'_row_'+str(1+rownum)] = latent_feature_extraction(hyp_im_plot2d_masked[:100, :], 'LLE', n_neighbors=40, n_components=n_comp)
                    Plots_LLE50[counter, :], Latent_Features['LLE50_plot_'+str(plotnum)+'_row_'+str(1+rownum)] = latent_feature_extraction(hyp_im_plot2d_masked[:100, :], 'LLE', n_neighbors=50, n_components=n_comp)
    
            if hyp_im_plot2d_masked.shape[0]>0:
                Plots_variability[counter] = np.percentile(hyp_im_plot2d_masked, 75, axis = 0) - np.percentile(hyp_im_plot2d_masked, 25, axis = 0)
            counter = counter + 1
    
    
    all_f = {}
    all_f['plots_number'] = Plots_mean[:,0:1]
    all_f['rows_number'] = Plots_mean[:,1:2]
    all_f['pixels_number'] = Plots_n_pixels
    all_f['spectral_features'] = Plots_mean[:,2:]
    all_f['spc_hist_features'] = Plots_Hist
    all_f['Indices_features'] = Plots_indices
    all_f['Variability_features'] = Plots_variability
    all_f['FDR_features'] = Plots_FDR
    all_f['SDR_features'] = Plots_SDR
    all_f['Integration_features'] = Plots_Intg
    all_f['Derivative_features'] = Plots_Derivative
    all_f['Derivative_features_ave'] = Plots_Derivative_ave
    all_f['wavelet_features'] = Plots_wavelet
    all_f['Indx_hist'] = indx_hist_features
    all_f['PCA'] = Plots_PCA
    all_f['Isomap10'] = Plots_Isomap10
    all_f['Isomap20'] = Plots_Isomap20
    all_f['Isomap30'] = Plots_Isomap30
    all_f['Isomap40'] = Plots_Isomap40
    all_f['Isomap50'] = Plots_Isomap50
    all_f['LLE10'] = Plots_LLE10
    all_f['LLE20'] = Plots_LLE20
    all_f['LLE30'] = Plots_LLE30
    all_f['LLE40'] = Plots_LLE40
    all_f['LLE50'] = Plots_LLE50
    all_f['spectral_features_var'] = Plots_mean_var
    all_f['Indices_features_var'] = Plots_indices_var
    all_f['Integration_features_var'] = Plots_Intg_var
    all_f['Derivative_features_var'] = Plots_Derivative_var
    
    
    
    print('size of the plots number: ', all_f['plots_number'].shape)
    print('size of the rows number: ', all_f['rows_number'].shape)
    print('size of the pixels number: ', all_f['pixels_number'].shape)
    print('size of the spectral features: ', all_f['spectral_features'].shape)
    print('size of the histogram features: ', all_f['spc_hist_features'].shape)
    print('size of the Indices features: ', all_f['Indices_features'].shape)
    print('size of the Variability features: ', all_f['Variability_features'].shape)
    print('size of the FDR features: ', all_f['FDR_features'].shape)
    print('size of the SDR features: ', all_f['SDR_features'].shape)
    print('size of the Integration features: ', all_f['Integration_features'].shape)
    print('size of the Derivative features: ', all_f['Derivative_features'].shape)
    print('size of the Derivative features ave: ', all_f['Derivative_features_ave'].shape)
    print('size of the Wavelet features: ', all_f['wavelet_features'].shape)
    print('size of the Indices hists: ', all_f['Indx_hist'].shape)
    print('size of the PCA: ', all_f['PCA'].shape)
    print('size of the Isomap10: ', all_f['Isomap10'].shape)
    print('size of the Isomap20: ', all_f['Isomap20'].shape)
    print('size of the Isomap30: ', all_f['Isomap30'].shape)
    print('size of the Isomap40: ', all_f['Isomap40'].shape)
    print('size of the Isomap50: ', all_f['Isomap50'].shape)
    print('size of the LLE10: ', all_f['LLE10'].shape)
    print('size of the LLE20: ', all_f['LLE20'].shape)
    print('size of the LLE30: ', all_f['LLE30'].shape)
    print('size of the LLE40: ', all_f['LLE40'].shape)
    print('size of the LLE50: ', all_f['LLE50'].shape)
    print('all features: ', [v for v in all_f.keys()])
    np.save(os.path.join(outdir,'hyper_'+heightdate+'_mean_Hist_features_allrowssep_'+str(n_p_trim)+'trimmed'), all_f)
    np.save(os.path.join(outdir,'hyper_'+heightdate+'_Latent_Features_'+str(n_p_trim)+'trimmed'), Latent_Features)
    
    ###################################################################
    header_to_excel='Plot, # Row, N of pixels, '
    for i in wavelength:
        header_to_excel = header_to_excel + str(i)+','
    header_to_excel = header_to_excel + Indices_names
    data_to_excel = np.concatenate((all_f['plots_number'],all_f['rows_number'], all_f['pixels_number'], all_f['spectral_features'], all_f['Indices_features']), axis=1)
    np.savetxt(os.path.join(outdir,'hyper_'+heightdate+'_mean_plots_'+str(n_p_trim)+'trimmed.csv'), data_to_excel, delimiter=",", header=header_to_excel, fmt='%10.5f')
    ###################################################################
    
    all_f_plot = {}
    all_f_plot['plots_number'] = plots
    all_f_plot['spectral_features'] = np.zeros((n_plots, all_f['spectral_features'].shape[1]))
    all_f_plot['spc_hist_features'] = np.zeros((n_plots, all_f['spc_hist_features'].shape[1]))
    all_f_plot['Indices_features'] = np.zeros((n_plots, all_f['Indices_features'].shape[1]))
    all_f_plot['Variability_features'] = np.zeros((n_plots, all_f['Variability_features'].shape[1]))
    all_f_plot['FDR_features'] = np.zeros((n_plots, all_f['FDR_features'].shape[1]))
    all_f_plot['SDR_features'] = np.zeros((n_plots, all_f['SDR_features'].shape[1]))
    all_f_plot['Integration_features'] = np.zeros((n_plots, all_f['Integration_features'].shape[1]))
    all_f_plot['Derivative_features'] = np.zeros((n_plots, all_f['Derivative_features'].shape[1]))
    all_f_plot['Derivative_features_ave'] = np.zeros((n_plots, all_f['Derivative_features_ave'].shape[1]))
    all_f_plot['wavelet_features'] = np.zeros((n_plots, all_f['wavelet_features'].shape[1]))
    all_f_plot['Indx_hist'] = np.zeros((n_plots, all_f['Indx_hist'].shape[1]))
    all_f_plot['PCA'] = np.zeros((n_plots, all_f['PCA'].shape[1]))
    all_f_plot['Isomap10'] = np.zeros((n_plots, all_f['Isomap10'].shape[1]))
    all_f_plot['Isomap20'] = np.zeros((n_plots, all_f['Isomap20'].shape[1]))
    all_f_plot['Isomap30'] = np.zeros((n_plots, all_f['Isomap30'].shape[1]))
    all_f_plot['Isomap40'] = np.zeros((n_plots, all_f['Isomap40'].shape[1]))
    all_f_plot['Isomap50'] = np.zeros((n_plots, all_f['Isomap50'].shape[1]))
    all_f_plot['LLE10'] = np.zeros((n_plots, all_f['LLE10'].shape[1]))
    all_f_plot['LLE20'] = np.zeros((n_plots, all_f['LLE20'].shape[1]))
    all_f_plot['LLE30'] = np.zeros((n_plots, all_f['LLE30'].shape[1]))
    all_f_plot['LLE40'] = np.zeros((n_plots, all_f['LLE40'].shape[1]))
    all_f_plot['LLE50'] = np.zeros((n_plots, all_f['LLE50'].shape[1]))
    all_f_plot['spectral_features_var'] = np.zeros((n_plots, all_f['spectral_features_var'].shape[1]))
    all_f_plot['Indices_features_var'] = np.zeros((n_plots, all_f['Indices_features_var'].shape[1]))
    all_f_plot['Integration_features_var'] = np.zeros((n_plots, all_f['Integration_features_var'].shape[1]))
    all_f_plot['Derivative_features_var'] = np.zeros((n_plots, all_f['Derivative_features_var'].shape[1]))
    
    for i,plotnum in enumerate(plots):
        ind_plot = np.where(all_f['plots_number'] == plotnum)[0]
        if plotnum>6440.0 and plotnum<6461.0:
            ind = ind_plot[np.where(all_f['rows_number'][ind_plot] == [1, 2])[0]]
        else:
            ind = ind_plot[np.where(all_f['rows_number'][ind_plot] == rows_to_extract)[0]]
        Weights = np.squeeze(all_f['pixels_number'][ind])
        if np.sum(Weights)==0:
            Weights[0] = 1
        all_f_plot['spectral_features'][i] = np.average(all_f['spectral_features'][ind], weights=Weights, axis=0)
        all_f_plot['spc_hist_features'][i] = np.average(all_f['spc_hist_features'][ind], weights=Weights, axis=0)
        all_f_plot['Indices_features'][i] = np.average(all_f['Indices_features'][ind], weights=Weights, axis=0)
        all_f_plot['Variability_features'][i] = np.average(all_f['Variability_features'][ind], weights=Weights, axis=0)
        all_f_plot['FDR_features'][i] = np.average(all_f['FDR_features'][ind], weights=Weights, axis=0)
        all_f_plot['SDR_features'][i] = np.average(all_f['SDR_features'][ind], weights=Weights, axis=0)
        all_f_plot['Integration_features'][i] = np.average(all_f['Integration_features'][ind], weights=Weights, axis=0)
        all_f_plot['Derivative_features'][i] = np.average(all_f['Derivative_features'][ind], weights=Weights, axis=0)
        all_f_plot['Derivative_features_ave'][i] = np.average(all_f['Derivative_features_ave'][ind], weights=Weights, axis=0)
        all_f_plot['wavelet_features'][i] = np.average(all_f['wavelet_features'][ind], weights=Weights, axis=0)
        all_f_plot['Indx_hist'][i] = np.average(all_f['Indx_hist'][ind], weights=Weights, axis=0)
        all_f_plot['PCA'][i] = np.average(all_f['PCA'][ind], weights=Weights, axis=0)
        all_f_plot['Isomap10'][i] = np.average(all_f['Isomap10'][ind], weights=Weights, axis=0)
        all_f_plot['Isomap20'][i] = np.average(all_f['Isomap20'][ind], weights=Weights, axis=0)
        all_f_plot['Isomap30'][i] = np.average(all_f['Isomap30'][ind], weights=Weights, axis=0)
        all_f_plot['Isomap40'][i] = np.average(all_f['Isomap40'][ind], weights=Weights, axis=0)
        all_f_plot['Isomap50'][i] = np.average(all_f['Isomap50'][ind], weights=Weights, axis=0)
        all_f_plot['LLE10'][i] = np.average(all_f['LLE10'][ind], weights=Weights, axis=0)
        all_f_plot['LLE20'][i] = np.average(all_f['LLE20'][ind], weights=Weights, axis=0)
        all_f_plot['LLE30'][i] = np.average(all_f['LLE30'][ind], weights=Weights, axis=0)
        all_f_plot['LLE40'][i] = np.average(all_f['LLE40'][ind], weights=Weights, axis=0)
        all_f_plot['LLE50'][i] = np.average(all_f['LLE50'][ind], weights=Weights, axis=0)
        
        var_plus_mean2 = all_f['spectral_features'][ind]**2+all_f['spectral_features_var'][ind]
        all_f_plot['spectral_features_var'][i] = np.average(var_plus_mean2, weights=Weights, axis=0) - all_f_plot['spectral_features'][i]**2
    
        var_plus_mean2 = all_f['Indices_features'][ind]**2+all_f['Indices_features_var'][ind]
        all_f_plot['Indices_features_var'][i] = np.average(var_plus_mean2, weights=Weights, axis=0) - all_f_plot['Indices_features'][i]**2
    
        var_plus_mean2 = all_f['Integration_features'][ind]**2+all_f['Integration_features_var'][ind]
        all_f_plot['Integration_features_var'][i] = np.average(var_plus_mean2, weights=Weights, axis=0) - all_f_plot['Integration_features'][i]**2
    
        var_plus_mean2 = all_f['Derivative_features'][ind]**2+all_f['Derivative_features_var'][ind]
        all_f_plot['Derivative_features_var'][i] = np.average(var_plus_mean2, weights=Weights, axis=0) - all_f_plot['Derivative_features'][i]**2
    
    
    print('size of the plots number: ', all_f_plot['plots_number'].shape)
    print('size of the spectral features: ', all_f_plot['spectral_features'].shape)
    print('size of the histogram features: ', all_f_plot['spc_hist_features'].shape)
    print('size of the Indices features: ', all_f_plot['Indices_features'].shape)
    print('size of the Variability features: ', all_f_plot['Variability_features'].shape)
    print('size of the FDR features: ', all_f_plot['FDR_features'].shape)
    print('size of the SDR features: ', all_f_plot['SDR_features'].shape)
    print('size of the Integration features: ', all_f_plot['Integration_features'].shape)
    print('size of the Derivative features: ', all_f_plot['Derivative_features'].shape)
    print('size of the Derivative features ave: ', all_f_plot['Derivative_features_ave'].shape)
    print('size of the Wavelet features: ', all_f_plot['wavelet_features'].shape)
    print('size of the Indices hists: ', all_f_plot['Indx_hist'].shape)
    print('size of the PCA: ', all_f_plot['PCA'].shape)
    print('size of the Isomap10: ', all_f_plot['Isomap10'].shape)
    print('size of the Isomap20: ', all_f_plot['Isomap20'].shape)
    print('size of the Isomap30: ', all_f_plot['Isomap30'].shape)
    print('size of the Isomap40: ', all_f_plot['Isomap40'].shape)
    print('size of the Isomap50: ', all_f_plot['Isomap50'].shape)
    print('size of the LLE10: ', all_f_plot['LLE10'].shape)
    print('size of the LLE20: ', all_f_plot['LLE20'].shape)
    print('size of the LLE30: ', all_f_plot['LLE30'].shape)
    print('size of the LLE40: ', all_f_plot['LLE40'].shape)
    print('size of the LLE50: ', all_f_plot['LLE50'].shape)
    
    print('size of the spectral features variance: ', all_f_plot['spectral_features_var'].shape)
    print('size of the Indices features variance: ', all_f_plot['Indices_features_var'].shape)
    print('size of the Integration features variance: ', all_f_plot['Integration_features_var'].shape)
    print('size of the Derivative features variance: ', all_f_plot['Derivative_features_var'].shape)
    
    print('all features: ', [v for v in all_f_plot.keys()])
    np.save(os.path.join(outdir,'hyper_'+heightdate+'_mean_Hist_features_meanrows_'+str(n_p_trim)+'trimmed_'+''.join(str(e) for e in rows_to_extract)), all_f_plot)
    
    #############################################################
    ###############################################################################
    Summery_Data = {}
    ind = np.where(all_f['rows_number'] == rows_to_extract)[0]
    Weights = np.squeeze(all_f['pixels_number'][ind])
    if np.sum(Weights)==0:
        Weights[0] = 1
    Summery_Data['spectral_features'] = np.average(all_f['spectral_features'][ind], weights=Weights, axis=0)
    Summery_Data['Indices_features'] = np.average(all_f['Indices_features'][ind], weights=Weights, axis=0)
    Summery_Data['Integration_features'] = np.average(all_f['Integration_features'][ind], weights=Weights, axis=0)
    Summery_Data['Derivative_features'] = np.average(all_f['Derivative_features'][ind], weights=Weights, axis=0)
    
    
    var_plus_mean2 = all_f['spectral_features'][ind]**2+all_f['spectral_features_var'][ind]
    Summery_Data['spectral_features_var'] = np.average(var_plus_mean2, weights=Weights, axis=0) - Summery_Data['spectral_features']**2
    
    var_plus_mean2 = all_f['Indices_features'][ind]**2+all_f['Indices_features_var'][ind]
    Summery_Data['Indices_features_var'] = np.average(var_plus_mean2, weights=Weights, axis=0) - Summery_Data['Indices_features']**2
    
    var_plus_mean2 = all_f['Integration_features'][ind]**2+all_f['Integration_features_var'][ind]
    Summery_Data['Integration_features_var'] = np.average(var_plus_mean2, weights=Weights, axis=0) - Summery_Data['Integration_features']**2
    
    var_plus_mean2 = all_f['Derivative_features'][ind]**2+all_f['Derivative_features_var'][ind]
    Summery_Data['Derivative_features_var'] = np.average(var_plus_mean2, weights=Weights, axis=0) - Summery_Data['Derivative_features']**2
    
    np.save(os.path.join(outdir,'hyper_'+heightdate+'_mean_var_plots_'+str(n_p_trim)+'trimmed_'+''.join(str(e) for e in rows_to_extract)), Summery_Data)
    
    '''
    for plotnum in plots:
        
        f = plt.figure(figsize = (30,15))
        ax1 = f.add_subplot(1,1,1)
        f.suptitle('plot number: '+ str(plotnum)[:4], fontsize=24)
        a = ax1.plot(np.squeeze(wavelength), all_f_plot['spectral_features'][int(plotnum-plots[0]), 0:], label="Nano")
        #a = ax1.plot(np.squeeze(wavelength[272:]), all_f_plot['spectral_features'][plotnum-plots[0], 272:], label="SWIR")
        #a = ax1.plot(np.squeeze(Nano_wavelength), np.ones(136))
        ax1.set_ylim([0.0,1])
        plt.legend(fontsize=20)
        plt.xlabel('Wavelength (nm)', fontsize=20)
        plt.ylabel('Reflectance', fontsize=20)
        plt.yticks(size=18)
        plt.xticks(size=18)
        
        plt.savefig(os.path.join(outdir+'Mean_plot_'+str(plotnum)[:4]+'.png'), bbox_inches='tight', pad_inches=0.2)
        plt.close()
    '''
    
    for i in list(globals().keys()):
        if(i[0] != '_'):
            try:
                size_var = globals()[i].nbytes
                if size_var>50000 and i!='np' and i!='scipy':
                    del globals()[i]
            except:
                continue
    gc.collect()