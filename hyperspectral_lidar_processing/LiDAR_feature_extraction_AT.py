# -*- coding: utf-8 -*-
"""
Created on Mon May 10 11:29:40 2021
@author: wang3241
Modified by huan1577
"""
import ast
import numpy as np
import pandas as pd
import os
from laspy.file import File
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew, variation
from matplotlib.path import Path
import time
import configparser
from scipy.spatial import ConvexHull 
from scipy.spatial import cKDTree as KDTree
import argparse
#import CSF 
import json

def cluster_MV(input,radius,acceptable_points):
    qqq=np.percentile(input[:,4],75)
    idx=np.logical_and(input[:,4]>(qqq-0.05),input[:,4]<(qqq+0.05))
    x=input[:,0][idx]
    y=input[:,1][idx]
    z=input[:,4][idx]
    pairs=[(x[i],y[i],z[i]) for i in range(len(z))]

    rS=radius
    Mdl=KDTree(pairs)
    cluster=[[] for i in range(len(pairs))]
    for i in range(len(pairs)):
        idx_S=Mdl.query_ball_point(pairs[i],rS)
        for j in idx_S:
            cluster[i].append(pairs[j])
    out = []
    while len(cluster)>0:
        first, *rest = cluster
        first=set(first)
        lf = -1
        while len(first)>lf: # this condition means there is no intersection between first and rest
            lf = len(first)

            rest2 = []
            for r in rest:
                if len(first.intersection(set(r)))>0:
                    first |= set(r)
                else:
                    rest2.append(r)     
            rest = rest2

        out.append(first)
        cluster = rest
        point_number=[]
    for i in range(len(out)):
        point_number.append(len(out[i]))
    point_number=np.array(point_number)
    out2=[]
    for i in range(len(out)):
        if point_number[i]>acceptable_points:
            out2.append(out[i])
    out2=list(out2)
    for i in range(len(out2)):
        out2[i]=list(out2[i])
    return out2
def get_CAP(ng):
    clust=cluster_MV(ng,0.15,10)
    area=0
    for i in clust:
        area=area+ConvexHull(i).area
    return area
def get_vol(ng):
    clust1=cluster_MV(ng,0.08,15)
    vol=0
    for i in clust1:
        vol=vol+ConvexHull(i).volume
    return vol
def get_VCI(ng):
    h=ng[:,4]
    min_z=np.min(h)
    max_z=np.percentile(h,0.95)
    p=h<max_z
    p=np.sum(p)
    VCI_HB_const_temp=[]
    bin=(max_z-min_z)/4
    for i in range(4):
        indexx=None
        q=None
        min_bin=min_z+bin*i
        max_bin=min_z+bin*(i+1)
        indexx=np.logical_and(ng[:,4]>=min_bin,ng[:,4]<max_bin)
        q=np.sum(indexx)
        VCI_HB_const_temp.append(q/p*np.log(q/p))
    VCI_HB_const=-np.sum(VCI_HB_const_temp)/np.log(4)
    return VCI_HB_const
def stat_feature_extraction(LiDAR_Feature, label):
    Standard_deviation = np.std(LiDAR_Feature)
    Quadratic_mean = np.sqrt(np.mean(np.square(LiDAR_Feature)))
    Skewness = skew(LiDAR_Feature)
    Kurtosis = kurtosis(LiDAR_Feature, fisher=False)
    Variation = variation(LiDAR_Feature)
    return [Standard_deviation, Quadratic_mean, Skewness, Kurtosis, Variation]

def LII_v2_extraction(LiDAR_Feature, num_pts, plot_height):
    canopy_h_thereshold = np.multiply([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75], plot_height)
    LII = []
    for i in canopy_h_thereshold:
        LII.append(round(10000*float(len(LiDAR_Feature[LiDAR_Feature>i])) / float(num_pts))/100)
    return LII

def bin_edges_to_centers(bin_edges):
    bin_centers = np.zeros(len(bin_edges)-1)
    for b in range(len(bin_edges)-1):
        bin_centers[b] = 0.5*(bin_edges[b]+bin_edges[b+1])
    return bin_centers


def read_point_cloud(lidardate, datadir, outdir, field_number, field_panel, field_boundary, file_name_to_save, file_name): 
    out_path = outdir+'/'+lidardate+'/'+field_number+'/'+field_panel +'/' 
    print('output path: ', outdir)
    if os.path.isdir(out_path)==False:
        os.makedirs(out_path)
        
    try:
        inFile = File(datadir + '/'+file_name, mode='rw')
    except:
        inFile = File(datadir+ '/'+file_name[1], mode='rw')

    dataset = np.vstack([inFile.x, inFile.y, inFile.z, inFile.intensity]).transpose()

    # Get arrays which indicate invalid X, Y, or Z values.
    x_sub1 = field_boundary[0]
    x_sub2 = field_boundary[2]
    y_sub1 = field_boundary[1]
    y_sub2 = field_boundary[3]
    X_sub = np.logical_and((x_sub1 <= inFile.x),
                          (x_sub2 >= inFile.x))
    Y_sub = np.logical_and((y_sub2 <= inFile.y),
                          (y_sub1 >= inFile.y))
    subset_id = np.where(np.logical_and(X_sub, Y_sub))
    dataset_sub = dataset[subset_id]
    print('sub area points precentage: ', len(dataset_sub)/len(inFile.points) )
    
    output_file = File(out_path+'/' + file_name_to_save, mode = "w", header = inFile.header)
    output_file.points = inFile.points[subset_id]
    output_file.close()
    return out_path

def extract_row23_point_cloud(dataset_sub, dataset_sub_NG, DEM_x, DEM_y, DEM_z, plot_grid_row23, LEFT_X, TOP_Y, res_extract,PLOT_ID):
    # add one column to non-ground points, to record relatvie height
    # dataset_sub_NG: x, y, z, intensity,  relative z
    dataset_sub_NG = np.hstack((dataset_sub_NG, np.zeros((len(dataset_sub_NG),1))))
    
    count_bad_pixels = 0
    for i in range(len(dataset_sub_NG)):
        col_grid = int( round((dataset_sub_NG[i,0] - LEFT_X)/res_extract))
        row_grid = int( round((TOP_Y - dataset_sub_NG[i,1] )/res_extract))
        relative_h = dataset_sub_NG[i,2] - DEM_z[row_grid-1, col_grid-1]
        if relative_h >= 0:
            dataset_sub_NG[i,4] = relative_h
        else:
            relative_h = 0
            dataset_sub_NG[i,4] = relative_h
            count_bad_pixels = count_bad_pixels + 1
    dataset_sub_NG = dataset_sub_NG[dataset_sub_NG[:,4] > 0]
    print('finished calcualting relative height.')
    
    # find all points in each plot row2&3, with relatvie height
    num_plots = len(PLOT_ID)
    plot_pts = [[] for k in range(num_plots)]
    plot_pts_ng = [[] for k in range(num_plots)]
    
    NG_xy = dataset_sub_NG[:,:2] 
    all_xy = dataset_sub[:,:2] 
    
    print('num_plots=', num_plots)
    
############################ Change this part to increase the efficiency
    for j in range(num_plots):
        tempp=plot_grid_row23[plot_grid_row23['plot_ID']==PLOT_ID[j]]
        for k in range(int(len(tempp['x0']))):
            tupVerts=[(tempp['x0'].iloc[k],tempp['y0'].iloc[k]),  # (1)
                    (tempp['x1'].iloc[k],tempp['y1'].iloc[k]),     # (2)
                    (tempp['x2'].iloc[k],tempp['y2'].iloc[k]), # (3)
                    (tempp['x3'].iloc[k],tempp['y3'].iloc[k])]        
########################### To extract all points in interested plots  
            p = Path(tupVerts)
            if k==0:
                NG_grid = p.contains_points(NG_xy)
                all_grid = p.contains_points(all_xy)
            else:
                tempp_NG=p.contains_points(NG_xy)
                NG_grid=np.logical_or(NG_grid,tempp_NG)
                tempp_all=p.contains_points(all_xy)
                all_grid=np.logical_or(all_grid,tempp_all)

        plot_pts_ng[j] = dataset_sub_NG[NG_grid]
        plot_pts[j] = dataset_sub[all_grid]

    return dataset_sub_NG, plot_pts,plot_pts_ng, count_bad_pixels
        

def bilinear_interpolate(im):
    h_50p = np.percentile(im[~np.isnan(im)], 50)
    print('50% height:', h_50p)
    im_2 = np.copy(im)
    im_2[np.isnan(im_2)] = 0
    [x,y] = np.where(im_2==0)
    for i in range(len(x)):
        kernel = im_2[x[i]-5:x[i]+5, y[i]-5:y[i]+5]
        if np.sum( kernel!=0 ) == 0:
            im_2[x[i], y[i]] = h_50p
        else:
            h_med = np.mean(kernel[kernel!=0])
            im_2[x[i], y[i]] = h_med
    im_2[np.isnan(im_2)] = h_50p
    return im_2, h_50p



def main(args):
    config = args.Config
    configParser = configparser.ConfigParser()   
    configFilePath = config
    # configFilePath= r'D:\GRYFN_SW\GRYFN_Processing\LFX2\config.txt'
    configParser.read(configFilePath)
    try:
        working_folder = configParser.get('lidar-config', 'path1') 
        file_list=os.listdir(working_folder)
    except:
        print('no working folder or incorrect path')
        exit()
    
    try:
        field_name = configParser.get('lidar-config', 'field_name')
        field_panel = configParser.get('lidar-config', 'field_panel')
        file_name_to_save = configParser.get('lidar-config', 'save_name')
    except:
        print('no field/panel name')
        exit()
    
    try:
        LEFT_X = float( configParser.get('lidar-config', 'left_x') )
        TOP_Y = float(  configParser.get('lidar-config', 'up_y') )
        RIGHT_X = float( configParser.get('lidar-config', 'right_x') )
        BOT_Y = float( configParser.get('lidar-config', 'bottom_y') )
    except:
        print('no subarea coordinates')
        exit()
    
    try:
        TRIM = configParser.get('lidar-config', 'trim')
        if TRIM=='true':
            TRIM='True'
    except:
        print('no trim or trim')
        exit()

    try:
        row_in_plot = configParser.get('lidar-config', 'row_in_plot')
        plot_ID=configParser.get('lidar-config', 'plot_ID')
        row_ID=configParser.get('lidar-config', 'row_ID')
        row_ID=ast.literal_eval(row_ID)
    

    except:
        print('Use default setting for plot_ID, row_in_plot, and row_ID')
        plot_ID='plot_ID'
        row_in_plot='row_in_plot'
        row_ID=[2,3]
    
    try:
        save_path = configParser.get('lidar-config', 'path2')
    except:
        print('no saving path')
        exit()
    try:
        geocoordinate=configParser.get('lidar-config', 'geocoordinate')
        # print(type(geocoordinate))
    except:
        geocoordinate='False'
        exit()
    try:
        output_csv=configParser.get('lidar-config', 'output')
        if output_csv=='csv' or 'CSV':
            output_csv=True
            print('save as .csv')
        else:
            print("save as .npy")
    except:
        output_csv=False
        exit()

    if LEFT_X>RIGHT_X:
        print('wrong input x')
        exit()
    if TOP_Y<BOT_Y:
        print('wrong input y')
        exit()
    
    print('loading inputs...')
    #####################################################################
    ###### Load grid and ort files and convert to geo-coordinate ########
    #####################################################################

    if geocoordinate=='False':
        lidar_date = []
        las_name=[]
        grid=None
        for i in file_list:
            if i.endswith('.ort'):
                ort = working_folder+'/'+i
            elif i.endswith('.csv'):
                csv = working_folder+'/'+i
            elif i.endswith('.las'):
                lidar_date.append( i.split('_')[0] )
                las_name.append(i)
        lidar_date.sort()
        
        ext = np.loadtxt(ort) 
        
        grid = pd.read_csv(csv, skiprows = [1], index_col=False) 
        grid = grid.drop(columns = 'Row')
        
        grid.columns = [col.strip() for col in grid.columns]
        
        new_grid = grid.copy()
        new_grid.iloc[:,1] = new_grid.iloc[:,1]/100 + ext[0]  # left x
        new_grid.iloc[:,3]= new_grid.iloc[:,3]/100 + ext[0]  # right x
        new_grid.iloc[:,0] = ext[3] -  new_grid.iloc[:,0]/100 # up y
        new_grid.iloc[:,2] = ext[3] -  new_grid.iloc[:,2]/100 # bottom y

    elif geocoordinate=='True':
        lidar_date = []
        las_name=[]
        for i in file_list:
            if i.endswith('.ort'):
                ort = working_folder+'/'+i
            elif i.endswith('.csv'):
                csv = working_folder+'/'+i
            elif i.endswith('.las'):
                lidar_date.append( i.split('_')[0] )
                las_name.append(i)
        lidar_date.sort()
        
        
        for i in file_list:
            if i.endswith('.json') or i.endswith('.geojson'):
                json_path=working_folder+'/'+i
                print(json_path)
        with open(json_path,'r') as j:
                points=json.load(j)['features']
                # print(points[5]['geometry']['coordinates'][0])
                num_rec=len(points)
        new_grid=dict()
        new_grid['y0']=[]
        new_grid['x0']=[]
        new_grid['y1']=[]
        new_grid['x1']=[]
        new_grid['x2']=[]
        new_grid['x3']=[]
        new_grid['y2']=[]
        new_grid['y3']=[]
        new_grid['plot_ID']=[]
        new_grid['row_in_plot']=[]
        num_rec=len((points))
        for i in range(num_rec):

            new_grid['x0'].append(points[i]['geometry']['coordinates'][0][0][0])
            new_grid['y0'].append(points[i]['geometry']['coordinates'][0][0][1])

            new_grid['x1'].append(points[i]['geometry']['coordinates'][0][1][0])
            new_grid['y1'].append(points[i]['geometry']['coordinates'][0][1][1])

            new_grid['x2'].append(points[i]['geometry']['coordinates'][0][2][0])
            new_grid['y2'].append(points[i]['geometry']['coordinates'][0][2][1])

            new_grid['x3'].append(points[i]['geometry']['coordinates'][0][3][0])
            new_grid['y3'].append(points[i]['geometry']['coordinates'][0][3][1])
    
            new_grid['plot_ID'].append(points[i]['properties'][plot_ID])
            new_grid['row_in_plot'].append(points[i]['properties'][row_in_plot])
        new_grid=pd.DataFrame(new_grid)
        # print(new_grid.head())
        # print(new_grid.shape)
    if TRIM=='True':
        n_p_trim = 0.4 # 10 pixels as hyper with 0.04 m gsd, here equals to 0.4 m
        # trim y or x, depends on the grid, always trim the long side of the grid
        new_grid['y0'] = new_grid['y0'] - n_p_trim
        new_grid['y1'] = new_grid['y1'] + n_p_trim
        print('trimed')
    else:
        print('not trimed')
    
    ##### plot grid for checking ######
    fig = plt.figure( figsize=(10,10))
    ax = fig.add_subplot(111)
    plt.xlim([LEFT_X, RIGHT_X])
    plt.ylim([BOT_Y, TOP_Y])
    for i in range(new_grid.shape[0]):
        rect = patches.Rectangle((new_grid.iloc[i,1],new_grid.iloc[i,0]),(new_grid.iloc[i,3]-new_grid.iloc[i,1]),(new_grid.iloc[i,2] - new_grid.iloc[i,0]),linewidth=0.2,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    ax.autoscale_view()
    plt.show()
    
    print('Continue?')
    user_input = input('[y/n]')
    if user_input == 'n':
        print('exit.')

        exit()
    elif user_input == 'y':
        ############################################################################################
        ###### Creat DEM grids & assign each grid plot number based on it's center position ########
        ############################################################################################
        ### when assigning grids plot number, here is only row 2 and 3
        
        res_extract = 0.16
        
        PLOT_ID = np.unique( new_grid['plot_ID'].values )
        
        width = abs(RIGHT_X - LEFT_X)
        height = abs(TOP_Y - BOT_Y)
        
        num_grid_x = int( np.ceil(width/res_extract))
        num_grid_y = int( np.ceil(height/res_extract))
        print('DSM size: ', num_grid_y, num_grid_x)
        
        dem_x = np.zeros((num_grid_y,num_grid_x))
        dem_y = np.zeros((num_grid_y,num_grid_x))
        for GG in range(len(row_ID)):
            if GG==0:
                indexx=new_grid['row_in_plot'] == row_ID[GG]
            else:
                tempp=new_grid['row_in_plot']==row_ID[GG]
                indexx=np.logical_or(indexx,tempp)


        plot_grid_row23 = new_grid.loc[indexx]
        plot_grid_row23 = plot_grid_row23.reset_index(drop=True)
        for j in range(num_grid_y):
            for k in range(num_grid_x):
                dem_x[j,k] = LEFT_X + k * res_extract + res_extract/2 # dem_x & y are representing the middle position of grid 
                dem_y[j,k] = TOP_Y - j * res_extract + res_extract/2
        
        NG_lidar_avgH = np.zeros((num_grid_y,num_grid_x, 2))
        points = np.vstack((dem_x.flatten(),dem_y.flatten())).T 
        num_plots = len(PLOT_ID)

        for j in range(num_plots):
            tempp=plot_grid_row23[plot_grid_row23['plot_ID']==PLOT_ID[j]] # get the coordinate of a plot


            for k in range((len(tempp['x0']))):
                tupVerts=[(tempp['x0'].iloc[k],tempp['y0'].iloc[k]),  # (1)
                        (tempp['x1'].iloc[k],tempp['y1'].iloc[k]),     # (2)
                        (tempp['x2'].iloc[k],tempp['y2'].iloc[k]), # (3)
                        (tempp['x3'].iloc[k],tempp['y3'].iloc[k])]              
                p = Path(tupVerts)
                if k==0:
                    grid = p.contains_points(points)

                else:
                    tempp_grid=p.contains_points(points)
                    grid=np.logical_or(grid,tempp_grid)

            mask = grid.reshape(num_grid_y,num_grid_x) # now you have a mask with points inside a polygon
            [a,b] = np.where( mask == True)
            NG_lidar_avgH[a,b,1] = plot_grid_row23['plot_ID'][j*(len(row_ID))] 


            
        ####################################################################
        ################# Start feature extraction #########################
        ###################################################################
        First_DTM_h50p = 0
        misalignment={}
        misalignment['date']=lidar_date[1::]
        misalignment['misalgnment']=[]
        for L in range(len(lidar_date)):
            start_time = time.time()
            ####### sub-area of lidar point cloud ######
            heightdate_dtm = lidar_date[L]
            print('lidar date: ', heightdate_dtm)

            outdir = read_point_cloud(heightdate_dtm, working_folder, save_path, field_name, field_panel, [LEFT_X, TOP_Y, RIGHT_X, BOT_Y],
                                      file_name_to_save, las_name[L])
            print('finished save sub area of lidar point cloud.')
            
            ###### cloth simulation filter out ground points ####
            #outdir = save_path+heightdate_dtm+'/'+field_name+'/'+field_panel +'/output/'
        
            inFile = File(outdir+'/'+file_name_to_save, mode='r')
            points = inFile.points
            # use negative z for DEM
            xyz = np.vstack((inFile.x, inFile.y, inFile.z)).transpose() # extract x, y, z and put into a list
            csf = CSF.CSF()
            # prameter settings
            csf.params.bSloopSmooth = False
            csf.params.cloth_resolution = 2
            csf.params.class_threshold = 0.1 # buffer for filter
            csf.params.rigidness = 3 # cloth softness, smaller softer
            ################################################
            #### csf.params.time_step = 0.65;
            #### csf.params.interations = 500;
            #### more details about parameter: http://ramm.bnu.edu.cn/projects/CSF/download/
            ################################################
        
            csf.setPointCloud(xyz)
            ground = CSF.VecInt()  # a list to indicate the index of ground points after calculation
            non_ground = CSF.VecInt() # a list to indicate the index of non-ground points after calculation
            csf.do_filtering(ground, non_ground) # do actual filtering.
            
            outFile = File(outdir + '/' + 'ground_point_res2_buf_01_rig3.las',mode='w', header=inFile.header)
            outFile.points = points[ground] # extract ground points, and save it to a las file.
            outFile.close() # do not forget this
            
            print('finished filtering.')
         
            ##### Based on ground points, creat DEM ####
            dem_lidar = np.zeros((num_grid_y,num_grid_x))
            dem_lidar_allpoints = [[[] for k in range(num_grid_x)] for j in range(num_grid_y)]
            
            # update xyz, use positive z
            inFile = File(outdir + '/' + 'ground_point_res2_buf_01_rig3.las', mode='r')
            G_points = np.vstack((inFile.x, inFile.y, inFile.z)).transpose() # ground points
            for i in range(G_points.shape[0]):
                x_pix = int( round((G_points[i,0] - LEFT_X) / res_extract))
                y_pix = int( round((TOP_Y - G_points[i,1]) / res_extract))
                dem_lidar_allpoints[y_pix - 1][x_pix - 1].append(G_points[i,2])
            
            for j in range(num_grid_y):
                for k in range(num_grid_x):
                    if len(dem_lidar_allpoints[j][k])==0:
                        dem_lidar[j,k] = float('NAN')
                    else:
                        dem_lidar[j,k] = np.percentile(dem_lidar_allpoints[j][k], 50)
        
            bili_dem_lidar,h_50p = bilinear_interpolate(dem_lidar)
        
            ##### SAVE DEM #####
            X = dem_x.flatten()
            Y = dem_y.flatten()
            Z = bili_dem_lidar.flatten()
            
            output_file = File(outdir + '/' + 'DEM.las',mode='w', header=inFile.header)
            output_file.x = X
            output_file.y = Y
            output_file.z = Z
            output_file.close()
            
            print('Saved DEM.')
            
        
            ########## Compare with the first DEM check difference #######
            inFile = File(outdir+'/'+file_name_to_save, mode='r')
            xyz = np.vstack((inFile.x, inFile.y, inFile.z, inFile.intensity)).transpose() # extract x, y, z and put into a list
            
            if L == 0:
                First_DTM_h50p = np.copy( h_50p )
                First_DTM_z = np.copy( Z )
                First_DTM = np.copy( bili_dem_lidar )
            else:
                diff = np.percentile(First_DTM_z - Z, 50)
                if  abs(diff) > 0.04:
                    print('The 50p DTM height is different from with the first date 50p DTM height.', diff)
                    xyz[:,2] = xyz[:,2] + diff
                    misalignment['misalgnment'].append(str(diff))
                else:
                    misalignment['misalgnment'].append('aligned')
                    print('Dont need shift.')
                    
            ######## Calculate relative height of non-ground point in each plot ########
            dataset_sub_NG = xyz[non_ground]  # sub-area non-ground points
            dataset_sub_NG_new, plot_pts,plot_pts_ng, count_bad_pionts = extract_row23_point_cloud(xyz, dataset_sub_NG, \
                                                                                                   dem_x, dem_y,First_DTM,\
                                                                                                   plot_grid_row23,\
                                                                                                   LEFT_X, TOP_Y, res_extract,PLOT_ID)
            
            #### bad points are non-ground points, however, most of them are 
            #### below ground points which return negative relative height 
            print('Number of bad pionts: ', count_bad_pionts)
            
            ##### grids with non-ground points for volumn ####
            NG_lidar_grids = [[[] for k in range(num_grid_x)] for j in range(num_grid_y)]
            for i in range(dataset_sub_NG_new.shape[0]):
                x_pix = int( round((dataset_sub_NG_new[i,0] - LEFT_X) / res_extract))
                y_pix = int( round((TOP_Y - dataset_sub_NG_new[i,1]) / res_extract))
                NG_lidar_grids[y_pix - 1][x_pix - 1].append(dataset_sub_NG_new[i,-1])
            
            # initialize the average height 
            NG_lidar_avgH[:,:,0] = 0
            
            for j in range(num_grid_y):
                for k in range(num_grid_x):
                    ## calculate avg height of each grid in NG_lidar_avgH
                    if len(NG_lidar_grids[j][k])==0:
                        NG_lidar_avgH[j,k, 0] = 0
                    else:
                        NG_lidar_avgH[j,k, 0] = (np.percentile(NG_lidar_grids[j][k], 95) + np.min(NG_lidar_grids[j][k]))/2
            
            print('finished calculating grid based average height with non-ground points')
            
            ####### Calculate features #########
            percents = [30, 50, 70, 90 ,95, 99]
            h_maxv=3.5
            h_minv=0.1
            nhist_bins = 25 # Number of height bins
        
            height_hist_features = np.zeros((len(plot_pts), nhist_bins))
            height_percent_features = np.zeros((len(plot_pts), len(percents)))
            int_percent_features = np.zeros((len(plot_pts), len(percents)))
            height_stat_features = np.zeros((len(plot_pts), 5))
            int_stat_features = np.zeros((len(plot_pts), 5))
            height_hist_features_norm = np.zeros((len(plot_pts), nhist_bins))
            plots_LII = np.zeros((len(plot_pts),7))
            plots_vol = np.zeros((len(plot_pts),1))
            plots_total_n_points = np.zeros((len(plot_pts),1))
            plot_LPI=np.zeros((len(plot_pts),1))
            plot_vci=np.zeros((len(plot_pts),1))
            plot_cap=np.zeros((len(plot_pts),1))
            plot_vol2=np.zeros((len(plot_pts),1))
            plot_std=np.zeros((len(plot_pts),1))



            for i in range(len(PLOT_ID)):
                plot_id = PLOT_ID[i]
                plot_sub = plot_pts_ng[i]
                num_pts_all = len( plot_pts[i] )
                if len(plot_sub) != 0:
                    height_percent_features[i, :] = np.percentile(plot_sub[:,4], percents)
                    int_percent_features[i, :] = np.percentile(plot_sub[:,3], percents)
                    plots_LII[i, :] = LII_v2_extraction(plot_sub[:,4], num_pts_all, np.percentile(plot_sub[:,4], 99))
                    height_stat_features[i, :] = stat_feature_extraction(plot_sub[:,4], 'height')
                    int_stat_features[i, :] = stat_feature_extraction(plot_sub[:,3], 'intensity')
                    plots_total_n_points[i, :] = len(plot_sub[:,4])
                    counts1, edges1 = np.histogram(plot_sub[:,4], range=(h_minv, h_maxv), bins = nhist_bins) 
                    bin_centers = bin_edges_to_centers(edges1)
                    height_hist_features[i, :] = counts1
                    # Normalized hist features
                    height_hist_features_norm[i, :] = np.atleast_2d(height_hist_features[i, :])/\
                                                        np.sum(np.atleast_2d(height_hist_features[i, :]), axis = 1).reshape(-1,1)
                    tmp_plot_grids_avgH = NG_lidar_avgH[np.where(NG_lidar_avgH[:,:,1] == plot_id)][:,0]
                    plots_vol[i, 0] = np.sum(tmp_plot_grids_avgH)/len(tmp_plot_grids_avgH)*res_extract*res_extract
                    #plots_vol[i, 1] = np.sum(tmp_plot_grids_avgH)/(plot_sub.shape[0]*plot_sub.shape[1])
                    plot_LPI[i,0]=len(plot_pts)/num_pts_all
                    plot_vci[i,0]=get_VCI(plot_sub)
                    plot_cap[i,0]=get_CAP(plot_sub)
                    plot_vol2[i,0]=get_vol(plot_sub)
                    plot_std[i,0]=np.std(plot_sub[:,4])
                else:
                    print('empty plot: ',i)

            all_f = {}
            all_f['plots_number'] = PLOT_ID
            all_f['height_hist_features_norm'] = height_hist_features_norm
            all_f['height_percent_features'] = height_percent_features
            all_f['int_percent_features'] = int_percent_features
            all_f['height_stat_features'] = height_stat_features
            all_f['int_stat_features'] = int_stat_features
            all_f['plots_LII'] = plots_LII
            all_f['plots_vol'] = plots_vol
            all_f['plots_total_n_points'] = plots_total_n_points
            all_f['bin_centers'] = bin_centers
            all_f['LPI']=plot_LPI
            all_f['CAP']=plot_cap
            all_f['vci']=plot_vci
            all_f['volh']=plot_vol2
            all_f['height_std']=plot_std
            # for key in all_f.keys():
            #     print(key,all_f[key].shape)
            # break
            name_to_save = os.path.join(outdir,'LiDAR_'+str(lidar_date[L])+'_mean_Hist_features_meanrows_v2_23'+'_'+str(int(100*res_extract))+'cm')
            np.save(name_to_save, all_f)

            if output_csv==True:
                # all_f=pd.DataFrame(all_f)
                # all_f.to_csv(name_to_save)
                csv_header = ['Plot', 'N of points', 
                'height_30p', 'height_50p', 'height_70p', 'height_90p', 'height_95p', 'height_99p', 
                'intensity_30p', 'intensity_50p', 'intensity_70p', 'intensity_90p', 'intensity_95p', 'intensity_99p',
                'canopy_cover_5p', 'pcanopy_cover_10p', 'canopy_cover_20p', 'canopy_cover_30p', 
                'canopy_cover_40p', 'canopy_cover_50p', 'canopy_cover_75p', 
                'standard_deviation_ht', 'quadratic_mean_ht', 'skewness_ht', 'kurtosis_ht', 'variation_ht',
                'standard_deviation_int', 'quadratic_mean_int', 'skewness_int', 'kurtosis_int', 'variation_int',
                "plot_volume","plot_volume2",'VCI','CAP','LPI']

                output = np.hstack((all_f['plots_number'].reshape(-1,1), 
                            np.hstack((all_f['plots_total_n_points'], 
                                np.hstack((all_f['height_percent_features'], 
                                    np.hstack((all_f['int_percent_features'], 
                                        np.hstack((all_f['plots_LII'], 
                                            np.hstack((all_f['height_stat_features'],
                                                np.hstack((all_f['int_stat_features'],
                                                    np.hstack((all_f['plots_vol'],
                                                        np.hstack((all_f['volh'],
                                                            np.hstack((all_f['vci'],
                                                                np.hstack((all_f['CAP'], all_f['LPI']))))))))))))))))))))))
                with open(outdir+'LiDAR_'+str(lidar_date[L])+'mean_Hist_features_meanrows_10trimmed_23_16cm.csv', 'w', newline='') as f:
                    import csv
                    writer = csv.writer(f, delimiter=',')
                    writer.writerow(csv_header)
                    writer.writerows(output)
    




            print('Saved LiDAR feature.')
            end_time = time.time()

            print('total time: %.2f '%(end_time - start_time) )
            print(' ')
        misalignment=pd.DataFrame(misalignment)
        misalignment.to_csv('misalignment_check.csv')
if __name__ == '__main__':
     parser = argparse.ArgumentParser()
     parser.add_argument('--Config', required=False,help = "Input config file path, ex: path/config.txt")
     args = parser.parse_args()
     main(args)
     