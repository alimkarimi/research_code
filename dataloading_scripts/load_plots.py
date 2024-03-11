import numpy as np
import os
import pandas as pd
import geopandas as gpd
import re
import json
from osgeo import gdal, osr

plot_path_root = '/Users/alim/Documents/prototyping/research_lab/HIPS_grid/'
plot_path_2022_hips = '20220609_f78_hips_manshrink.geojson'

plot_path_2021_hips = '20210617_india_f42mYS_HIPS_1cm_noshrinkWei.json'


def load_plots_coords_for_field(path = None, field='hips_2022', img_coords=False, geo_coords=True):
    """
    The HIPS 2022 data is in a txt file, with image x coordinates in the 20,989 - 22,435 and y coordinates in the
    2,000 - 10,000 range. However, the hyperspectral images themselves when they are read are have 2526 x 2402 pixels. 

    The HIPS 2021 data is in a txt file as well, but in terms of NAD83. 

    
    """
    if img_coords:
        if field == 'hips_2022':
            
            path = plot_path_root + plot_path_2022_hips
            df = pd.read_csv(path)
            df = df.drop(0) 
            df = df.reset_index(drop=True) # reset indexing for the dropped rows. We do not want to change in the loop above
            # as that can cause issues with going through the range/shape in the for i in range(df.shape[0]) command.
            print(df.describe())

            return df # df has columns y0, x0, y1, x1 with a plot_ID and row_in_plot that can help with extracting plot level data
            # from an image.

        if field == 'hips_2021':
            path = plot_path_root + plot_path_2021_hips
            df = pd.read_csv(path, sep='\t')
            print(df)
            print(df.shape)
            print(df.describe())
            return df

    if geo_coords:
        if field == 'hips_2021':
            # load json for 2021
            # Specify the path to the json file. Keep the base root as plot_path_root
            file_path = plot_path_root + '20210617_india_f42mYS_HIPS_1cm_noshrinkWei.json'

            # Open the JSON file and load its contents
            with open(file_path, "r") as file:
                data = json.load(file)

            return data

        if field == 'hips_2022':
            file_path = plot_path_root + '20220609_f78_hips_manshrink.geojson'

            with open(file_path, "r") as file:
                data = json.load(file)

            return data # returns json

def load_individual_plot_xyxy(plot_json, index, field):
    """
    This function takes in a plot json object, an index, and returns the x0, x1, y0, and y1 based plot boundary and index. 
    """
    if field == 'hips_2021':
        xyxy = plot_json['features'][index]['properties']
        #print(xyxy)
        x0 = xyxy['x0']
        y0 = xyxy['y0']
        x1 = xyxy['x1']
        y1 = xyxy['y1']
        plot_id = xyxy['plot_ID']
        plot_row = xyxy['row_in_plot']
        return x0, y0, x1, y1, plot_id, plot_row
    
    if field == 'hips_2022':
        x0y0 = plot_json['features'][index]['geometry']['coordinates'][0][0]
        #coordinates_1 = plot_json['features'][index]['geometry']['coordinates'][0][1]
        x1y1 = plot_json['features'][index]['geometry']['coordinates'][0][2]
        #coordinates_3 = plot_json['features'][index]['geometry']['coordinates'][0][3]
        x0, y0 = x0y0
        x1, y1 = x1y1
        plot_id = plot_json['features'][index]['properties']['plot']
        plot_row = plot_json['features'][index]['properties']['row']
        #print(plot_id, plot_row)
        #print(x0, y0, x1, y1)
        return x0, y0, x1, y1, plot_id, plot_row

def load_entire_plot_xyxy(plot_json, plot_id_query, field):
    # get all rows that belong to a plot.
    """
    Gets the convex hull of both rows
    """

    matches = []

    for i in range(100,277):
        out = load_individual_plot_xyxy(plot_json, i, field)
        x0, y0, x1, y1, plot_id, plot_row = out
        if plot_id_query == plot_id:
            matches.append(out)

    #print(matches[0]) # western plot
    #print(matches[1]) # eastern plot
    west_x0, west_y0, a, b, c, d = matches[0] # a, b, c, d are dummy vars
    east_x1, east_y1, e, f, g, h = matches[1] # e, f, g, h are dummy vars

    return west_x0, west_y0, east_x1, east_y1, plot_id_query
            


if __name__ == "__main__":
    field = 'hips_2021'
    data = load_plots_coords_for_field(field=field, geo_coords=True)
    #print(data)
    print(len(data['features']))
    num_feats = len(data['features'])
    x0, y0, x1, y1, _, __= load_individual_plot_xyxy(data, 2, field = field)
    for i in range(0,277):
        out = load_individual_plot_xyxy(data, i, field= field)
        print(out)

    matches = load_entire_plot_xyxy(data, plot_id_query = 4400, field = 'hips_2021')
    print(matches)
