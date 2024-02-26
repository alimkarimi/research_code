import numpy as np
import os
import pandas as pd
import geopandas as gpd
import re
import json
from osgeo import gdal, osr

plot_path_root = '/Users/alim/Documents/prototyping/research_lab/HIPS_grid/'
plot_path_2022_hips = 'hips_2022_plot_extraction_Alim_modified_on_gsheets_20220609_f78_hips_manshrink.csv'
plot_path_2021_hips = '20210617_india_f42mYS_HIPS_1cm_noshrinkWei.txt'


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


            # Now you can work with the loaded JSON data, for example:
            #print(data)

            """
            To get a sense of what the data looks like:
            {'type': 'FeatureCollection', 'crs': {'type': 'name', 'properties': {'name': 'EPSG:26916'}}, 
            'features': [{'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': 
            [[[500164.0, 4480666.85], [500164.78, 4480666.85], [500164.78, 4480672.27], [500164.0, 4480672.27]]]},
             'properties': {'x0': 500164.0, 'y0': 4480666.85, 'x1': 500164.78, 'y1': 4480672.27, 'plot_ID': 4301, 'row_in_plot': 1}},. 

             Note that the spatial reference for the hyperspectral data looks like:
                    spatial info: PROJCS["unnamed",
            GEOGCS["NAD83",
                DATUM["North_American_Datum_1983",
                    SPHEROID["GRS 1980",6378137,298.257222101,
                        AUTHORITY["EPSG","7019"]],
                    AUTHORITY["EPSG","6269"]],
                PRIMEM["Greenwich",0,
                    AUTHORITY["EPSG","8901"]],
                UNIT["degree",0.0174532925199433,
                    AUTHORITY["EPSG","9122"]],
                AUTHORITY["EPSG","4269"]],
            PROJECTION["Transverse_Mercator"],
            PARAMETER["latitude_of_origin",0],
            PARAMETER["central_meridian",-87],
            PARAMETER["scale_factor",0.9996],
            PARAMETER["false_easting",500000],
            PARAMETER["false_northing",0],
            UNIT["Meter",1],
            AXIS["Easting",EAST],
            AXIS["Northing",NORTH]]
            """

            # print(data['type'])
            # print(data['crs']['properties']['name'])
            # print(data['features'][0]['geometry']['type'])
            #print(data['features'][0]['geometry']['coordinates'][0])
            #print(data['features'][0]['properties'])
            #return data['features'][0]['properties'], data['features'][0]['geometry']['coordinates'][0]
            return data # returns json

def load_individual_plot_xyxy(plot_json, index):
    """
    This function takes in a plot json object, an index, and returns the x0, x1, y0, and y1 based plot boundary and index. 
    """
    xyxy = plot_json['features'][index]['properties']
    #print(xyxy)
    x0 = xyxy['x0']
    y0 = xyxy['y0']
    x1 = xyxy['x1']
    y1 = xyxy['y1']
    plot_id = xyxy['plot_ID']
    plot_row = xyxy['row_in_plot']
    return x0, y0, x1, y1, plot_id, plot_row


if __name__ == "__main__":

    data = load_plots_coords_for_field(field='hips_2021', geo_coords=True)
    print(data)
    print(len(data['features']))
    num_feats = len(data['features'])
    x0, y0, x1, y1, _, __= load_individual_plot_xyxy(data, 2)
    for i in range(num_feats):
        print(load_individual_plot_xyxy(data, i))