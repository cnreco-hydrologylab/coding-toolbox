#!/usr/bin/env python
# coding: utf-8

# In[77]:


import os
import sys
import glob
import warnings
import datetime as dt
import gc
import re
import logging
import json
gc.enable()

fn_log = 'Process_GPP.log'
if os.path.exists(fn_log):
    os.remove(fn_log)
logger = logging.getLogger(__name__)
logging.basicConfig(filename=fn_log, level=logging.INFO)
logging.info('START...')
logging.info('Load modules...')

gc.enable()

import numpy as np
import xarray as xr
import netCDF4 as nc
import geopandas as gpd
from scipy import stats
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from shapely.geometry import box
import rasterio as rst
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import cartopy.crs as ccrs

# -------------------------------------------------------------------------------------
# method to get data settings
def get_data_settings(file_name):
    if os.path.exists(file_name):
        with open(file_name) as file_handle:
            data_settings = json.load(file_handle)
    else:
        logging.error(' ===> Error in reading settings file "' + file_name + '"')
        raise IOError('File not found')
    return data_settings

# -------------------------------------------------------------------------------------

def substitute_keywords(template, **kwargs):
    for key, value in kwargs.items():
        template = template.replace('{' + key + '}', str(value))
    return template

# -------------------------------------------------------------------------------------
# resample and clip a single GeoTIFF
def resample_clip_tif(tif_file, geojson, output_path, target_res_x, target_res_y, opt_save_tif):
    """resamples and clips tif_file on target bounding box and resolution"""
    with rst.open(tif_file) as src:
        # Check CRS and reproject GeoJSON if needed
        if src.crs.to_string() != grid_crs:
            geojson = geojson.to_crs(src.crs)

        # Calculate target transform and dimensions
        transform, width, height = calculate_default_transform(
            src.crs, src.crs,
            src.width, src.height,
            left=lon_min, bottom=lat_min,
            right=lon_max, top=lat_max,
            resolution=(target_res_x, target_res_y)
        )
        profile = src.profile
        profile.update({
            'transform': transform,
            'width': width,
            'height': height,
            'crs': src.crs
        })

        # Create in-memory raster for resampled data
        with rst.MemoryFile() as memfile:
            with memfile.open(**profile) as dst:

                if resampling=='nearest':         resampling_method=Resampling.nearest
                elif resampling=='bilinear':      resampling_method=Resampling.bilinear
                elif resampling=='cubic':         resampling_method=Resampling.cubic
                elif resampling=='cubic_spline':  resampling_method=Resampling.cubic_spline
                elif resampling=='average':       resampling_method=Resampling.average
                elif resampling=='mode':          resampling_method=Resampling.mode
                elif resampling=='max':           resampling_method=Resampling.max
                elif resampling=='min':           resampling_method=Resampling.min
                elif resampling=='med':           resampling_method=Resampling.med
                elif resampling=='sum':           resampling_method=Resampling.sum
                else:
                    raise ValueError('Method not implemented: check function resample_clip_tif')
                    
                for i in range(1, src.count + 1):
                    reproject(
                        source=rst.band(src, i),
                        destination=rst.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=src.crs,
                        # resampling=Resampling.bilinear  # Adjust resampling method as needed
                        resampling=resampling_method
                    )

            # Clip the raster using the bounding box
            with memfile.open() as resampled_src:
                out_image, out_transform = mask(resampled_src, geojson.geometry, crop=True)
                out_meta = resampled_src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })

                # Save clipped GeoTIFF
                if opt_save_tif:
                    with rst.open(output_path, "w", **out_meta) as dest:
                        dest.write(out_image)


# -------------------------------------------------------------------------------------
# plotters
# plot  nc grid
def plot_nc_grid(lat, lon, opt_save_plot):
    plt.figure(figsize=(5,5))
    plt.scatter(lon, lat, c=lat, cmap="viridis", s=1, label="Grid Points")
    plt.colorbar(label="Latitude")
    plt.title("NetCDF Grid")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.legend(); plt.grid(True)
    if opt_save_plot: plt.savefig(
        os.path.join(folder_processed_method, "nc_grid_plot.png"), bbox_inches='tight', dpi=300
    )

# plot geotiff
def plot_geotiff(title, tif_file, opt_save_plot, output_path):
    with rst.open(tif_file) as src:
        data = src.read(1)  # Read the first band
        extent = [
            src.bounds.left,
            src.bounds.right,
            src.bounds.bottom,
            src.bounds.top
        ]

        plt.figure(figsize=(5,5))
        plt.imshow(data, extent=extent, origin="upper", cmap="viridis")
        plt.colorbar(label="Value")
        plt.xlabel("Longitude"); plt.ylabel("Latitude")
        plt.grid(True)
        if opt_save_plot: plt.savefig(
            output_path, bbox_inches='tight', dpi=300
        )

# ------------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------------#


logging.info('START...')

# Parse data settings
file_settings        = 'configuration.json'
data_settings        = get_data_settings(file_settings)

# Function to recursively assign variables from nested dictionaries
def assign_variables(config_dict, prefix=''):
    var_dict = {}
    """Warning: this function works on the global() namespace of the module in which it is defined"""
    for key, value in config_dict.items():
        if isinstance(value, dict):
            # For nested dictionaries, add the current key to the prefix and recursively call the function
            assign_variables(value, prefix=prefix)
        else:
            # Assign the value to a global variable with the constructed name
            name_var = key
            if value=='True': value=True
            elif value=='False': value=False
            globals().update({name_var:value})
            del name_var

# Call the function with the parsed configuration
assign_variables(data_settings)

folder_products = os.path.join(root, folder_products)
folder_processed = os.path.join(root, folder_processed)
folder_processed_method = os.path.join(folder_processed, resampling)

if not os.path.exists(folder_processed):
    os.mkdir(folder_processed)
if not os.path.exists(folder_processed_method):
    os.mkdir(folder_processed_method)

products_list = glob.glob(folder_products+filename_template)
filename_processed_list = [f"{tif.split('/')[-1].replace('.tif', filename_processed_add)}" for tif in products_list]
processed_list = [os.path.join(folder_processed_method, proc_tif) for proc_tif in filename_processed_list]

# open nc file for grid
if grid_nc_path=='':
    raise FileNotFoundError('Provide a nc file for reference grid.')

ds = xr.open_dataset(grid_nc_path, decode_coords="all")
lat = ds[lat_variable].values
if lat[-1][0] > lat[0][0]: lat = np.flip(lat)
lon = ds[lon_variable].values

# get bounding box
lon_min, lon_max = np.min(lon), np.max(lon)
lat_min, lat_max = np.min(lat), np.max(lat)
bounding_box = box(lon_min, lat_min, lon_max, lat_max)
target_res_x = abs(lon[0, 1] - lon[0, 0])
target_res_y = abs(lat[1, 0] - lat[0, 0])
geojson = gpd.GeoDataFrame({'geometry': [bounding_box]}, crs=grid_crs).to_crs(grid_crs)

for tif, out_tif in zip(products_list, processed_list):
    resample_clip_tif(tif, geojson, out_tif, target_res_x, target_res_y, opt_save_tif)

# plotters
# plot nc grid
plot_nc_grid(lat, lon, opt_save_plot)
# plot processed tif
sample_output_file = f"{folder_processed_method}/{products_list[0].split('/')[-1].replace('.tif', filename_processed_add)}"
title = 'clipped geotiff'
plot_geotiff(title, sample_output_file, opt_save_plot, output_path=os.path.join(folder_processed_method, "tif_clipped_plot.png"))
# plot original tif
sample_output_file = products_list[0]
title = 'original geotiff'
plot_geotiff(title, sample_output_file, opt_save_plot, output_path=os.path.join(folder_products, "tif_original_plot.png"))


# In[ ]:




