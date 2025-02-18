import os
import gc
import sys
import json
import math
import glob
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc
import matplotlib.pyplot as plt

def read_nc_basic(filename):
    "reads an nc file. to be used after check_nc_integrity."
    import xarray as xr
    try:
        ds = xr.open_dataset(filename)
        return ds
    except (KeyError, TypeError, OSError, ValueError):
        pass
    try:
        import netCDF4 as nc
        ds = nc.Dataset(filename)
        return ds
    except (OSError):
        print(f'File {filename} cannot be read and may be corrupted. Skipping...')
        pass
        
# read nc file
ds = read_nc_basic(fn_nc)
        
import geopandas as gpd
from shapely.geometry import mapping

gdf = gpd.read_file(fn_shapefile)
geometries = gdf.geometry.apply(mapping)
# if you have a bounding box
min_x, min_y, max_x, max_y = gdf.bounds.values.squeeze()

import rioxarray

# clip on a rectangular bounding box
ds = ds.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
ds = ds.rio.write_crs('epsg:4326')
ds_clipped = ds.rio.clip_box(
    min_x,min_y,max_x,max_y,
    crs='epsg:4326'
)
# if you have a generic polygon use function clip
