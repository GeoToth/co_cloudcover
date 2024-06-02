# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:21:15 2024

@author: Chris
"""

import math
import numpy as np
import rasterio
from rasterio.transform import Affine



def calculate_affine(coeffs):
    """Calculate Affine transformation for rasterio write
    
    Args described here:
    https://trac.osgeo.org/postgis/wiki/DevWikiAffineParameters"""
    sx,sy,tx,ty = coeffs['sx'],coeffs['sy'],coeffs['tx'],coeffs['ty']
    θ,kx,ky = coeffs['θ'],coeffs['kx'],coeffs['ky']
    
    o11 = sx * ((1 + kx*ky) * math.cos(θ) + ky*math.sin(θ) )
    o12 = sx * ( kx*math.cos(θ) + math.sin(θ) )
    o21 = -sy * ( -(1 + kx*ky) * math.sin(θ) + ky*math.cos(θ) )
    o22 = sy * ( -kx*math.sin(θ) + math.cos(θ) )

    return(Affine(o11,o12,tx,o21,o22,ty))

def rbgArray_to_RGBgeotiff(rgb,f_name,directory,profile):
    """Writes 3 dimensional array as 3-band tiff
    
    Arguments
    rgb       : 3xMxN ndarray of channels
    f_name    : name of image
    directory : location to export image
    profile   : parameters for rasterio write
    
    Returns
    None
    """
    with rasterio.open(directory + f_name,'w',**profile) as dst:
        dst.write(rgb)

    return(None)
    

def make_RGBarray(gdf,img_date,rgb_bands):
    """constructs 3-band RGB arrays for selected date in gdf

    Arguments
    gdf       : ImageCollection as GeoDataFrame
    img_date  : date to extract from gdf
    rgb_bands : dict of bands to use for RGB image

    Returns
    rgb   : 3xMxN ndarray of channels
    dim_x : len of x axis
    dim_y : len of y axis
    """
    
    #select data by date:
    date_gdf = gdf[gdf['time']==img_date]
    
    # Select Bands:
    r_band = rgb_bands["red"]
    g_band = rgb_bands["green"]
    b_band = rgb_bands["blue"]
    rgb_gdf = date_gdf[[r_band,g_band,b_band,"geometry"]]
    
    # Extract Geometry:
    rgb_gdf['x'],rgb_gdf['y'] = rgb_gdf.geometry.x,rgb_gdf.geometry.y
    dim_x,dim_y = get_ICgdf_dimensions(gdf)

    # Construct channel arrays
    r_array = rgb_gdf[r_band].to_numpy().reshape((dim_x,dim_y))
    g_array = rgb_gdf[g_band].to_numpy().reshape((dim_x,dim_y))
    b_array = rgb_gdf[b_band].to_numpy().reshape((dim_x,dim_y))

    #Construct RGB Array
    rgb = np.stack((r_array,g_array,b_array))
    
    return(rgb,dim_x,dim_y)



def get_ICgdf_dimensions(gdf):
    """ Returns the number of samples along the x and y axes"""
    
    dim_x,dim_y = len(set(gdf.x)),len(set(gdf.y))
    
    return(dim_x,dim_y)    



def calculate_affine_coeffs(IC_gdf,dim_x,dim_y):
    """
    

    Returns
    -------
    None.

    """
    transform_coeffs = {
        'sx':abs((min(IC_gdf['lon']) - max(IC_gdf['lon']))/dim_y)  ,# scale factor in x direction
        'sy':abs((min(IC_gdf['lat']) - max(IC_gdf['lat']))/dim_x)  ,# scale factor in y direction
        'tx':min(IC_gdf['lon'])  ,# offset in x direction
        'ty':min(IC_gdf['lat'])  ,# offset in y direction
        'θ':math.pi/2   ,# angle of rotation clockwise around origin
        'kx':0  ,# shearing parallel to x axis
        'ky':0  ,# shearing parallel to y axis 
        }

    return(transform_coeffs)