from pickle import load
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import rasterio

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.data.export_gtiff import calculate_affine,\
                                  get_ICgdf_dimensions,\
                                  calculate_affine_coeffs
                                  
                                  
base_dir = "C:\\Users\\Chris\\co_cloudcover\\"
model_dir = base_dir + "models\\"
interim_dir = base_dir + "data\\interim\\"
fig_dir = base_dir + "reports\\figures\\"
raw_dir = "D:\\proj\\co_cloudcover\\raw_data\\"

features = [
    "sur_refl_b01",
    "sur_refl_b02",
    "sur_refl_b03",
    "sur_refl_b04",
    "sur_refl_b05",
    "sur_refl_b06",
    "sur_refl_b07",
]

def load_model_components(rfc_name, pca_name, scaler_name):
    
    with open(model_dir + "rfc_20240529.pkl", "rb") as f:
        rfc = load(f)
    with open(model_dir + "pca_20240529.pkl", "rb") as f:
        pca = load(f)
    with open(model_dir + "scaler_20240529.pkl", "rb") as f:
        scaler = load(f)
        
    return(rfc, pca, scaler)
    
    
    
    
def load_data_as_gdf(IC_parquet):
    print(IC_parquet)
    IC = gpd.read_parquet(IC_parquet)
    return(IC)



def prepare_gdf(IC):
    IC['date'] = IC['time'].dt.strftime("%Y-%m-%d")
    IC = IC.dropna().reset_index(drop=True)
    return(IC)



def run_model(IC,scaler,pca,rfc):
    IC_scaled=scaler.transform(IC[features])#.reset_index(drop=True)
    IC_pca=pca.transform(IC_scaled)
    IC["prediction"] = rfc.predict(IC_pca)
    return(IC)



def export_monthly_results(model_results,fig_dir,interim_dir):
    for result in model_results:
        name,days,clouds = result['name'], result['day_count'], result['cloud_count']
        name = name.split('.')[0]
        print(name)
        clouds = gpd.GeoDataFrame(clouds)
        
        fig, ax = plt.subplots()
        clouds.plot(column="prediction",legend=True, figsize = (12,10), markersize=.2, ax=ax, vmin=0, vmax=30)
        fig.set_size_inches(12,10)
        plt.savefig(fig_dir + f"monthly\\cloudCount_{name}.jpeg")
        del(fig)
    
        clouds.to_parquet(path=interim_dir + f"cloudCount_{name}" + ".parquet")



def export_yearly_results(model_results,fig_dir,interim_dir):
    clouds_list = []
    for result in model_results:
        name,days,clouds = result['name'], result['day_count'], result['cloud_count']
        clouds_list.append(clouds)
    
    clouds_yearly = pd.concat(clouds_list).\
            groupby(['geometry']).\
            agg("sum").\
            reset_index()\
            [["prediction","geometry"]]
    
    fig, ax = plt.subplots()
    clouds = gpd.GeoDataFrame(clouds_yearly)
    clouds["365"] = 365
    clouds["DoS"] = clouds["365"] - clouds["prediction"]
    
    clouds.plot(column="DoS",legend=True, figsize = (12,10), markersize=.2, ax=ax, vmin=200, vmax=350)
    fig.set_size_inches(12,10)
    plt.savefig(fig_dir + f"monthly\\DaysOfSun_{name[0:4]}.jpeg")
    del(fig)
    
    clouds.to_parquet(path=interim_dir + f"yearly_cloudCount_{name[0:4]}" + ".parquet")
    
    clouds['x'],clouds['y'] = clouds.geometry.x,clouds.geometry.y
    clouds['lon'],clouds['lat'] = clouds.geometry.x,clouds.geometry.y
    clouds.sort_values(['x', 'y'], ascending=[True, True],inplace=True)
    
    dim_x, dim_y = get_ICgdf_dimensions(clouds)
    transform_coeffs = calculate_affine_coeffs(clouds,dim_x,dim_y)
    band_data = clouds['prediction'].to_numpy().reshape((dim_x,dim_y))
    cloud_profile = {
            'driver':'GTiff',
            "width": dim_x,
            "height": dim_y,
            "count": 1,
            "dtype": band_data.dtype,
            "crs": "EPSG:4326",
            "transform": calculate_affine(transform_coeffs),
            "nodata": 0,
            "compress":'lzw',
            }
    with rasterio.open(fig_dir + f"yearly\\yearly_cloudCount_{name[0:4]}.tif", 'w', **cloud_profile) as dst:
            dst.write(band_data,indexes=1)
    print("Done")
    
    
    
    
    
    
    
    