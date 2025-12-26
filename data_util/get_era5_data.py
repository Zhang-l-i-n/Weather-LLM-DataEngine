import xarray as xr 
import os 

def get_era5_data_by_vars(file_path, var_name): 
    if not os.path.exists(file_path): 
        print(f"Error: The file at '{file_path}' was not found.") 
        return None 
 
    try: 
        print(f"Reading GRIB file: {file_path}") 
        ds = xr.open_dataset(file_path, engine='cfgrib') 
        
        if var_name in ds.data_vars: 
            ds_value = ds[[var_name]] 
            print(f"\n--- Dataset for variable: {var_name} ---") 
            print(ds_value) 
            return ds_value 
        else: 
            ds_value = ds_value = xr.open_dataset(
                file_path, 
                engine='cfgrib', 
                backend_kwargs={'filter_by_keys': {'shortName': var_name}})
            print(f"\n--- Dataset for variable: {var_name} ---") 
            print(ds_value) 
            return ds_value
 
    except Exception as e: 
        print(f"An error occurred while trying to read the GRIB file: {e}") 
        print("Please ensure 'cfgrib', 'xarray', and 'eccodes' are correctly installed.") 
        return None  
