import numpy as np
import torch
from datetime import datetime, timedelta
import math

# Mapping from original class values to indices for LULC and cloud mask
value_to_index_mapping = {
    'LULC': {
        0: 0,  # 0: Black        | No data
        1: 1,  # 1: Blue         | Water
        2: 2,  # 2: Green        | Trees
        4: 3,  # 3: Light green  | Flooded vegetation
        5: 4,  # 4: Yellow       | Crops
        7: 5,  # 5: Red          | Built-up areas
        8: 6,  # 6: Light yellow | Bare Ground
        9: 7,  # 7: Light blue   | Snow/Ice
        10: 8, # 8: Dark gray    | Clouds
        11: 9  # 9: Brown        | Rangeland
    },
    # Cloud mask values: [0, 1, 2, 3] for Clear, Thick, Thin, Shadow
    'cloud_mask': {i: i for i in range(4)},
}

# Mapping from original class values to names for LULC and cloud mask
name_list = {
    'LULC': [
        'None/NoData',
        'Water',
        'Trees',
        'Flooded Vegetation',
        'Crops',
        'Built-up Areas',
        'Bare Ground',
        'Snow/Ice',
        'Clouds',
        'Rangeland',
    ],
    'cloud_mask': [
        'Clear',
        'Thick',
        'Thin',
        'Shadow',
    ],
}


def get_value_to_index(satellite_type):
    if satellite_type == 'LULC_LULC' or satellite_type == 'LULC':
        return value_to_index_mapping['LULC']
    elif satellite_type == 'S2L1C_cloud_mask' or satellite_type == 'cloud_mask':
        return value_to_index_mapping['cloud_mask']
    else:
        raise ValueError(f"Satellite type {satellite_type} does not have a value to index mapping.")

def is_leap_year(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

def decode_single_date(sin_doy, cos_doy, year_norm, min_year=1950, max_year=2050):
    """
    This function is used in the pre_post_process_data.py file to decode the encoded date back to a datetime object.
    """
    # Convert torch tensors to scalars if needed
    if torch.is_tensor(sin_doy):
        sin_doy = sin_doy.item()
    if torch.is_tensor(cos_doy):
        cos_doy = cos_doy.item()
    if torch.is_tensor(year_norm):
        year_norm = year_norm.item()
    
    # Decode year
    year = ((year_norm + 1) / 2) * (max_year - min_year) + min_year
    year = int(round(year))

    days_in_year = 366 if is_leap_year(year) else 365

    # Decode day of year
    angle = np.arctan2(sin_doy, cos_doy)
    if angle < 0:
        angle += 2 * np.pi

    doy = int(round(angle * days_in_year / (2 * np.pi))) % days_in_year
    if doy == 0:
        doy = days_in_year
    decoded_date = datetime(year, 1, 1) + timedelta(days=doy - 1)
    return [decoded_date.day, decoded_date.month, decoded_date.year]

def decode_date(data):
    B, _, _, C = data.shape
    data = data.reshape(-1, C)
    decoded_dates = [decode_single_date(data[i, 0], data[i, 1], data[i, 2]) for i in range(len(data))]
    decoded_dates = torch.tensor(decoded_dates)  # Convert list of lists to tensor
    return decoded_dates.reshape(B, 1, 1, 3)

def encode_date(date, min_year=1950, max_year=2050, add_spatial_dims=True):
    """
    This function is used in the pre_post_process_data.py file to encode the date to a 3D vector.
    """
    doy = date.timetuple().tm_yday
    year = date.year
    days_in_year = 366 if is_leap_year(year) else 365

    sin_doy = np.sin(2 * np.pi * doy / days_in_year)
    cos_doy = np.cos(2 * np.pi * doy / days_in_year)
    year_norm = 2 * ((year - min_year) / (max_year - min_year)) - 1
    date_vec = np.array([sin_doy, cos_doy, year_norm], dtype=np.float32)
    if add_spatial_dims:
        return date_vec.reshape(1, 1, 1, 3)
    else:
        return date_vec

def encode_lat_lon(lat, lon, output_format, add_spatial_dims=True):
    if output_format == "3d_cartesian_lat_lon":
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        x = math.cos(lat_rad) * math.cos(lon_rad)
        y = math.cos(lat_rad) * math.sin(lon_rad)
        z = math.sin(lat_rad)
        xyz = np.array([x, y, z], dtype=np.float32)
        if add_spatial_dims:
            return xyz.reshape(1, 1, 1, 3)
        else:
            return xyz
    else:
        raise ValueError(f"ERROR: Invalid output format: {output_format}")


def decode_lat_lon(data, output_format):
    # Convert from 3D cartesian coordinates back to lat/lon in radians
    # Input shape is (B, 1, 1, 3) for x,y,z coordinates
    # Reshape to (B*H*W, 3)
    B, _, _, C = data.shape
    data = data.reshape(-1, C)
    if output_format == "3d_cartesian_lat_lon":
        # Extract x,y,z coordinates
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]

        # Convert to lat/lon using arcsin and atan2
        if (z.abs() > 1).any():
            print(f"WARNING: z is out of bounds: {z}. Clipping to [-1,1]")
            print(f"x: {x}")
            print(f"y: {y}")
            z = torch.clamp(z, -1, 1)
        lat = torch.arcsin(z)  # latitude in radians
        lon = torch.atan2(y, x) # longitude in radians
        
        # Convert from radians to degrees
        lat = lat * 180.0 / np.pi
        lon = lon * 180.0 / np.pi
        # Stack lat/lon and reshape back to original dimensions
        return torch.stack([lat, lon], dim=-1).reshape(B, 1, 1, 2)
    else:
        raise ValueError(f"ERROR: Invalid output format: {output_format}")

def one_hot_encode(data, value_to_index):
    num_classes = len(value_to_index.keys())
    data = data.squeeze(0)
    # Map original values to indices
    mapped_data = np.zeros_like(data, dtype=np.int64)
    for orig_val, idx in value_to_index.items():
        mapped_data[data == orig_val] = idx
    # Create one-hot encoding
    one_hot = np.zeros((num_classes, *data.shape), dtype=np.float32)
    for i in range(num_classes):
        one_hot[i] = (mapped_data == i).astype(np.float32)
        
    return one_hot

def one_hot_decode(data, value_to_index):
    one_hot_indices = torch.argmax(data, dim=1).unsqueeze(1)
    # Convert one-hot ordered indices to original class values
    mapped_data = torch.zeros_like(one_hot_indices, dtype=torch.int64)
    for orig_val, one_hot_idx in value_to_index.items():
        mapped_data[one_hot_indices == one_hot_idx] = torch.tensor(orig_val)
    return mapped_data

def pre_process_data(band_data, satellite_type, data_config, already_encoded=False, in_db=False):
    """Apply satellite-specific preprocessing to band data"""
    
    # If data_config has key "class_name" and it is "SatelliteDataset"
    if data_config.get('class_name', None) == 'SatelliteDataset':
        normalize_to_neg_one_to_one = data_config['normalize_to_neg_one_to_one']
        min_db = data_config['min_db']
        max_db = data_config['max_db']
        min_positive = data_config['min_positive']
    else:
        # Get the data configuration
        normalize_to_neg_one_to_one = data_config.get('normalize_to_neg_one_to_one', False)
        min_db = data_config.get('min_db', None)
        min_db = min_db[satellite_type] if min_db is not None else None
        max_db = data_config.get('max_db', None)
        max_db = max_db[satellite_type] if max_db is not None else None
        min_positive = data_config.get('min_positive', None)
        min_positive = min_positive[satellite_type] if min_positive is not None else None
    
    # Support both NumPy arrays and torch tensors
    is_torch = torch.is_tensor(band_data)
    if is_torch:
        bd_dtype = band_data.dtype
        bd_device = band_data.device
        min_db_t = torch.tensor(min_db, dtype=bd_dtype, device=bd_device) if min_db is not None else None
        max_db_t = torch.tensor(max_db, dtype=bd_dtype, device=bd_device) if max_db is not None else None
        min_positive_t = torch.tensor(min_positive, dtype=bd_dtype, device=bd_device) if min_positive is not None else None
    else:
        min_db_t = min_db
        max_db_t = max_db
        min_positive_t = min_positive
    
    # if 'cloud_mask' in satellite_type:
    #     # Cloud mask is in the range [0, 3] -> Convert to [0, 1] range
    #     data = band_data.astype(np.float32) / 3.0
    #     # Scale to [-1,1] range for the model if needed
    #     if normalize_to_neg_one_to_one:
    #         data = data * 2.0 - 1.0
    #     return data
    # elif 'LULC' in satellite_type:
    #     # LULC is in the range [0, 11] -> Convert to [0, 1] range
    #     data = band_data.astype(np.float32) / 11.0
    #     # Scale to [-1,1] range for the model if needed
    #     if normalize_to_neg_one_to_one:
    #         data = data * 2.0 - 1.0
    #     return data
    
    # Cloud mask needs to be checked first to avoid confusion with S2L1C_cloud_mask and other modalities with S2L1C prefix (e.g. S2L1C_B02_B03_B04_B08)
    if 'cloud_mask' in satellite_type or 'LULC' in satellite_type:
        band_data = one_hot_encode(band_data, value_to_index=get_value_to_index(satellite_type))
        if is_torch:
            band_data = torch.from_numpy(band_data).to(bd_device)
        if normalize_to_neg_one_to_one:
            band_data = band_data * 2.0 - 1.0
        return band_data
    elif '3d_cartesian_lat_lon' in satellite_type:
        if already_encoded:
            xyz_np = band_data
        else:
            lat_val = float(band_data[0].item() if is_torch else band_data[0])
            lon_val = float(band_data[1].item() if is_torch else band_data[1])
            xyz = encode_lat_lon(lat_val, lon_val, output_format='3d_cartesian_lat_lon')
            xyz_np = np.array(xyz, dtype=np.float32)
        if is_torch:
            xyz_np = torch.from_numpy(xyz_np).to(bd_device)
        return xyz_np
    elif 'mean_timestamps' in satellite_type:
        if already_encoded:
            enc = band_data
        else:
            ts = float(band_data.item() if is_torch else band_data)
            date = datetime.fromtimestamp(ts)
            enc = encode_date(date)
        if is_torch:
            enc = torch.from_numpy(enc).to(bd_device)
        return enc
    elif 'S2L2A' in satellite_type or 'S2L1C' in satellite_type:
        # Sentinel-2 data needs to be divided by 10,000
        # This ensures proper scaling as Sentinel-2 reflectance values
        data = (band_data.to(torch.float32) if is_torch else band_data.astype(np.float32)) / 10000.0
        # Scale to [-1,1] range for the model if needed
        if normalize_to_neg_one_to_one:
            data = data * 2.0 - 1.0
        return data
    elif 'S1RTC' in satellite_type:
        band_data = band_data.to(torch.float32) if is_torch else band_data.astype(np.float32)
        if in_db:
            # If input data is in db range, convert to backscatter values
            band_data = 10 ** (band_data / 10.0)
        # STEP 1: Replace nodata and negative values with the minimum positive value
        if min_positive_t is None:
            raise ValueError(f"ERROR: No min_positive value defined for {satellite_type}. Please provide a min_positive value.")
        if is_torch:
            band_data = torch.where(band_data <= 0, min_positive_t, band_data)
        else:
            band_data = np.where(band_data <= 0, min_positive_t, band_data)
        # STEP 2: Apply log scaling
        band_data = 10 * (torch.log10(band_data) if is_torch else np.log10(band_data))
        # STEP 3: Use defined min/max values for normalization
        if (min_db_t is not None) and (max_db_t is not None):
            band_data = (band_data - min_db_t) / (max_db_t - min_db_t)
            # Scale to [-1,1] range for the model if needed
            if normalize_to_neg_one_to_one:
                band_data = band_data * 2.0 - 1.0
        else:
            print(f"WARNING: No min/max values defined for {satellite_type}. No normalization will be applied.")
        return band_data
    elif 'DEM' in satellite_type:
        band_data = band_data.to(torch.float32) if is_torch else band_data.astype(np.float32)
        # STEP 1: Replace nodata and negative values with the minimum positive value
        if min_positive_t is None:
            raise ValueError(f"ERROR: No min_positive value defined for {satellite_type}. Please provide a min_positive value.")
        if is_torch:
            band_data = torch.where(band_data <= 0, min_positive_t, band_data)
        else:
            band_data = np.where(band_data <= 0, min_positive_t, band_data)
        # STEP 2: Apply log scaling
        band_data = torch.log1p(band_data) if is_torch else np.log1p(band_data)
        # band_data = 10 * np.log10(band_data)
        # STEP 3: Use defined min/max values for normalization
        if (min_db_t is not None) and (max_db_t is not None):
            band_data = (band_data - min_db_t) / (max_db_t - min_db_t)
            # Scale to [-1,1] range for the model if needed
            if normalize_to_neg_one_to_one:
                band_data = band_data * 2.0 - 1.0
        else:
            print(f"WARNING: No min/max values defined for {satellite_type}. No normalization will be applied.")
        # print(f"INPUT. band_data min: {np.min(band_data)}, max: {np.max(band_data)}")
        return band_data
    else:
        raise ValueError(f"ERROR: Pre-processing not implemented for satellite type: {satellite_type}")
        # Unknown type - just normalize the data
        data = band_data.astype(np.float32)
        # Estimate min/max from the data
        min_val = np.min(data)
        max_val = np.max(data)
        if min_val == max_val:
            data = np.zeros_like(data)
        else:
            data = (data - min_val) / (max_val - min_val)
            if normalize_to_neg_one_to_one:
                data = data * 2.0 - 1.0
        return data

def post_process_data(processed_data, satellite_type, data_config):
    """Reverse the preprocessing applied to band data"""
    
    normalize_to_neg_one_to_one = data_config.get('normalize_to_neg_one_to_one', False)
    min_db = data_config.get('min_db', None)
    max_db = data_config.get('max_db', None)
    min_positive = data_config.get('min_positive', None)
    if data_config.get('name', None) == 'copgen_lmdb_features':
        min_db = min_db[satellite_type] if min_db is not None else None
        max_db = max_db[satellite_type] if max_db is not None else None
        min_positive = min_positive[satellite_type] if min_positive is not None else None
    
    # First, if data was normalized to [-1,1] range, revert to [0,1] range
    # Skip normalization reversal for mean timesteps and lat/lon
    if 'mean_timestamps' in satellite_type or '3d_cartesian_lat_lon' in satellite_type:
        data = processed_data
    elif normalize_to_neg_one_to_one:
        data = (processed_data + 1.0) / 2.0
    else:
        data = processed_data
    
    # if 'cloud_mask' in satellite_type:
    #     data = data * 3.0
    #     return data
    # elif 'LULC' in satellite_type:
    #     data = data * 11.0
    #     return data
    
    # Cloud mask needs to be checked first to avoid confusion with S2L1C_cloud_mask and other modalities with S2L1C prefix (e.g. S2L1C_B02_B03_B04_B08)
    if 'cloud_mask' in satellite_type or 'LULC' in satellite_type:
        return one_hot_decode(data, value_to_index=get_value_to_index(satellite_type))
    elif "thumbnail" in satellite_type:
        return data
    elif 'S2L2A' in satellite_type or 'S2L1C' in satellite_type:
        # Reverse the division by 10,000
        data = data * 10000.0
        return data
    elif 'S1RTC' in satellite_type:
        # Reverse normalization for S1 backscatter values:
        # model outputs are in normalized dB space (optionally [-1,1] already handled above).
        # Map [0,1] to [min_db, max_db], then convert dB -> linear backscatter.
        if min_db is not None and max_db is not None:
            if torch.is_tensor(data):
                data = torch.clamp(data, 0.0, 1.0)
            else:
                data = np.clip(data, 0.0, 1.0)
            data = data * (max_db - min_db) + min_db
        else:
            print(f"WARNING: No min/max values defined for {satellite_type}. No reverse normalization will be applied.")
        # Convert from dB to backscatter values
        data = 10 ** (data / 10.0)
        # Apply safety floor consistent with pre-processing
        if min_positive is not None:
            if torch.is_tensor(data):
                data = torch.clamp(data, min=min_positive)
            else:
                data = np.clip(data, a_min=min_positive, a_max=None)
        return data
    elif 'DEM' in satellite_type:
        # print(f"OUTPUT. data min: {np.min(data.cpu().numpy())}, max: {np.max(data.cpu().numpy())}")
        if min_positive is None:
            raise ValueError(f"ERROR: No min_positive value defined for {satellite_type}. Please provide a min_positive value.")
        # Reverse the normalization for DEM values
        if min_db is not None and max_db is not None:
            if torch.is_tensor(data):
                data = torch.clamp(data, 0.0, 1.0)
            else:
                data = np.clip(data, 0.0, 1.0)
            # Scale from [0,1] back to [min_db, max_db]
            data = data * (max_db - min_db) + min_db
        else:
            print(f"WARNING: No min/max values defined for {satellite_type}. No reverse normalization will be applied.")
        # Convert from normalized values to actual DEM values
        if torch.is_tensor(data):
            data = torch.expm1(data)
            # Optional safety floor to counter numeric noise
            data = torch.clamp(data, min=min_positive)
        else:
            data = np.expm1(data)
            data = np.clip(data, a_min=min_positive)
        # data = 10 ** (data / 10.0)
        # print(f"data min: {np.min(data)}, max: {np.max(data)}")
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 6))
        # plt.hist(data.flatten(), bins=100)
        # plt.xlabel('DEM Value')
        # plt.ylabel('Frequency')
        # plt.title('Histogram of DEM Values')
        # plt.grid(alpha=0.3)
        # plt.savefig(f'dem_histogram_{np.min(data)}_{np.max(data)}.png', dpi=300, bbox_inches='tight')
        # plt.close()
        # return torch.from_numpy(data)
        return data
    elif '3d_cartesian_lat_lon' in satellite_type:
        return decode_lat_lon(data, output_format='3d_cartesian_lat_lon')
    elif 'mean_timestamps' in satellite_type:
        return decode_date(data)
    else:
        raise ValueError(f"ERROR: Post-processing not implemented for satellite type: {satellite_type}")
        # For unknown type, we cannot reverse the normalization
        # since we don't have the original min/max values
        return data