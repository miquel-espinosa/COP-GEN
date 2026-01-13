from rasterio.io import MemoryFile
from PIL import Image
import numpy as np
from pathlib import Path
from matplotlib.colors import LightSource
import torch

def _process_s2_bands_torch(bands, gain=1.3, gamma=0.6, return_uint8=False):
    """
    Helper function for processing Sentinel-2 bands
    Applies gain, gamma correction and scales to 0-1 float or 0-255 uint8
    
    Args:
        bands: Tensor with S2 band data (assumed to be in 0-10000 range)
        gain: Gain factor to apply
        gamma: Gamma correction factor
        return_uint8: If True, return uint8 tensor, otherwise return float tensor
        
    Returns:
        Processed tensor in 0-1 float range or 0-255 uint8 range
    """
    # apply gain & gamma
    processed = gain * ((bands / 10_000) ** gamma)
    processed = torch.clamp(processed, 0, 1)
    
    if return_uint8:
        return (processed * 255).to(torch.uint8)
    return processed

def s2_thumbnail_torch(B04, B03, B02, gain=1.3, gamma=0.6, return_uint8=False):
    """
    Creates an RGB thumbnail from Sentinel-2 bands
    
    Args:
        B04, B03, B02: Red, Green, Blue band tensors
        gain: Gain factor to apply
        gamma: Gamma correction factor
        return_uint8: If True, return uint8 tensor, otherwise return float tensor
    
    Returns:
        RGB tensor in 0-1 float range or 0-255 uint8 range
    """
    # concatenate
    thumb = torch.stack([B04, B03, B02], dim=-1)
    # process using the helper function
    return _process_s2_bands_torch(thumb, gain, gamma, return_uint8)

def s2_singleband_torch(band, gain=1.3, gamma=0.6, return_uint8=False):
    """
    Creates a grayscale visualisation from a single Sentinel-2 band
    
    Args:
        band: Single band tensor
        gain: Gain factor to apply
        gamma: Gamma correction factor
        return_uint8: If True, return uint8 tensor, otherwise return float tensor
    
    Returns:
        Grayscale tensor in 0-1 float range or 0-255 uint8 range
    """
    return _process_s2_bands_torch(band, gain, gamma, return_uint8)

def _process_s1rtc_band_torch(band, nodata_value=-32768.0, return_uint8=False):
    """
    Helper function to process a single radar band (vv or vh)
    
    Args:
        band: Tensor with radar band data
        nodata_value: Value representing no data
        return_uint8: If True, return uint8 tensor, otherwise return float tensor
    
    Returns:
        Normalized dB values in 0-1 float range or 0-255 uint8 range
    """
    # valid data mask
    mask = band != nodata_value
    # remove invalid values before log op
    band = torch.where(band < 0, band[band >= 0].min(), band)
    # apply log op
    band_dB = 10 * torch.log10(band)
    # normalize to 0-1
    band_min = band_dB[mask].min()
    band_max = band_dB[mask].max()
    band_normalized = (band_dB - band_min) / (band_max - band_min)
    # represent nodata as 0
    band_normalized = torch.where(mask == 0, torch.tensor(0.0, device=band.device), band_normalized)
    
    if return_uint8:
        return (band_normalized * 255).to(torch.uint8)
    return band_normalized

def s1rtc_thumbnail_torch(vv, vh, vv_NODATA=-32768.0, vh_NODATA=-32768.0, return_uint8=False):
    """
    Creates an RGB thumbnail from Sentinel-1 VV and VH bands
    
    Args:
        vv, vh: VV and VH band tensors
        vv_NODATA, vh_NODATA: No data values
        return_uint8: If True, return uint8 tensor, otherwise return float tensor
    
    Returns:
        RGB tensor in 0-1 float range or 0-255 uint8 range
    """
    # Process each band to 0-1 range
    vv_norm = _process_s1rtc_band_torch(vv, vv_NODATA, False)
    vh_norm = _process_s1rtc_band_torch(vh, vh_NODATA, False)
    
    # Create false color composite
    combined = vv_norm + vh_norm
    combined_max = torch.max(combined)
    if combined_max > 0:  # Avoid division by zero
        combined_norm = combined / combined_max
    else:
        combined_norm = combined
        
    # Stack into RGB
    rgb = torch.stack([vv_norm, combined_norm, vh_norm], dim=-1)
    
    if return_uint8:
        return (rgb * 255).to(torch.uint8)
    return rgb

def s1rtc_singleband_torch(band, nodata_value=-32768.0, return_uint8=False):
    """
    Creates a grayscale visualisation from a single Sentinel-1 band
    
    Args:
        band: Single band tensor (vv or vh)
        nodata_value: No data value
        return_uint8: If True, return uint8 tensor, otherwise return float tensor
    
    Returns:
        Grayscale tensor in 0-1 float range or 0-255 uint8 range
    """
    return _process_s1rtc_band_torch(band, nodata_value, return_uint8)

def normalize_tensor(x, return_uint8=False, topo_color=True):
    """
    Normalize tensor to 0-1 range
    
    Args:
        x: Input tensor
        return_uint8: If True, return uint8 tensor, otherwise return float tensor
        topo_color: If True, return colorful topographic visualisation (teal to orange)
        
    Returns:
        Normalized tensor in 0-1 float range or 0-255 uint8 range
    """
    # Normalize per image
    if x.ndim == 4:
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
    else:
        x_min, x_max = x.min(), x.max()
    
    if x_max.max() > x_min.min():
        x_norm = (x - x_min) / (x_max - x_min + 1e-7)
    else:
        x_norm = torch.zeros_like(x)
    
    if topo_color:
        # Apply teal to orange colormap: teal (low) -> white (mid) -> orange (high)
        teal = torch.tensor([0.0, 0.5, 0.5], device=x.device).view(1, 3, 1, 1)
        white = torch.tensor([1.0, 1.0, 1.0], device=x.device).view(1, 3, 1, 1)
        orange = torch.tensor([1.0, 0.65, 0.0], device=x.device).view(1, 3, 1, 1)
        
        # Split into two gradients at 0.5
        mask_low = (x_norm <= 0.5).float()
        mask_high = (x_norm > 0.5).float()
        
        # Low: teal to white (0 to 0.5)
        t_low = x_norm * 2  # Scale 0-0.5 to 0-1
        color_low = teal + t_low * (white - teal)
        
        # High: white to orange (0.5 to 1)
        t_high = (x_norm - 0.5) * 2  # Scale 0.5-1 to 0-1
        color_high = white + t_high * (orange - white)
        
        x_norm = color_low * mask_low + color_high * mask_high
        
    if return_uint8:
        return (x_norm * 255).to(torch.uint8)
    return x_norm

def get_hillshade_torch_differentiable(x, azdeg=315, altdeg=45, ve=1, return_uint8=False, normalize_per_image=False):
    """
    Creates a differentiable hillshade visualisation for DEM data using only torch operations
    
    Args:
        x: DEM tensor with shape (batch_size, channels, height, width)
        azdeg: Azimuth angle in degrees
        altdeg: Altitude angle in degrees
        ve: Vertical exaggeration
        return_uint8: If True, return uint8 tensor, otherwise return float tensor
        normalize_per_image: If True, normalize each image separately (for visualisation), 
                            if False, just clip values (better for loss functions)
        
    Returns:
        Hillshade tensor in 0-1 float range or 0-255 uint8 range with gradient flow preserved
    """

    elevation = x.squeeze(1)  # Remove channel dim
    assert elevation.ndim == 3
    
    # Calculate light source direction
    azimuth_rad = torch.tensor(np.radians(90 - azdeg), device=x.device)
    altitude_rad = torch.tensor(np.radians(altdeg), device=x.device)
    
    direction = torch.tensor([
        torch.cos(azimuth_rad) * torch.cos(altitude_rad),
        torch.sin(azimuth_rad) * torch.cos(altitude_rad),
        torch.sin(altitude_rad)
    ], device=x.device)
    
    # Compute the gradients using torch.gradient for better accuracy
    # Note: dy is negated to match the convention in matplotlib
    dy_dx = torch.gradient(ve * elevation, dim=(1, 2))
    e_dy = -dy_dx[0]  # Negate dy to match matplotlib's convention
    e_dx = dy_dx[1]
    
    # Create normal vectors (oriented with z-up)
    normal = torch.zeros((*elevation.shape, 3), device=x.device)
    normal[..., 0] = -e_dx
    normal[..., 1] = -e_dy
    normal[..., 2] = 1.0
    
    # Normalize the normal vectors
    norm = torch.sqrt(torch.sum(normal**2, dim=-1, keepdim=True))
    normal = normal / (norm + 1e-7)  # Add small epsilon to avoid division by zero
    
    # Calculate intensity (dot product of normal and light direction)
    intensity = torch.sum(normal * direction, dim=-1)
    
    if normalize_per_image:
        # Apply contrast stretch (using fraction=1.0)
        # This can be parametrized if needed
        fraction = 1.0
        
        # Calculate min and max per batch item for proper rescaling
        batch_min = intensity.amin(dim=(1, 2), keepdim=True)
        batch_max = intensity.amax(dim=(1, 2), keepdim=True)
        
        # Rescale to 0-1 range
        intensity = intensity * fraction
        
        # Rescale based on original range, avoiding division by zero
        range_mask = (batch_max - batch_min) > 1e-6
        normalized = torch.zeros_like(intensity)
        
        for i in range(intensity.shape[0]):
            if range_mask[i, 0, 0]:
                normalized[i] = (intensity[i] - batch_min[i]) / (batch_max[i] - batch_min[i])
            else:
                # For flat surfaces, use 0.5 as in matplotlib
                normalized[i] = torch.ones_like(intensity[i]) * 0.5
        
        result = torch.clamp(normalized, 0, 1)
    else:
        # For loss functions, use raw intensity values without normalisation or clamping
        # This preserves the full information about surface orientation
        result = intensity
    
    # Add channel dimension back
    result = result.unsqueeze(1)
    
    if return_uint8:
        return (result * 255).to(torch.uint8)
    return result


def get_hillshade_torch(x, azdeg=315, altdeg=45, ve=1, return_uint8=False):
    """
    Creates a hillshade visualisation for DEM data
    
    Args:
        x: DEM tensor with shape (batch_size, channels, height, width)
        azdeg: Azimuth angle in degrees
        altdeg: Altitude angle in degrees
        ve: Vertical exaggeration
        return_uint8: If True, return uint8 tensor, otherwise return float tensor
        
    Returns:
        Hillshade tensor in 0-1 float range or 0-255 uint8 range
    """
    batch_size = x.shape[0]
    hillshade_list = []
    
    ls = LightSource(azdeg=azdeg, altdeg=altdeg)
    for i in range(batch_size):
        x_i = x[i].squeeze().cpu().numpy()
        hillshade = ls.hillshade(x_i, vert_exag=ve)
        hillshade_list.append(hillshade)
    hillshade_tensor = torch.tensor(np.stack(hillshade_list), device=x.device).unsqueeze(1)
    if return_uint8:
        return (hillshade_tensor * 255).to(torch.uint8)
    return hillshade_tensor

def dem_thumbnail_torch(dem, dem_NODATA=-32768.0, hillshade=True, return_uint8=False):
    """
    Creates a visualisation for DEM data
    
    Args:
        dem: DEM tensor
        dem_NODATA: No data value
        hillshade: If True, use hillshade visualisation, otherwise use grayscale
        return_uint8: If True, return uint8 tensor, otherwise return float tensor
        
    Returns:
        DEM visualisation tensor in 0-1 float range or 0-255 uint8 range
    """
    if hillshade:
        return get_hillshade_torch(dem, return_uint8=return_uint8)
    else:
        return normalize_tensor(dem, return_uint8=return_uint8)

def classes_thumbnail_torch(tensor, value_to_index, return_uint8=False):
    """
    Creates a visualisation for LULC or cloud mask data
    
    Args:
        tensor: Tensor with class data in original class values (e.g., 0, 1, 2, 4, 5, 7, 8, 9, 10, 11 for LULC)
        value_to_index: Dictionary mapping original class values to indices
        return_uint8: If True, return uint8 tensor, otherwise return float tensor
        
    Returns:
        Class visualisation tensor in 0-1 float range or 0-255 uint8 range
    """
    
    # Safety check that the tensor only contains integers values (even if they are floats)
    if not torch.all(tensor.int() == tensor):
        raise ValueError("For visualisation of LULC/cloud_mask, tensor must contain only integer values")
    
    # create a reasonable color map - using distinct colors for each class
    colors = [
        [0,   0,   0],      # 0: Black                               | Clear
        [26,  91,  171],    # 1: Blue         | Water                | Thick
        [53,  130, 33],     # 2: Green        | Trees                | Thin
        [135, 209, 158],    # 3: Light green  | Flooded vegetation   | Shadow
        [255, 219, 92],     # 4: Yellow       | Crops
        [237, 2,   42],     # 5: Red          | Built-up areas
        [227, 226, 195],    # 6: Light yellow | Bare Ground
        [168, 235, 255],    # 7: Light blue   | Snow/Ice
        [97,  97,  97],     # 8: Dark gray    | Clouds
        [165, 155, 143],    # 9: Brown        | Rangeland
    ]
    colors_tensor = torch.tensor(colors, device=tensor.device).float() / 255.0

    # create RGB image by mapping original class values to colors via their indices
    result = torch.zeros((tensor.shape[0], 3, tensor.shape[2], tensor.shape[3]), 
                        device=tensor.device)
    
    # Map original values to their corresponding colors
    for orig_val, idx in value_to_index.items():
        mask = (tensor == orig_val)
        result += mask * colors_tensor[idx].view(1, 3, 1, 1)
        
    if return_uint8:
        return (result * 255).to(torch.uint8)
    return result

def lat_lon_thumbnail_torch(tensor, return_uint8=False):
    """
    Creates a visualisation for lat/lon data, a thumbnail render of the lat/lon values
    
    Args:
        tensor: Tensor with lat/lon data, already in degrees
        return_uint8: If True, return uint8 tensor, otherwise return float tensor
        
    Returns:
        Lat/lon visualisation tensor in 0-1 float range or 0-255 uint8 range
    """
    import PIL.Image
    import PIL.ImageDraw
    import PIL.ImageFont
    
    # Create a white canvas
    batch_size = tensor.shape[0]
    result = torch.ones((batch_size, 3, 128, 128), device=tensor.device)
    
    # Process each image in batch
    for i in range(batch_size):
        # Create PIL Image with white background
        img = PIL.Image.new('RGB', (128, 128), color='white')
        draw = PIL.ImageDraw.Draw(img)
        
        # Get lat/lon values
        lat = tensor[i, 0, 0, 0].item()
        lon = tensor[i, 0, 0, 1].item()
        
        # Format text
        text = f"{lat:.2f}\n{lon:.2f}"
        
        # Calculate text size and position to center
        font = PIL.ImageFont.load_default()
        text_bbox = draw.textbbox((0,0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        x = (128 - text_w) // 2
        y = (128 - text_h) // 2
        
        # Draw text in black
        draw.text((x, y), text, fill='black', font=font)
        
        # Convert to tensor and normalize
        img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
        result[i] = img_tensor.to(tensor.device)
    
    if return_uint8:
        return (result * 255).to(torch.uint8)
    return result

def date_thumbnail_torch(tensor, return_uint8=False):
    """
    Creates a visualisation for date information, rendering day-month-year text.

    Args:
        tensor: Tensor with date data in day-month-year order, shape (batch_size, 1, 1, 3)
        return_uint8: If True, return uint8 tensor, otherwise return float tensor

    Returns:
        Date visualisation tensor in 0-1 float range or 0-255 uint8 range
    """
    import PIL.Image
    import PIL.ImageDraw
    import PIL.ImageFont

    # Create a white canvas
    batch_size = tensor.shape[0]
    result = torch.ones((batch_size, 3, 128, 128), device=tensor.device)

    # Process each image in the batch
    for i in range(batch_size):
        # Create PIL Image with white background
        img = PIL.Image.new('RGB', (128, 128), color='white')
        draw = PIL.ImageDraw.Draw(img)

        # Get day, month, year values
        day = int(tensor[i, 0, 0, 0].item())
        month = int(tensor[i, 0, 0, 1].item())
        year = int(tensor[i, 0, 0, 2].item())

        # Format text as DD-MM-YYYY
        text = f"{day:02d}-{month:02d}-{year:04d}"

        # Calculate text size and position to center
        font = PIL.ImageFont.load_default()
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        x = (128 - text_w) // 2
        y = (128 - text_h) // 2

        # Draw text in black
        draw.text((x, y), text, fill='black', font=font)

        # Convert to tensor and normalize
        img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
        result[i] = img_tensor.to(tensor.device)

    if return_uint8:
        return (result * 255).to(torch.uint8)
    return result
        
# def get_hillshade_np(x, azdeg=315, altdeg=45,ve=1):
#     """
#         Hillshade visualisation for DEM
#     """
#     ls = LightSource(azdeg=azdeg, altdeg=altdeg)
#     return np.uint8(255*ls.hillshade(x, vert_exag=ve))

# def dem_thumbnail_np(dem, dem_NODATA = -32768.0, hillshade=True):
#     """
#         Takes vv and vh numpy arrays along with the corresponding NODATA values (default is -32768.0)
#         Returns a numpy array with the thumbnail
#     """
#     if hillshade:
#         return get_hillshade_np(dem)
#     else:
#         return get_grayscale_np(dem)

# def get_grayscale_np(x):
#     """
#         Normalized grayscale visualisation
#     """
#     # normalize
#     x_n = x-x.min()
#     x_n = x_n/x_n.max()
#     return np.uint8(x_n*255)

# def dem_thumbnail_from_datarow(datarow):
#     """
#         Takes a datarow directly from one of the data parquet files
#         Returns a PIL Image
#     """
#     with MemoryFile(datarow['DEM'][0].as_py()) as mem_f:
#         with mem_f.open(driver='GTiff') as f:
#             dem=f.read().squeeze()
#             dem_NODATA = f.nodata
#     img = dem_thumbnail(dem, dem_NODATA)
#     return Image.fromarray(img,'L')


# def s2_thumbnail_np(B04, B03, B02, gain=1.3, gamma=0.6):
#     """
#         Takes B04, B03, B02 numpy arrays along with the corresponding NODATA values (default is -32768.0)
#         Returns a numpy array with the thumbnail
#     """
#     # concatenate
#     thumb = np.stack([B04, B03, B02], -1)
#     # apply gain & gamma
#     thumb = gain*((thumb/10_000)**gamma)
#     return (thumb.clip(0,1)*255).astype(np.uint8)

# def s2_thumbnail_from_datarow(datarow):
#     """
#         Takes a datarow directly from one of the data parquet files
#         Returns a PIL Image
#     """
#     # red
#     with MemoryFile(datarow['B04'][0].as_py()) as mem_f:
#         with mem_f.open(driver='GTiff') as f:
#             B04=f.read().squeeze()
#             B04_NODATA = f.nodata
#     # green
#     with MemoryFile(datarow['B03'][0].as_py()) as mem_f:
#         with mem_f.open(driver='GTiff') as f:
#             B03=f.read().squeeze()
#             B03_NODATA = f.nodata
#     # blue
#     with MemoryFile(datarow['B02'][0].as_py()) as mem_f:
#         with mem_f.open(driver='GTiff') as f:
#             B02=f.read().squeeze()
#             B02_NODATA = f.nodata
#     img = s2_thumbnail(B04,B03,B02)
#     return Image.fromarray(img)

# def s1rtc_thumbnail_np(vv, vh, vv_NODATA = -32768.0, vh_NODATA = -32768.0):
#     """
#         Takes vv and vh numpy arrays along with the corresponding NODATA values (default is -32768.0)
#         Returns a numpy array with the thumbnail
#     """
#     # valid data masks
#     vv_mask = vv != vv_NODATA
#     vh_mask = vh != vh_NODATA
#     # remove invalid values before log op
#     vv[vv<0] = vv[vv>=0].min()
#     vh[vh<0] = vh[vh>=0].min()
#     # apply log op
#     vv_dB = 10*np.log10(vv)
#     vh_dB = 10*np.log10(vh)
#     # scale to 0-255
#     vv_dB = (vv_dB - vv_dB[vv_mask].min()) / (vv_dB[vv_mask].max() - vv_dB[vv_mask].min()) * 255
#     vh_dB = (vh_dB - vh_dB[vh_mask].min()) / (vh_dB[vh_mask].max() - vh_dB[vh_mask].min()) * 255
#     # represent nodata as 0
#     vv_dB[vv_mask==0] = 0
#     vh_dB[vh_mask==0] = 0
#     # false colour composite
#     return np.stack([vv_dB,
#                     255*(vv_dB+vh_dB)/np.max(vv_dB+vh_dB),
#                     vh_dB
#                    ],-1).astype(np.uint8)

# def s1rtc_thumbnail_from_datarow(datarow):
#     """
#         Takes a datarow directly from one of the data parquet files
#         Returns a PIL Image
#     """
#     with MemoryFile(datarow['vv'][0].as_py()) as mem_f:
#         with mem_f.open(driver='GTiff') as f:
#             vv=f.read().squeeze()
#             vv_NODATA = f.nodata
#     with MemoryFile(datarow['vh'][0].as_py()) as mem_f:
#         with mem_f.open(driver='GTiff') as f:
#             vh=f.read().squeeze()
#             vh_NODATA = f.nodata
#     img = s1rtc_thumbnail(vv, vh, vv_NODATA=vv_NODATA, vh_NODATA=vh_NODATA)
#     return Image.fromarray(img)

if __name__ == '__main__':  
    # from fsspec.parquet import open_parquet_file
    # import pyarrow.parquet as pq

    # print('[example run] reading file from HuggingFace...')
    # url = "https://huggingface.co/datasets/Major-TOM/Core-S2L1C/resolve/main/images/part_01000.parquet"
    # with open_parquet_file(url, columns = ["B04", "B03", "B02"]) as f:
    #     with pq.ParquetFile(f) as pf:
    #         first_row_group = pf.read_row_group(1, columns = ["B04", "B03", "B02"])
    
    # print('[example run] computing the thumbnail...')    
    # thumbnail = s2_thumbnail_from_datarow(first_row_group)
    
    # thumbnail.save('example_thumbnail.png', format = 'PNG')
    
    # Save the LULC classes legend
    import matplotlib.pyplot as plt

    # Define LULC class names and colors (should match classes_thumbnail_torch)
    lulc_classes = [
        "0: None",
        "1: Water",
        "2: Trees",
        "3: Flooded veg",
        "4: Crops",
        "5: Built-up",
        "6: Bare Ground",
        "7: Snow/Ice",
        "8: Clouds",
        "9: Rangeland"
    ]
    colors = [
        [0,   0,   0],      # 0: Black                               | Clear
        [26,  91,  171],    # 1: Blue         | Water                | Thick
        [53,  130, 33],     # 2: Green        | Trees                | Thin
        [135, 209, 158],    # 3: Light green  | Flooded vegetation   | Shadow
        [255, 219, 92],     # 4: Yellow       | Crops
        [237, 2,   42],     # 5: Red          | Built-up areas
        [227, 226, 195],    # 6: Light yellow | Bare Ground
        [168, 235, 255],    # 7: Light blue   | Snow/Ice
        [97,  97,  97],     # 8: Dark gray    | Clouds
        [165, 155, 143],    # 9: Brown        | Rangeland
    ]
    colors = [[c[0]/255, c[1]/255, c[2]/255] for c in colors]

    fig, ax = plt.subplots(figsize=(4, 4))
    for i, (name, color) in enumerate(zip(lulc_classes, colors)):
        ax.barh(i, 1, color=color, edgecolor='black')
    ax.set_yticks(range(len(lulc_classes)))
    ax.set_yticklabels(lulc_classes, fontsize=10)
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.set_title("LULC Classes Legend", fontsize=12)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("plots/lulc_classes_legend.png", dpi=200)
    plt.close()
    
    
    # Cloud cover legend

    # Define cloud cover class names and colors (should match classes_thumbnail_torch for cloud mask)
    cloud_classes = [
        "0: Clear",
        "1: Thick",
        "2: Thin",
        "3: Shadow"
    ]
    cloud_colors = [
        [0,   0,   0],      # 0: Black (Clear)
        [26,  91,  171],    # 1: Blue (Thick)
        [53,  130, 33],     # 2: Green (Thin)
        [135, 209, 158],    # 3: Light green (Shadow)
    ]
    cloud_colors = [[c[0]/255, c[1]/255, c[2]/255] for c in cloud_colors]

    fig, ax = plt.subplots(figsize=(3, 2.5))
    for i, (name, color) in enumerate(zip(cloud_classes, cloud_colors)):
        ax.barh(i, 1, color=color, edgecolor='black')
    ax.set_yticks(range(len(cloud_classes)))
    ax.set_yticklabels(cloud_classes, fontsize=10)
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.set_title("Cloud Mask Legend", fontsize=12)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("plots/cloud_mask_legend.png", dpi=200)
    plt.close()
    
    
