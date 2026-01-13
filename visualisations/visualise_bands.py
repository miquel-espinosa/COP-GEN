import os
import torch
import torch.nn.functional as F
import torchvision as tv
import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path

from visualisations.thumbnails_vis import (
    s2_thumbnail_torch, s2_singleband_torch,
    s1rtc_thumbnail_torch, s1rtc_singleband_torch,
    dem_thumbnail_torch, normalize_tensor,
    classes_thumbnail_torch, lat_lon_thumbnail_torch,
    date_thumbnail_torch
)
from ddm.pre_post_process_data import get_value_to_index

def visualise_bands(inputs, reconstructions, save_dir, repeat_n_times=None, n_images_to_log=4,
                    milestone=None, satellite_type=None,
                    view_difference=False, view_histogram=False,
                    resize=None, input_frame_color=(0.0, 1.0, 0.0), recon_frame_color=(0.0, 0.0, 1.0)):
    """
    Visualize multi-band satellite imagery with input and reconstruction side by side.
    
    Args:
        inputs: Input images tensor [B, C, H, W] or None for reconstruction-only visualisation
        reconstructions: Reconstructed images tensor [B, C, H, W]
        save_dir: Directory to save visualisations
        n_images_to_log: Number of images to visualize
        milestone: Optional milestone/step number for naming
        satellite_type: Type of satellite data ('S2L2A', 'S2L1C', 'S1RTC', 'DEM')
    """
    
    # Setup directories and prepare data
    save_dir, vis_folder, results = _setup_visualisation_dirs(save_dir, milestone)
    
    # If modality was used as condition, we need to repeat the input and reconstruction for each sample
    if repeat_n_times is not None:
        if inputs is not None:
            inputs = inputs.repeat_interleave(repeat_n_times, dim=0)
        if reconstructions is not None:
            reconstructions = reconstructions.repeat_interleave(repeat_n_times, dim=0)
    
    # Limit to requested number of images
    n_images_to_log = min(n_images_to_log, reconstructions.shape[0])
    reconstructions = reconstructions[:n_images_to_log]
    inputs = inputs[:n_images_to_log] if inputs is not None else None
    
    num_bands = reconstructions.shape[1]
    
    # 1. Visualize individual bands
    _visualize_individual_bands(
        inputs, reconstructions, vis_folder, satellite_type, n_images_to_log, 
        num_bands, results, view_difference=view_difference, view_histogram=view_histogram,
        resize=resize, input_frame_color=input_frame_color, recon_frame_color=recon_frame_color
    )
    
    if num_bands == 1:
        return results
    
    # 2. Create composite visualisations based on satellite type
    _create_composite_visualisations(
        inputs, reconstructions, vis_folder, satellite_type, n_images_to_log, 
        num_bands, results, view_difference=view_difference, view_histogram=view_histogram,
        resize=resize, input_frame_color=input_frame_color, recon_frame_color=recon_frame_color
    )
    
    # 3. Create the main sample visualisation
    # _create_main_visualisation(
    #     inputs, reconstructions, save_dir, vis_folder, satellite_type, 
    #     n_images_to_log, num_bands, milestone, results, view_difference=view_difference, view_histogram=view_histogram
    # )
    
    return results

def _setup_visualisation_dirs(save_dir, milestone):
    """Setup visualisation directories and return initialized results dict"""
    
    save_dir = Path(save_dir)
    if milestone is not None:
        vis_folder = save_dir / f'visualisations-{milestone}'
    else:
        vis_folder = save_dir / 'visualisations'
    vis_folder.mkdir(exist_ok=True, parents=True)
    
    return save_dir, vis_folder, {}

def _ensure_tensor_format(tensor):
    """Ensure tensor is in the format [B, C, H, W]"""
    # If tensor is in format [B, H, W, C]
    if tensor.ndim == 4 and tensor.shape[3] in [1, 3]:
        return tensor.permute(0, 3, 1, 2)
    # If tensor is in format [H, W, C]
    elif tensor.ndim == 3 and tensor.shape[2] in [1, 3]:
        return tensor.permute(2, 0, 1).unsqueeze(0)
    # If already in format [B, C, H, W] or [C, H, W]
    return tensor

def _get_tile_hw_and_channels(x):
    """Return (H, W, C) after ensuring tensor format is [B, C, H, W]."""
    x_fmt = _ensure_tensor_format(x)
    # return int(x_fmt.shape[2]), int(x_fmt.shape[3]), int(x_fmt.shape[1])
    return 192, 192, int(x_fmt.shape[1])

def _render_histogram_tile(gt: np.ndarray, pred: np.ndarray, width: int, height: int, to_db: bool = False) -> torch.Tensor:
    """Render a single histogram tile (overlay GT vs Pred) to a CHW tensor in [0,1].
    - gt, pred are flattened numpy arrays of pixel values for one image (possibly multi-band interleaved)
    - width, height define the output tile size matching other modalities
    """
    if to_db:
        gt = 10 * np.log10(gt)
        pred = 10 * np.log10(pred)
    fig = plt.figure(figsize=(2, 2), dpi=100)
    ax = fig.add_axes([0.1, 0.2, 0.85, 0.7])  # tight layout, leave space for x-axis and small legend
    bins = 100
    ax.hist(gt, bins=bins, alpha=0.5, label='GT', color='blue', density=True)
    ax.hist(pred, bins=bins, alpha=0.5, label='Pred', color='orange', density=True)
    ax.tick_params(axis='both', which='both', labelsize=6)
    # Small legend
    ax.legend(fontsize=6, loc='upper right', frameon=False)
    # Only x axis label ticks, no title
    # Render to numpy array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    # Convert to tensor CHW in [0,1], resize to target width/height
    img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img_t = img_t.unsqueeze(0)  # 1,C,H,W
    img_t = F.interpolate(img_t, size=(height, width), mode='bilinear', align_corners=False)
    img_t = img_t.squeeze(0)
    return img_t

def _generate_histogram_row_tensor(gt_tensor: torch.Tensor, pred_tensor: torch.Tensor, tile_h: int, tile_w: int, out_channels: int, to_db: bool = False) -> torch.Tensor:
    """Generate a row of histogram tiles (B, C, H, W) aligned with batch order.
    - gt_tensor, pred_tensor: shapes [B, C, H, W] for the subset of bands to compare
    - out_channels: expected number of channels to match visual tiles (1 or 3). If 1, convert to grayscale
    """
    B = pred_tensor.shape[0]
    tiles = []
    gt_np = gt_tensor.detach().cpu().numpy()
    pred_np = pred_tensor.detach().cpu().numpy()

    for i in range(B):
        gt_flat = gt_np[i].reshape(-1)
        pred_flat = pred_np[i].reshape(-1)
        tile = _render_histogram_tile(gt_flat, pred_flat, tile_w, tile_h, to_db=to_db)
        # Adjust channels to match out_channels
        if out_channels == 1 and tile.shape[0] == 3:
            # Convert RGB to grayscale using luminance weights
            r, g, b = tile[0:1], tile[1:2], tile[2:3]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            tile = gray
        elif out_channels == 3 and tile.shape[0] == 1:
            tile = tile.repeat(3, 1, 1)
        tiles.append(tile.unsqueeze(0))

    return torch.cat(tiles, dim=0)

def _apply_color_frame(batch_tensor: torch.Tensor, color_rgb, border_px: int = 2) -> torch.Tensor:
    """Return a new tensor with a colored frame outside each image (padding).
    Does not modify the input. The output shape is [B, C, H+2*border_px, W+2*border_px].
    color_rgb should be a 3-tuple with values in [0,1]. For single-channel images, use luminance.
    """
    t = _ensure_tensor_format(batch_tensor)
    B, C, H, W = t.shape
    if border_px <= 0:
        return t
    new_h = H + 2 * border_px
    new_w = W + 2 * border_px
    out = torch.zeros((B, C, new_h, new_w), dtype=t.dtype, device=t.device)
    if C == 1:
        # Convert RGB to luminance for single-channel
        y_val = 0.2126 * float(color_rgb[0]) + 0.7152 * float(color_rgb[1]) + 0.0722 * float(color_rgb[2])
        # Top and bottom borders
        out[:, :, 0:border_px, :] = y_val
        out[:, :, -border_px:, :] = y_val
        # Left and right borders
        out[:, :, :, 0:border_px] = y_val
        out[:, :, :, -border_px:] = y_val
    else:
        # Use first 3 channels for color application, leave others as zeros in the frame
        color = torch.tensor(color_rgb, dtype=t.dtype, device=t.device).view(1, 3, 1, 1)
        # Top and bottom
        out[:, 0:3, 0:border_px, :] = color
        out[:, 0:3, -border_px:, :] = color
        # Left and right
        out[:, 0:3, :, 0:border_px] = color
        out[:, 0:3, :, -border_px:] = color
    # Paste original image in the center
    out[:, :, border_px:border_px + H, border_px:border_px + W] = t
    return out

def _save_comparison_image(input_vis, recon_vis, save_path, n_images_to_log, view_difference, extra_row=None,
                           resize=None, input_frame_color=(0.0, 1.0, 0.0), recon_frame_color=(0.0, 0.0, 1.0)):
    """Save a side-by-side comparison of input and reconstruction, or just reconstruction if input is None.
    If extra_row (tensor [B, C, H, W]) is provided, append it as an additional row aligned by column.
    """
    FRAME_PX = 4
    raw_path = str(Path(save_path).with_name(Path(save_path).stem + '_raw' + Path(save_path).suffix))
    # Handle case where input_vis is None (reconstruction-only mode)
    if input_vis is None:
        recon_vis = _ensure_tensor_format(recon_vis)
        if resize is not None:
            recon_vis = F.interpolate(recon_vis, size=(resize, resize), mode='bilinear', align_corners=False)
        # Save raw reconstruction without frame
        tv.utils.save_image(recon_vis, raw_path, nrow=n_images_to_log)
        # Apply frame to reconstructions only
        recon_vis = _apply_color_frame(recon_vis, recon_frame_color, border_px=FRAME_PX)
        grid = recon_vis
        if extra_row is not None:
            extra_row = _ensure_tensor_format(extra_row).to(device=recon_vis.device)
            grid = torch.cat([grid, extra_row], dim=0)
        tv.utils.save_image(grid, save_path, nrow=n_images_to_log)
        return save_path
    
    # Ensure tensors are in correct format
    input_vis = _ensure_tensor_format(input_vis).to(device=recon_vis.device)
    recon_vis = _ensure_tensor_format(recon_vis).to(device=recon_vis.device)
    # Optional resize per-tile prior to any further processing
    if resize is not None:
        input_vis = F.interpolate(input_vis, size=(resize, resize), mode='bilinear', align_corners=False)
        recon_vis = F.interpolate(recon_vis, size=(resize, resize), mode='bilinear', align_corners=False)
    # Save raw reconstruction without frame
    tv.utils.save_image(recon_vis, raw_path, nrow=n_images_to_log)
    # Prepare framed copies for saving (keep unframed for diff computation)
    input_vis_framed = _apply_color_frame(input_vis, input_frame_color, border_px=FRAME_PX)
    recon_vis_framed = _apply_color_frame(recon_vis, recon_frame_color, border_px=FRAME_PX)
    comparison = torch.cat([input_vis_framed, recon_vis_framed], dim=0)
    
    if view_difference:
        # Calculate absolute difference
        diff = torch.abs(input_vis - recon_vis)
        
        # Calculate scaled difference (x5) for better visibility
        scaled_diff = torch.clamp(diff * 5, 0, 1)
        
        # Concatenate along batch dimension and save
        comparison = torch.cat([comparison, diff, scaled_diff], dim=0)
        
    if extra_row is not None:
        extra_row = _ensure_tensor_format(extra_row).to(device=recon_vis.device)
        # Resize the smaller image if necessary
        if extra_row.shape[2] < comparison.shape[2] or extra_row.shape[3] < comparison.shape[3]:
            extra_row = F.interpolate(extra_row, size=(comparison.shape[2], comparison.shape[3]), mode='bilinear', align_corners=False)
        if extra_row.shape[2] > comparison.shape[2] or extra_row.shape[3] > comparison.shape[3]:
            comparison = F.interpolate(comparison, size=(extra_row.shape[2], extra_row.shape[3]), mode='bilinear', align_corners=False)
        comparison = torch.cat([comparison, extra_row], dim=0)
    tv.utils.save_image(comparison, save_path, nrow=n_images_to_log)
    
    return save_path

def _visualize_individual_bands(inputs, recons, vis_folder, satellite_type,
                               n_images_to_log, num_bands, results, view_difference=False, view_histogram=False,
                               resize=None, input_frame_color=(0.0, 1.0, 0.0), recon_frame_color=(0.0, 0.0, 1.0)):
    """Visualize each band individually"""
    
    for b in range(num_bands):
        # Generate visualisations for the current band
        if inputs is not None:
            input_vis, recon_vis = _get_band_visualisation(
                inputs[:, b:b+1], recons[:, b:b+1], satellite_type
            )
        else:
            input_vis, recon_vis = _get_band_visualisation(
                None, recons[:, b:b+1], satellite_type
            )
        
        # Save the band visualisation
        band_path = str(vis_folder / f'band_{b}.png')

        # Optional histogram row (requires both inputs and reconstructions)
        hist_row = None
        if view_histogram and inputs is not None:
            # Compute histogram over this single band per image
            tile_h, tile_w, out_c = _get_tile_hw_and_channels(recon_vis)
            hist_row = _generate_histogram_row_tensor(
                inputs[:, b:b+1], recons[:, b:b+1], tile_h, tile_w, out_channels=out_c,
                to_db=True if 'S1RTC' in satellite_type else False
            )

        _save_comparison_image(
            input_vis, recon_vis, band_path, n_images_to_log,
            view_difference=view_difference, extra_row=hist_row,
            resize=resize, input_frame_color=input_frame_color, recon_frame_color=recon_frame_color
        )
        results[f'band_{b}'] = band_path
        
        # For DEM data, also save grayscale visualisation
        if 'DEM' in satellite_type:
            if inputs is not None:
                input_gray_vis = dem_thumbnail_torch(inputs[:, b:b+1], hillshade=False)
            else:
                input_gray_vis = None
                
            recon_gray_vis = dem_thumbnail_torch(recons[:, b:b+1], hillshade=False)
            
            gray_path = str(vis_folder / f'band_{b}_gray.png')
            # Attach same histogram row for gray as well for consistency
            hist_row_gray = None
            if view_histogram and inputs is not None:
                tile_h, tile_w, out_c = _get_tile_hw_and_channels(recon_gray_vis)
                hist_row_gray = _generate_histogram_row_tensor(
                    inputs[:, b:b+1], recons[:, b:b+1], tile_h, tile_w, out_channels=out_c,
                    to_db=True if 'S1RTC' in satellite_type else False
                )
            _save_comparison_image(
                input_gray_vis, recon_gray_vis, gray_path, n_images_to_log,
                view_difference=view_difference, extra_row=hist_row_gray,
                resize=resize, input_frame_color=input_frame_color, recon_frame_color=recon_frame_color
            )
            results[f'band_{b}_gray'] = gray_path

def _get_band_visualisation(band_input, band_recon, satellite_type):
    """Generate visualisation for a single band based on satellite type"""
    
    # Handle reconstruction only mode
    if band_input is None:
        if 'LULC' in satellite_type or 'cloud_mask' in satellite_type:
            recon_vis = classes_thumbnail_torch(band_recon, value_to_index=get_value_to_index(satellite_type))
        elif 'S2L2A' in satellite_type or 'S2L1C' in satellite_type:
            recon_vis = s2_singleband_torch(band_recon)
        elif 'S1RTC' in satellite_type:
            recon_vis = s1rtc_singleband_torch(band_recon)
        elif 'DEM' in satellite_type:
            recon_vis = dem_thumbnail_torch(band_recon, hillshade=True)
        elif 'lat_lon' in satellite_type:
            recon_vis = lat_lon_thumbnail_torch(band_recon)
        elif 'mean_timestamps' in satellite_type:
            recon_vis = date_thumbnail_torch(band_recon)
        else:
            # Fallback for unknown types - simple normalization
            # raise ValueError(f"Visualisation error: Unknown satellite type: {satellite_type}")
            print(f"Visualisation error: Unknown satellite type: {satellite_type}")
            recon_vis = normalize_tensor(band_recon)
        return None, recon_vis
    
    # Check if cloud mask first
    if 'LULC' in satellite_type or 'cloud_mask' in satellite_type:
        input_vis = classes_thumbnail_torch(band_input, value_to_index=get_value_to_index(satellite_type))
        recon_vis = classes_thumbnail_torch(band_recon, value_to_index=get_value_to_index(satellite_type))
    elif 'S2L2A' in satellite_type or 'S2L1C' in satellite_type:
        input_vis = s2_singleband_torch(band_input)
        recon_vis = s2_singleband_torch(band_recon)
    elif 'S1RTC' in satellite_type:
        input_vis = s1rtc_singleband_torch(band_input)
        recon_vis = s1rtc_singleband_torch(band_recon)
    elif 'DEM' in satellite_type:
        input_vis = dem_thumbnail_torch(band_input, hillshade=True)
        recon_vis = dem_thumbnail_torch(band_recon, hillshade=True)
    elif 'lat_lon' in satellite_type:
        input_vis = lat_lon_thumbnail_torch(band_input)
        recon_vis = lat_lon_thumbnail_torch(band_recon)
    elif 'mean_timestamps' in satellite_type:
        input_vis = date_thumbnail_torch(band_input)
        recon_vis = date_thumbnail_torch(band_recon)
    else:
        # Fallback for unknown types - simple normalization
        # raise ValueError(f"Visualisation error: Unknown satellite type: {satellite_type}")
        print(f"Visualisation error: Unknown satellite type: {satellite_type}")
        input_vis = normalize_tensor(band_input)
        recon_vis = normalize_tensor(band_recon)
        
    return input_vis, recon_vis

def _create_composite_visualisations(inputs, recons, vis_folder, 
                                    satellite_type, n_images_to_log, num_bands, results, view_difference=False, view_histogram=False,
                                    resize=None, input_frame_color=(0.0, 1.0, 0.0), recon_frame_color=(0.0, 0.0, 1.0)):
    """Create composite visualisations by grouping bands based on satellite type"""
    
    if 'S2L2A' in satellite_type or 'S2L1C' in satellite_type:
        # For Sentinel-2, create RGB composites from triplets of bands
        num_triplets = num_bands // 3
        
        for t in range(num_triplets):
            indices = [t*3, t*3+1, t*3+2]
            
            # Create RGB composites
            if inputs is not None:
                input_rgb = s2_thumbnail_torch(
                    inputs[:, indices[0]], inputs[:, indices[1]], inputs[:, indices[2]]
                )
            else:
                input_rgb = None
                
            recon_rgb = s2_thumbnail_torch(
                recons[:, indices[0]], recons[:, indices[1]], recons[:, indices[2]]
            )
            
            # Optional histogram row across these bands
            hist_row = None
            if view_histogram and inputs is not None:
                tile_h, tile_w, out_c = _get_tile_hw_and_channels(recon_rgb)
                hist_row = _generate_histogram_row_tensor(
                    inputs[:, indices], recons[:, indices], tile_h, tile_w, out_channels=out_c,
                    to_db=True if 'S1RTC' in satellite_type else False
                )

            # Save composite visualisation
            composite_path = str(vis_folder / f'composite_{indices[0]}_{indices[1]}_{indices[2]}.png')
            _save_comparison_image(
                input_rgb, recon_rgb, composite_path, n_images_to_log,
                view_difference=view_difference, extra_row=hist_row,
                resize=resize, input_frame_color=input_frame_color, recon_frame_color=recon_frame_color
            )
            results[f'composite_{t}'] = composite_path
        
        # Handle remaining bands if not divisible by 3
        _handle_remaining_s2_bands(inputs, recons, vis_folder, n_images_to_log, num_bands, results, view_difference=view_difference, view_histogram=view_histogram,
                                   resize=resize, input_frame_color=input_frame_color, recon_frame_color=recon_frame_color)
            
    elif 'S1RTC' in satellite_type and num_bands >= 2:
        # For Sentinel-1, create composite from VV and VH bands
        if inputs is not None:
            input_rgb = s1rtc_thumbnail_torch(inputs[:, 0], inputs[:, 1])
        else:
            input_rgb = None
            
        recon_rgb = s1rtc_thumbnail_torch(recons[:, 0], recons[:, 1])
        
        # Optional histogram row VV+VH
        hist_row = None
        if view_histogram and inputs is not None:
            tile_h, tile_w, out_c = _get_tile_hw_and_channels(recon_rgb)
            hist_row = _generate_histogram_row_tensor(
                inputs[:, 0:2], recons[:, 0:2], tile_h, tile_w, out_channels=out_c,
                to_db=True if 'S1RTC' in satellite_type else False
            )

        composite_path = str(vis_folder / f'composite_s1rtc.png')
        _save_comparison_image(
            input_rgb, recon_rgb, composite_path, n_images_to_log,
            view_difference=view_difference, extra_row=hist_row,
            resize=resize, input_frame_color=input_frame_color, recon_frame_color=recon_frame_color
        )
        results['composite_s1rtc'] = composite_path
    elif 'DEM' in satellite_type or 'cloud_mask' in satellite_type or 'LULC' in satellite_type or 'lat_lon' in satellite_type or 'mean_timestamps' in satellite_type:
        pass
    else:
        # raise ValueError(f"Visualisation composite, error: Unknown satellite type: {satellite_type}")
        print(f"Visualisation composite, error: Unknown satellite type: {satellite_type}")

def _handle_remaining_s2_bands(inputs, recons, vis_folder, n_images_to_log, num_bands, results, view_difference=False, view_histogram=False,
                               resize=None, input_frame_color=(0.0, 1.0, 0.0), recon_frame_color=(0.0, 0.0, 1.0)):
    """Handle remaining Sentinel-2 bands when not divisible by 3"""
    
    remaining = num_bands % 3
    if remaining > 0:
        # Determine which bands to use
        if remaining == 1:
            indices = [num_bands-1, 0, 1]  # Last band + first two bands
        elif remaining == 2:
            indices = [num_bands-2, num_bands-1, 0]  # Last two bands + first band
        
        # Create RGB composites
        if inputs is not None:
            input_rgb = s2_thumbnail_torch(
                inputs[:, indices[0]], inputs[:, indices[1]], inputs[:, indices[2]]
            )
        else:
            input_rgb = None
            
        recon_rgb = s2_thumbnail_torch(
            recons[:, indices[0]], recons[:, indices[1]], recons[:, indices[2]]
        )
        
        # Optional histogram row across these bands
        hist_row = None
        if view_histogram and inputs is not None:
            tile_h, tile_w, out_c = _get_tile_hw_and_channels(recon_rgb)
            hist_row = _generate_histogram_row_tensor(
                inputs[:, indices], recons[:, indices], tile_h, tile_w, out_channels=out_c
            )
        
        # Save composite visualisation
        composite_path = str(vis_folder / f'composite_remaining.png')
        _save_comparison_image(
            input_rgb, recon_rgb, composite_path, n_images_to_log,
            view_difference=view_difference, extra_row=hist_row,
            resize=resize, input_frame_color=input_frame_color, recon_frame_color=recon_frame_color
        )
        results['composite_remaining'] = composite_path

# def _create_main_visualisation(inputs, recons, save_dir, vis_folder, 
#                                 satellite_type, n_images_to_log, num_bands, milestone, results, view_difference=False, view_histogram=False):
#     """Create the main sample visualisation"""
    
#     # Select bands based on satellite type
#     if 'S2L2A' in satellite_type or 'S2L1C' in satellite_type:
#         if num_bands >= 3:
#             # Use RGB bands for Sentinel-2 (usually bands 4,3,2 for natural color)
#             rgb_indices = [3, 2, 1] if num_bands >= 4 else [2, 1, 0]
#         elif num_bands == 2:
#             rgb_indices = [1, 0, 0]
#         else:
#             rgb_indices = [0, 0, 0]
        
#         if inputs is not None:
#             input_vis = s2_thumbnail_torch(
#                 inputs[:, rgb_indices[0]], inputs[:, rgb_indices[1]], inputs[:, rgb_indices[2]]
#             )
#         else:
#             input_vis = None
            
#         recon_vis = s2_thumbnail_torch(
#             recons[:, rgb_indices[0]], recons[:, rgb_indices[1]], recons[:, rgb_indices[2]]
#         )
#     elif 'S1RTC' in satellite_type and num_bands >= 2:
#         # Use VV and VH bands for Sentinel-1
#         if inputs is not None:
#             input_vis = s1rtc_thumbnail_torch(inputs[:, 0], inputs[:, 1])
#         else:
#             input_vis = None
            
#         recon_vis = s1rtc_thumbnail_torch(recons[:, 0], recons[:, 1])
#     elif 'DEM' in satellite_type:
#         # Use hillshade visualisation for DEM
#         if inputs is not None:
#             input_vis = dem_thumbnail_torch(inputs[:, 0:1], hillshade=True)
#         else:
#             input_vis = None
            
#         recon_vis = dem_thumbnail_torch(recons[:, 0:1], hillshade=True)
#     elif 'LULC' in satellite_type or 'cloud_mask' in satellite_type:
#         if inputs is not None:
#             input_vis = classes_thumbnail_torch(inputs)
#         else:
#             input_vis = None
            
#         recon_vis = classes_thumbnail_torch(recons)
#     elif 'lat_lon' in satellite_type:
#         if inputs is not None:
#             input_vis = lat_lon_thumbnail_torch(inputs)
#         else:
#             input_vis = None
            
#         recon_vis = lat_lon_thumbnail_torch(recons)
#     elif 'mean_timestamps' in satellite_type:
#         if inputs is not None:
#             input_vis = date_thumbnail_torch(inputs)
#         else:
#             input_vis = None
            
#         recon_vis = date_thumbnail_torch(recons)
#     else:
#         print(f"Main visualisation error: Unknown satellite type: {satellite_type}")
#         # Fallback for unknown types or single-band data
#         if num_bands >= 3:
#             # Use first 3 bands as RGB
#             if inputs is not None:
#                 input_vis = normalize_tensor(inputs[:, :3])
#             else:
#                 input_vis = None
                
#             recon_vis = normalize_tensor(recons[:, :3])
#         else:
#             # Single-band visualisation
#             if inputs is not None:
#                 input_vis = normalize_tensor(inputs[:, 0:1])
#             else:
#                 input_vis = None
                
#             recon_vis = normalize_tensor(recons[:, 0:1])
            
#             # For single-channel data, repeat to create RGB
#             if recon_vis.shape[1] == 1:
#                 if input_vis is not None:
#                     input_vis = input_vis.repeat(1, 3, 1, 1)
#                 recon_vis = recon_vis.repeat(1, 3, 1, 1)
    
#     # Optional histogram row for the same chosen bands/modalities
#     hist_row = None
#     if view_histogram and inputs is not None:
#         tile_h, tile_w, out_c = _get_tile_hw_and_channels(recon_vis)
#         if 'S2L2A' in satellite_type or 'S2L1C' in satellite_type:
#             if num_bands >= 3:
#                 rgb_indices = [3, 2, 1] if num_bands >= 4 else [2, 1, 0]
#             elif num_bands == 2:
#                 rgb_indices = [1, 0]
#             else:
#                 rgb_indices = [0]
#             hist_row = _generate_histogram_row_tensor(
#                 inputs[:, rgb_indices], recons[:, rgb_indices], tile_h, tile_w, out_channels=out_c
#             )
#         elif 'S1RTC' in satellite_type and num_bands >= 2:
#             hist_row = _generate_histogram_row_tensor(
#                 inputs[:, 0:2], recons[:, 0:2], tile_h, tile_w, out_channels=out_c
#             )
#         else:
#             hist_row = _generate_histogram_row_tensor(
#                 inputs[:, 0:1], recons[:, 0:1], tile_h, tile_w, out_channels=out_c
#             )

#     # Save the main visualisation
#     combined_path = str(save_dir / (f'sample-{milestone}.png' if milestone else 'sample.png'))
#     _save_comparison_image(input_vis, recon_vis, combined_path, n_images_to_log, view_difference=view_difference, extra_row=hist_row)
#     results['combined'] = combined_path