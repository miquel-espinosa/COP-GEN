import torch
import numpy as np
import logging
from visualisations.thumbnails_vis import (
    s1rtc_thumbnail_torch,
    s2_thumbnail_torch,
    dem_thumbnail_torch,
    classes_thumbnail_torch,
    lat_lon_thumbnail_torch
)
from benchmark.utils.plottingutils import (
    s1_to_rgb,
    s1_power_to_rgb,  
    s2_to_rgb, 
    dem_to_rgb,
    lulc_to_rgb,
    plot_modality
)
from ddm.pre_post_process_data import get_value_to_index, one_hot_decode

LULC_CLASS_NAMES = {
    0: "None/NoData",
    1: "Water",
    2: "Trees",
    3: "Flooded vegetation",
    4: "Crops",
    5: "Built Area",
    6: "Bare ground",
    7: "Snow/Ice",
    8: "Clouds",
    9: "Rangeland",
}

def render_output(modality, tensor: torch.Tensor):
    tensor = tensor.float()
    if modality == "S1RTC":
        return s1rtc_thumbnail_torch(tensor[:, 0], tensor[:, 1])
        # # Since your data is already in power format, use s1_power_to_rgb directly
        # numpy_data = tensor.squeeze(0).cpu().numpy()
        # rgb = s1_power_to_rgb(numpy_data)  # Use s1_power_to_rgb since your data is in power format
        # return torch.tensor(rgb).permute(2, 0, 1).float()
    elif modality in ["S2L2A", "S2L1C"]:
        return s2_thumbnail_torch(tensor[:, 3], tensor[:, 2], tensor[:, 1])
    elif modality == "DEM":
        return dem_thumbnail_torch(tensor[:, 0:1], hillshade=False)
        # rgb = dem_to_rgb(tensor[:, 0:1])
        # return torch.tensor(rgb).permute(2, 0, 1).float() / 255
    elif modality == "LULC":
        if tensor.shape[1] == 10:
            one_hot_decoded = one_hot_decode(tensor, value_to_index=get_value_to_index('LULC'))
            return classes_thumbnail_torch(one_hot_decoded, value_to_index=get_value_to_index('LULC'))
        else:
            # Remap data
            mapped_data = torch.zeros_like(tensor[:, 0:1], dtype=torch.int64)
            for orig_val, one_hot_idx in get_value_to_index('LULC').items():
                mapped_data[tensor[:, 0:1] == one_hot_idx] = torch.tensor(orig_val)
            return classes_thumbnail_torch(mapped_data, value_to_index=get_value_to_index('LULC'))
    # elif modality in ["coords"]:
    #     return lat_lon_thumbnail_torch(tensor)
    else:
        logging.error(f"Unsupported modality: {modality}")
        raise ValueError(f"Unsupported modality: {modality}")

def print_lulc_stats(label: str, arr: np.ndarray):
    if arr.ndim == 3 and arr.shape[0] == 10:
        class_data = arr.argmax(axis=0)
    elif arr.ndim == 2:
        class_data = arr
    else:
        class_data = arr[0] if arr.ndim > 2 else arr

    unique_vals, counts = np.unique(class_data, return_counts=True)
    total_pixels = class_data.size

    top_indices = np.argsort(counts)[::-1][:3]
    logging.info(f"{label} LULC class distribution (top 3):")
    for i, idx in enumerate(top_indices, 1):
        class_id = int(unique_vals[idx])
        percentage = (counts[idx] / total_pixels) * 100
        class_name = LULC_CLASS_NAMES.get(class_id, f"Unknown {class_id}")
        logging.info(f"  {i}. Class {class_id} ({class_name}): {percentage:.1f}%")
