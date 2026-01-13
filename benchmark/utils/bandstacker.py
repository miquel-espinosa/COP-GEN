from pathlib import Path
import rasterio
from rasterio.enums import Resampling
import numpy as np
import logging
from tqdm import tqdm
from rasterio.warp import transform
from json import dump
from collections import defaultdict
from benchmark.common.modalities import S2_BAND_ORDER
from ddm.pre_post_process_data import get_value_to_index

class BandStacker:
    def __init__(self, modality, root=None, target_resolution=10):
        self.modality = modality
        self.target_resolution = target_resolution
        self.root = Path(root) if root else Path(__file__).parent.parent.parent.resolve()
        self.input_root = self.root / f"Core-{modality}"
        self.output_root = self.root / "terramind_inputs" / modality
        self.output_root.mkdir(parents=True, exist_ok=True)
        if self.modality not in ["DEM", "LULC", "S1RTC", "S2L2A", "S2L1C", "coords"]:
            logging.error("mode must be 'DEM', 'LULC', 'S1RTC', 'S2L2A', 'S2L1C' or 'coords'")
            raise ValueError("mode must be 'DEM', 'LULC', 'S1RTC', 'S2L2A', 'S2L1C' or 'coords'")

    def _resample_to_target_resolution(self, src_path, target_transform, target_width, target_height):
        with rasterio.open(src_path) as src:
            data = src.read(
                out_shape=(src.count, target_height, target_width),
                resampling=Resampling.nearest
            )
            return data[0] if src.count == 1 else data

    def _get_target_resolution_params(self, reference_path):
        with rasterio.open(reference_path) as src:
            pixel_size_x = abs(src.transform[0])
            pixel_size_y = abs(src.transform[4])
            scale_x = pixel_size_x / self.target_resolution
            scale_y = pixel_size_y / self.target_resolution
            new_width = int(src.width * scale_x)
            new_height = int(src.height * scale_y)
            new_transform = rasterio.transform.from_bounds(
                *src.bounds, new_width, new_height
            )
            return new_transform, new_width, new_height

    def stack_all(self, max_files=None):
        if self.modality == "S1RTC":
            self._stack_s1(max_files)
        elif self.modality in ["S2L2A", "S2L1C"]:
            self._stack_s2(max_files)
        elif self.modality == "LULC":
            self._process_lulc(max_files)
        elif self.modality == "DEM":
            self._process_dem(max_files)
        elif self.modality == "coords":
            self._process_latlon()

    def _stack_s1(self, max_files=None):
        vv_paths = list(self.input_root.rglob("vv.tif"))
        if max_files is not None:
            vv_paths = vv_paths[:max_files]
        for vv_path in tqdm(vv_paths):
            vh_path = vv_path.parent / "vh.tif"
            if not vh_path.exists():
                continue
            with rasterio.open(vv_path) as vv_src, rasterio.open(vh_path) as vh_src:
                vv = vv_src.read(1)
                vh = vh_src.read(1)
                meta = vv_src.meta.copy()
                meta.update(count=2)
                stack = np.stack([vv, vh])
            tile_name = vv_path.parent.parent.name
            output_path = self.output_root / f"{tile_name}.tif"
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(stack)
                dst.update_tags(SOURCE=str(vv_path.parent.relative_to(self.input_root.parent.parent)))

    def _stack_s2(self, max_files=None):
        band_order = S2_BAND_ORDER[self.modality]

        folders = [f for f in self.input_root.rglob("*") if f.is_dir()]
        valid_folders = [f for f in folders if all((f / b).exists() for b in band_order)]
        if max_files is not None:
            valid_folders = valid_folders[:max_files]
        if not valid_folders:
            logging.error(f"No valid {self.modality} folders found!")
            return
        for folder in tqdm(valid_folders):
            band_paths = [folder / b for b in band_order]
            reference_bands = ["B02.tif", "B03.tif", "B04.tif", "B08.tif"]
            reference_path = next((folder / b for b in reference_bands if (folder / b).exists()), None)
            if reference_path is None:
                continue
            target_transform, target_width, target_height = self._get_target_resolution_params(reference_path)
            bands, meta = [], None
            for band_path in band_paths:
                band = self._resample_to_target_resolution(band_path, target_transform, target_width, target_height)
                bands.append(band)
                if meta is None:
                    with rasterio.open(band_path) as src:
                        meta = src.meta.copy()
            meta.update({
                'count': len(band_order),
                'transform': target_transform,
                'width': target_width,
                'height': target_height
            })
            stack = np.stack(bands)
            tile_name = folder.parent.name if folder.parent.name != f"Core-{self.modality}" else folder.name
            output_path = self.output_root / f"{tile_name}.tif"
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(stack)
                dst.update_tags(SOURCE=str(folder.relative_to(self.input_root.parent.parent)))

    def _process_lulc(self, max_files=None):
        lulc_paths = list(set([p for p in self.input_root.rglob("*.tif") if p.is_file()]))
        if max_files is not None:
            lulc_paths = lulc_paths[:max_files]
        if not lulc_paths:
            logging.error("No LULC files found!")
            return

        self._analyze_lulc_classes(lulc_paths, sample_size=None)
    
        for lulc_path in tqdm(lulc_paths):
            with rasterio.open(lulc_path) as src:
                pixel_size_x = abs(src.transform[0])
                pixel_size_y = abs(src.transform[4])
                current_resolution = min(pixel_size_x, pixel_size_y)
                if abs(current_resolution - self.target_resolution) > 0.1:
                    target_transform, target_width, target_height = self._get_target_resolution_params(lulc_path)
                    data = self._resample_to_target_resolution(lulc_path, target_transform, target_width, target_height)
                    meta = src.meta.copy()
                    meta.update({
                        'transform': target_transform,
                        'width': target_width,
                        'height': target_height
                    })
                else:
                    data = src.read(1)
                    meta = src.meta.copy()
         
            mapped_data = np.zeros_like(data, dtype=np.uint8)  # Map the classes and use uint8 for classes 0-9
            for original_class, new_class in get_value_to_index('LULC').items():
                mapped_data[data == original_class] = new_class
            
            # Log mapping for first few files
            if lulc_path in lulc_paths[:5]:
                original_classes = np.unique(data)
                mapped_classes = np.unique(mapped_data)
                logging.info(f"Class mapping example - Original: {original_classes} → Mapped: {mapped_classes}")

            output_filename = self._remove_year_suffix(lulc_path.name)
            output_path = self.output_root / output_filename
        
            meta.update(dtype='uint8', nodata=None)
            
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(mapped_data, 1)
                dst.update_tags(SOURCE=str(lulc_path.relative_to(self.input_root.parent.parent)))

    def _analyze_lulc_classes(self, lulc_paths, sample_size=None):
        """Analyze LULC classes in the original data"""
        logging.info("Analyzing LULC classes in original data...")
        
        all_classes = set()
        class_counts = defaultdict(int)
        files_analyzed = 0
        
        sample_paths = lulc_paths[:sample_size]
        
        for lulc_path in sample_paths:
            try:
                with rasterio.open(lulc_path) as src:
                    data = src.read(1)  # Read first band
                    unique_classes = np.unique(data)
                    all_classes.update(unique_classes)
                    
                    # Count pixels per class
                    for cls in unique_classes:
                        class_counts[cls] += np.sum(data == cls)
                    
                    files_analyzed += 1
                    
            except Exception as e:
                logging.warning(f"Could not analyze {lulc_path}: {e}")
        
        sorted_classes = sorted(all_classes)
        
        logging.info(f"=== LULC Class Analysis (from {files_analyzed} files) ===")
        logging.info(f"Total unique classes found: {len(sorted_classes)}")
        logging.info(f"Class range: {min(sorted_classes)} to {max(sorted_classes)}")
        logging.info(f"All classes: {[int(cls) for cls in sorted_classes]}")
        
        # Show most common classes
        logging.info("Class frequency (total pixels):")
        for cls in sorted(class_counts.keys()):
            count = class_counts[cls]
            logging.info(f"  Class {int(cls):2d}: {count:,} pixels")
        
        return sorted_classes

    def _process_dem(self, max_files=None):
        dem_paths = list(self.input_root.rglob("DEM.tif"))
        if max_files is not None:
            dem_paths = dem_paths[:max_files]
        if not dem_paths:
            logging.error("No DEM files found!")
            return
        for dem_path in tqdm(dem_paths):
            with rasterio.open(dem_path) as src:
                pixel_size_x = abs(src.transform[0])
                pixel_size_y = abs(src.transform[4])
                current_resolution = min(pixel_size_x, pixel_size_y)
                if abs(current_resolution - self.target_resolution) > 0.1:
                    target_transform, target_width, target_height = self._get_target_resolution_params(dem_path)
                    data = self._resample_to_target_resolution(dem_path, target_transform, target_width, target_height)
                    meta = src.meta.copy()
                    meta.update({
                        'transform': target_transform,
                        'width': target_width,
                        'height': target_height
                    })
                else:
                    data = src.read(1)
                    meta = src.meta.copy()
            tile_name = dem_path.parent.parent.name
            output_path = self.output_root / f"{tile_name}.tif"
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(data, 1)
                dst.update_tags(SOURCE=str(dem_path.relative_to(self.input_root.parent.parent)))

    def _process_latlon(self):
        tile_to_coords = {}
        input_path = self.root / "Core-LULC" # using LULC as reference
        if not input_path.exists():
            logging.error("No LULC files found! Cannot generate LATLON tokens without reference tiles.")
            return
        
        tif_paths = list(set([p for p in input_path.rglob("*.tif") if p.is_file()]))
        
        for tif in tqdm(tif_paths, desc="Generating lat/lon tokens and ground truth files"):
            # Use the helper function to get base tile name
            output_filename = self._remove_year_suffix(tif)
            tile_id = Path(output_filename).stem  # Get stem of the processed filename
            
            try:
                with rasterio.open(tif) as src:
                    bounds = src.bounds
                    center_x = (bounds.left + bounds.right) / 2
                    center_y = (bounds.top + bounds.bottom) / 2
                    lon, lat = transform(src.crs, "EPSG:4326", [center_x], [center_y])
                    lat_r = round(lat[0] * 4) / 4
                    lon_r = round(lon[0] * 4) / 4
                    token = f"lat={lat_r:.2f} lon={lon_r:.2f}"
                    
                    if tile_id not in tile_to_coords:
                        tile_to_coords[tile_id] = token
                    
                    self._create_coords_tif(tile_id, lat_r, lon_r)
                            
            except Exception as e:
                logging.warning(f"Failed on {tif}: {e}")
        
        out_json = self.output_root / "tile_to_coords.json"
        with open(out_json, "w") as f:
            dump(tile_to_coords, f, indent=2)
        logging.info(f"Wrote {len(tile_to_coords)} coordinate tokens to {out_json}")

    def _remove_year_suffix(self, filename):
        """Remove year suffix from filename if present"""
        base_name = Path(filename).stem
        parts = base_name.split('_')
        
        if len(parts) >= 3:
            last_part = parts[-1]
            if last_part.isdigit() and len(last_part) == 4 and last_part.startswith('20'):
                # Remove the year part
                base_name_no_year = '_'.join(parts[:-1])
                return f"{base_name_no_year}.tif"
        
        return Path(filename).name

    def _create_coords_tif(self, tile_name, lat, lon):
        """Create a .tif file for coordinates ground truth"""
        
        output_data = np.array([lat, lon], dtype=np.float32).reshape(2, 1, 1) # output data as [lat, lon]
        
        meta = {
            'driver': 'GTiff',
            'width': 1,
            'height': 1,
            'count': 2,
            'dtype': np.float32,
            'crs': 'EPSG:4326',
            'transform': rasterio.transform.from_bounds(-180, -90, 180, 90, 1, 1),
            'nodata': -9999.0
        }
        
        output_path = self.output_root / f"{tile_name}.tif"
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(output_data)
            dst.update_tags(SOURCE=f"Generated from LULC tile bounds")  

if __name__ == "__main__":
    BandStacker(modality="S1RTC").stack_all()