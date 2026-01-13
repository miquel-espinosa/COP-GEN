from pathlib import Path
import rioxarray as rxr
import numpy as np

def analyze_lulc_simple(lulc_dir, sample_size=100):
    """Simplified LULC class analysis"""
    print("Analyzing LULC classes in dataset...")
    
    lulc_path = Path(lulc_dir)
    tif_files = list(lulc_path.glob("*.tif"))[:sample_size]
    
    all_classes = set()
    class_counts = {}
    
    for lulc_file in tif_files:
        try:
            arr = rxr.open_rasterio(lulc_file).squeeze().values.astype(np.float32)
            unique_classes = np.unique(arr)
            all_classes.update(unique_classes)
            
            for cls in unique_classes:
                class_counts[cls] = class_counts.get(cls, 0) + np.sum(arr == cls)
                
        except Exception as e:
            print(f"Warning: Could not analyze {lulc_file}: {e}")
    
    sorted_classes = sorted(all_classes)
    
    print(f"\n=== LULC Class Analysis ===")
    print(f"Files analyzed: {len(tif_files)}")
    print(f"Total unique classes: {len(sorted_classes)}")
    print(f"Class range: {min(sorted_classes):.0f} to {max(sorted_classes):.0f}")
    print(f"All classes: {[int(cls) for cls in sorted_classes]}")
    
    return sorted_classes

# Usage:
lulc_classes = analyze_lulc_simple("/home/egm/Data/Projects/CopGen/data/input/LULC")