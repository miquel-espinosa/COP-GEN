import os
import glob
import shutil

def extract_thumbnails(src_root, dst_root):
    os.makedirs(dst_root, exist_ok=True)

    # find all thumbnail.png
    for thumb_path in glob.glob(f"{src_root}/**/thumbnail.png", recursive=True):
        # grid_id = folder above the product subfolder
        grid_id = os.path.basename(os.path.dirname(os.path.dirname(thumb_path)))

        dst_name = f"{grid_id}_thumbnail.png"
        shutil.copy2(thumb_path, os.path.join(dst_root, dst_name))

    print(f"Done extracting thumbnails for {src_root}")

# ---- RUN FOR EACH CORE ----

extract_thumbnails("Core-S1RTC", "Core-S1RTC-thumbnails")
extract_thumbnails("Core-DEM", "Core-DEM-thumbnails")
extract_thumbnails("Core-S2L2A", "Core-S2L2A-thumbnails")
extract_thumbnails("Core-S2L1C", "Core-S2L1C-thumbnails")
