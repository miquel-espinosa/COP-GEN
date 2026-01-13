import rasterio
from pathlib import Path

def check_file_format(file_path):
    """Check and print the format details of a raster file"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return
    
    try:
        with rasterio.open(file_path) as src:
            print(f"Checking file: {file_path}")
            print(f"File: {file_path.name}")
            print(f"Data type: {src.dtypes[0]}")
            print(f"Shape: {src.shape} (Height x Width)")
            print(f"Bands: {src.count}")
            print(f"CRS: {src.crs}")
            print(f"Transform: {src.transform}")
            print(f"Bounds: {src.bounds}")
            if src.nodata is not None:
                print(f"NoData value: {src.nodata}")
            
            # Read a small sample to show value ranges
            sample = src.read(1, window=((0, min(100, src.height)), (0, min(100, src.width))))
            print(f"Sample values range: {sample.min()} to {sample.max()}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    # Add your file paths here
    files_to_check = [
        # "/home/egm/Data/Projects/CopGen/data/cop-gen-small-test/Core-S2L2A/433U/433U_310R/S2B_MSIL2A_20200629T081609_N0500_R121_T36SYJ_20230505T211303/B01.tif",  
        # "/home/egm/Data/Projects/CopGen/data/cop-gen-small-test/Core-S2L2A/433U/433U_310R/S2B_MSIL2A_20200629T081609_N0500_R121_T36SYJ_20230505T211303/B02.tif",
        # "/home/egm/Data/Projects/CopGen/data/cop-gen-small-test/Core-S2L2A/433U/433U_310R/S2B_MSIL2A_20200629T081609_N0500_R121_T36SYJ_20230505T211303/B03.tif",
        # "/home/egm/Data/Projects/CopGen/data/cop-gen-small-test/Core-S2L2A/433U/433U_310R/S2B_MSIL2A_20200629T081609_N0500_R121_T36SYJ_20230505T211303/B04.tif",
        # "/home/egm/Data/Projects/CopGen/data/cop-gen-small-test/Core-S2L2A/433U/433U_310R/S2B_MSIL2A_20200629T081609_N0500_R121_T36SYJ_20230505T211303/B05.tif",
        # "/home/egm/Data/Projects/CopGen/data/cop-gen-small-test/Core-S2L2A/433U/433U_310R/S2B_MSIL2A_20200629T081609_N0500_R121_T36SYJ_20230505T211303/B06.tif",
        # "/home/egm/Data/Projects/CopGen/data/cop-gen-small-test/Core-S2L2A/433U/433U_310R/S2B_MSIL2A_20200629T081609_N0500_R121_T36SYJ_20230505T211303/B07.tif",
        # "/home/egm/Data/Projects/CopGen/data/cop-gen-small-test/Core-S2L2A/433U/433U_310R/S2B_MSIL2A_20200629T081609_N0500_R121_T36SYJ_20230505T211303/B08.tif",
        # "/home/egm/Data/Projects/CopGen/data/cop-gen-small-test/Core-S2L2A/433U/433U_310R/S2B_MSIL2A_20200629T081609_N0500_R121_T36SYJ_20230505T211303/B8A.tif",
        # "/home/egm/Data/Projects/CopGen/data/cop-gen-small-test/Core-S2L2A/433U/433U_310R/S2B_MSIL2A_20200629T081609_N0500_R121_T36SYJ_20230505T211303/B09.tif",
        # "/home/egm/Data/Projects/CopGen/data/cop-gen-small-test/Core-S2L2A/433U/433U_310R/S2B_MSIL2A_20200629T081609_N0500_R121_T36SYJ_20230505T211303/B11.tif",
        # "/home/egm/Data/Projects/CopGen/data/cop-gen-small-test/Core-S2L2A/433U/433U_310R/S2B_MSIL2A_20200629T081609_N0500_R121_T36SYJ_20230505T211303/B12.tif",
        # "/home/egm/Data/Projects/CopGen/data/input/S1RTC/433U_183R.tif",
        # "/home/egm/Data/Projects/CopGen/data/output/S1RTC/433U_183R.tif",
        # "/home/egm/Data/Projects/CopGen/data/input/S2L2A/433U_183R.tif",
        # "/home/egm/Data/Projects/CopGen/data/input/LULC/92U_5R_2020.tif",
        "/home/egm/Data/Projects/CopGen/data/output/S2L2A_from_DEM_LULC_S2L1C_S1RTC/92U_5R.tif",
    ]
    
    for file_path in files_to_check:
        check_file_format(file_path)