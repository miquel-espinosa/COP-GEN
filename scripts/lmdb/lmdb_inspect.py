import lmdb
import pickle
import numpy as np
import argparse

latent_sizes = {
    "S2L2A_B02_B03_B04_B08": 24,
    "S2L2A_B05_B06_B07_B11_B12_B8A": 12,
    "S2L2A_B01_B09": 4,
    "S2L1C_B02_B03_B04_B08": 24,
    "S2L1C_B05_B06_B07_B11_B12_B8A": 12,
    "S2L1C_B01_B09_B10": 4,
    "S2L1C_cloud_mask": 24,
    "S1RTC_vh_vv": 24,
    "DEM_DEM": 8,
    "LULC_LULC": 24,
}


def main():
    parser = argparse.ArgumentParser(description="Inspect LMDB database contents")
    parser.add_argument("lmdb_path", help="Path to the LMDB database")
    parser.add_argument("--max-entries", type=int, default=3, help="Maximum number of entries to inspect (default: 3)")
    
    args = parser.parse_args()
    lmdb_path = args.lmdb_path

    env = lmdb.open(lmdb_path, readonly=True, lock=False)

    with env.begin() as txn:
        cursor = txn.cursor()
        for i, (key, value) in enumerate(cursor):
            print(f"\nEntry {i} - Key: {key.decode()}")

            try:
                obj = pickle.loads(value)

                for modality, byte_data in obj.items():
                    dtype = np.float32
                    latent = latent_sizes.get(modality)

                    if latent is None:
                        print(f"  ⚠️ Unknown modality '{modality}' — skipping")
                        continue

                    expected_shape = (16, latent, latent)
                    arr = np.frombuffer(byte_data, dtype=dtype)

                    if arr.size != np.prod(expected_shape):
                        print(f"  ❌ Shape mismatch for {modality}: expected {expected_shape}, got {arr.size} elements")
                        continue

                    arr = arr.reshape(expected_shape)
                    print(f"  ✅ {modality}: shape {arr.shape}, dtype {arr.dtype}")
                    print(f"    Preview: {arr.ravel()[:5]}")

            except Exception as e:
                print(f"  ❌ Failed to decode entry {i}: {e}")

            if i == args.max_entries - 1:
                break

    # Now, let's count the number of entries in the LMDB
    with env.begin() as txn:
        cursor = txn.cursor()
        count = 0
        for key, value in cursor:
            count += 1
    print(f"Number of entries in the LMDB: {count}")


if __name__ == "__main__":
    main()