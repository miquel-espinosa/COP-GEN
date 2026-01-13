import os
import time
import pickle
import gzip
import io
import zipfile
import numpy as np
import zarr
from zarr.storage import MemoryStore  # MemoryStore is defined under zarr.storage
import blosc
from joblib import dump
import rasterio as rio

def size_mb(path):
    if isinstance(path, bytes):
        return len(path) / (1024 * 1024)
    if os.path.isdir(path):
        return sum(
            os.path.getsize(os.path.join(dirpath, f))
            for dirpath, _, files in os.walk(path)
            for f in files
        ) / (1024 * 1024)
    return os.path.getsize(path) / (1024 * 1024)

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        return time.time() - start, result
    return wrapper

@timeit
def save_pickle_raw(arr):
    with open('test.bytes', 'wb') as f:
        pickle.dump(arr, f, protocol=pickle.HIGHEST_PROTOCOL)
    return 'test.bytes'

@timeit
def save_pickle_gzip(arr, level=9):
    fname = f'gzip_compress_{level}_test.bytes.gz'
    with gzip.open(fname, 'wb', compresslevel=level) as f:
        pickle.dump(arr, f, protocol=pickle.HIGHEST_PROTOCOL)
    return fname

@timeit
def save_numpy_compressed(arr):
    fname = 'test_compressed.npz'
    np.savez_compressed(fname, arr=arr)
    return fname

@timeit
def save_joblib_zlib(arr):
    fname = 'test_array_zlib.joblib'
    dump(arr, fname, compress=('zlib', 3))
    return fname

@timeit
def save_joblib_lz4(arr):
    fname = 'test_array_lz4.joblib'
    dump(arr, fname, compress=('lz4', 3))
    return fname

@timeit
def save_zarr(arr):
    fname = 'array.zarr'
    zarr.save(fname, arr)
    return fname

@timeit
def save_geotiff(arr, profile):
    fname = 'compressed.tif'
    with rio.open(
        fname, 'w',
        driver='GTiff',
        height=arr.shape[1],
        width=arr.shape[2],
        count=arr.shape[0],
        dtype=arr.dtype,
        compress='lzw',
        crs=profile.get('crs'),
        transform=profile.get('transform')
    ) as dst:
        dst.write(arr)
    return fname

@timeit
def save_zarr_pickle_blob(arr):
    # Use an in-memory Zarr store; MemoryStore lives in the zarr.storage submodule
    store = MemoryStore()
    zarr.save(store, arr)
    blob = pickle.dumps(dict(store), protocol=pickle.HIGHEST_PROTOCOL)
    return blob  # In-memory binary blob

# -----------------------------------------------------------------------------
# Single-blob Zarr via an in-memory ZipStore
# -----------------------------------------------------------------------------

@timeit
def save_zarr_zip_blob(arr):
    """Serialize *arr* into a Zarr ZipStore that lives entirely in memory and write it to disk."""

    store = zarr.storage.MemoryStore()
    zarr.save(store, arr)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_STORED) as zf:
        for key, value in store._store_dict.items():
            zf.writestr(str(key), value.to_bytes())

    buffer.seek(0)

    # Persist the in-memory ZIP to disk so its size can be measured later.
    fname = "zarr_zip_blob.zip"
    with open(fname, "wb") as f:
        f.write(buffer.getvalue())

    return fname

def decode_zarr_zip_blob(fname):
    """Load a NumPy array back from the ZIP file written by *save_zarr_zip_blob*."""

    # Use Zarr's dedicated ZipStore for convenience – it provides a read-only
    # mapping interface over the contents of the ZIP archive.
    store = zarr.storage.ZipStore(fname, mode="r")
    try:
        arr = zarr.load(store)
    finally:
        store.close()

    return arr


@timeit
def save_blosc_raw_blob(arr):
    blob = blosc.compress(arr.tobytes(), typesize=4, cname='zstd', clevel=5)
    return blob

@timeit
def save_joblib_lz4_blob(arr):
    buffer = io.BytesIO()
    dump(arr, buffer, compress=('lz4', 3))
    return buffer.getvalue()

def benchmark_all():
    with rio.open('606D_453L_2020.tif') as src:
        arr = src.read()
        profile = src.profile

    print(f"\nArray shape: {arr.shape}, dtype: {arr.dtype}\n")

    results = []

    funcs = [
        ("pickle_raw", lambda: save_pickle_raw(arr)),
        ("pickle_gzip_lvl9", lambda: save_pickle_gzip(arr, level=9)),
        ("pickle_gzip_lvl1", lambda: save_pickle_gzip(arr, level=1)),
        ("np_savez_compressed", lambda: save_numpy_compressed(arr)),
        ("joblib_zlib", lambda: save_joblib_zlib(arr)),
        ("joblib_lz4", lambda: save_joblib_lz4(arr)),
        ("zarr", lambda: save_zarr(arr)),
        ("geotiff_lzw", lambda: save_geotiff(arr, profile)),
        # ("zarr_pickle_blob", lambda: save_zarr_pickle_blob(arr)),
        ("zarr_zip_blob", lambda: save_zarr_zip_blob(arr)),
        ("blosc_raw_blob", lambda: save_blosc_raw_blob(arr)),
        ("joblib_lz4_blob", lambda: save_joblib_lz4_blob(arr)),
    ]

    for label, func in funcs:
        duration, output = func()
        size = size_mb(output)
        results.append((label, duration, size))

    print("\n🧪 Benchmark Results:")
    print(f"{'Method':<22} {'Time (s)':>10} {'Size (MB)':>12}")
    print("-" * 46)
    for label, duration, size in sorted(results, key=lambda x: x[1]):
        print(f"{label:<22} {duration:>10.2f} {size:>12.1f}")

if __name__ == '__main__':
    benchmark_all()

