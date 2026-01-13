from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import torch
import math
import random
from PIL import Image
import os
import glob
import einops
import torchvision.transforms.functional as F
import pickle
import lmdb
import math
import collections, time
import json


class _LatLonMemmapCache:
    """Memory-mapped reader for precomputed lat/lon cache written as an npy directory.

    Directory layout (produced by scripts/precompute_lat_lon.py with --storage=npy_dir):
        - cells.npy            (N,) array of Unicode cell labels
        - latlon_cells.npy     (N, D) float32, per-cell centres
        - latlon_patches.npy   (N, S, S, D) float32, per-patch centres
        - meta.json            metadata (side, format, vec_dim, ...)

    Access pattern mirrors the old dict[(cell, pr, pc)] interface via __getitem__.
    """

    def __init__(self, root_dir: str):
        cells_path = os.path.join(root_dir, "cells.npy")
        cells = np.load(cells_path, mmap_mode="r")
        # Keep a mapping for O(1) index lookup
        self._cell_to_idx = {str(cells[i]): int(i) for i in range(cells.shape[0])}
        self._cells = cells  # keep reference to avoid GC

        self._cells_centres = np.load(os.path.join(root_dir, "latlon_cells.npy"), mmap_mode="r")
        self._patch_centres = np.load(os.path.join(root_dir, "latlon_patches.npy"), mmap_mode="r")

        with open(os.path.join(root_dir, "meta.json"), "r") as f:
            self._meta = json.load(f)

        self.side = int(self._meta.get("side"))
        self.vec_dim = int(self._meta.get("vec_dim"))

    def __getitem__(self, key):
        cell, pr, pc = key
        idx = self._cell_to_idx[cell]
        if pr == -1 and pc == -1:
            # return torch.from_numpy(self._cells_centres[idx]).float()
            return self._cells_centres[idx]
        return self._patch_centres[idx, pr, pc]
        # return torch.from_numpy(self._patch_centres[idx, pr, pc]).float()

    def __contains__(self, key):
        cell, pr, pc = key
        if cell not in self._cell_to_idx:
            return False
        if pr == -1 and pc == -1:
            return True
        s = self.side
        return 0 <= pr < s and 0 <= pc < s


class UnlabeledDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        # data = tuple(self.dataset[item][:-1])  # remove label
        data = self.dataset[item]
        if len(data) == 1:
            data = data[0]
        return data


class LabeledDataset(Dataset):
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item], self.labels[item]


class CFGDataset(Dataset):  # for classifier free guidance
    def __init__(self, dataset, p_uncond, empty_token):
        self.dataset = dataset
        self.p_uncond = p_uncond
        self.empty_token = empty_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x, y = self.dataset[item]
        if random.random() < self.p_uncond:
            y = self.empty_token
        return x, y


class DatasetFactory(object):

    def __init__(self):
        self.train = None
        self.test = None

    def get_split(self, split, labeled=False, nosplit=False):
        if nosplit:
            return self.dataset
        if split == "train":
            dataset = self.train
        elif split == "test":
            dataset = self.test
        else:
            raise ValueError

        if self.has_label:
            return dataset if labeled else UnlabeledDataset(dataset)
        else:
            assert not labeled
            return dataset

    def unpreprocess(self, v):  # to B C H W and [0, 1]
        v = 0.5 * (v + 1.)
        v.clamp_(0., 1.)
        return v

    @property
    def has_label(self):
        return True

    @property
    def data_shape(self):
        raise NotImplementedError

    @property
    def data_dim(self):
        return int(np.prod(self.data_shape))

    @property
    def fid_stat(self):
        return None

    def sample_label(self, n_samples, device):
        raise NotImplementedError

    def label_prob(self, k):
        raise NotImplementedError


# CIFAR10

class CIFAR10(DatasetFactory):
    r""" CIFAR10 dataset

    Information of the raw dataset:
         train: 50,000
         test:  10,000
         shape: 3 * 32 * 32
    """

    def __init__(self, path, random_flip=False, cfg=False, p_uncond=None):
        super().__init__()

        transform_train = [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
        transform_test = [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
        if random_flip:  # only for train
            transform_train.append(transforms.RandomHorizontalFlip())
        transform_train = transforms.Compose(transform_train)
        transform_test = transforms.Compose(transform_test)
        self.train = datasets.CIFAR10(path, train=True, transform=transform_train, download=True)
        self.test = datasets.CIFAR10(path, train=False, transform=transform_test, download=True)

        assert len(self.train.targets) == 50000
        self.K = max(self.train.targets) + 1
        self.cnt = torch.tensor([len(np.where(np.array(self.train.targets) == k)[0]) for k in range(self.K)]).float()
        self.frac = [self.cnt[k] / 50000 for k in range(self.K)]
        print(f'{self.K} classes')
        print(f'cnt: {self.cnt}')
        print(f'frac: {self.frac}')

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
            self.train = CFGDataset(self.train, p_uncond, self.K)

    @property
    def data_shape(self):
        return 3, 32, 32

    @property
    def fid_stat(self):
        return 'assets/fid_stats/fid_stats_cifar10_train_pytorch.npz'

    def sample_label(self, n_samples, device):
        return torch.multinomial(self.cnt, n_samples, replacement=True).to(device)

    def label_prob(self, k):
        return self.frac[k]


# ImageNet


class FeatureDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        # names = sorted(os.listdir(path))
        # self.files = [os.path.join(path, name) for name in names]

    def __len__(self):
        return 1_281_167 * 2  # consider the random flip

    def __getitem__(self, idx):
        path = os.path.join(self.path, f'{idx}.npy')
        z, label = np.load(path, allow_pickle=True)
        return z, label


class MajorTOM_S2_FeatureDataset(Dataset):

    def __init__(self, path, transform=None):
        super().__init__()
        self.path = path
        self.transform = transform
        # names = sorted(os.listdir(path))
        # self.files = [os.path.join(path, name) for name in names]

    def __len__(self):
        return len(glob.glob(f"{self.path}/*.npy"))

    def __getitem__(self, idx):
        path = os.path.join(self.path, f"{idx}.npy")
        moment = np.load(path, allow_pickle=True).copy()
        if self.transform is not None:
            moment = self.transform(moment)
        return moment


class MajorTOM_Tuples_FeatureDataset(Dataset):

    def __init__(self, paths, transform=None):
        super().__init__()
        self.paths = paths
        self.transform = transform
        print(f"Gathering filenames...")
        self.filenames = [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"{self.paths[0]}/*.npy")]
        print(f"Found {len(self.filenames)} filenames across all paths")
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Return npy files for each modality. Always in the same order
        moments = []
        for path in self.paths:
            path = os.path.join(path, f"{self.filenames[idx]}.npy")
            moment = np.load(path, allow_pickle=True).copy()
            if self.transform is not None:
                moment = self.transform(moment)
            moments.append(moment)
        return moments


class MajorTOM_S2_Features(DatasetFactory):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print("Prepare dataset...")
        # transform_train = [transforms.ToTensor()]
        transform_train = []
        self.train = MajorTOM_S2_FeatureDataset(
            path, transform=transforms.Compose(transform_train)
        )
        self.path = path
        print("Prepare dataset ok")
        self.K = 1000

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f"prepare the dataset for classifier free guidance with p_uncond={p_uncond}")
            self.train = CFGDataset(self.train, p_uncond, self.K)

    def get_split(self, split, labeled=False):
        if split == "train":
            dataset = self.train
        elif split == "test":
            dataset = self.test
        else:
            raise ValueError

        if self.has_label:
            return dataset if labeled else UnlabeledDataset(dataset)
        else:
            assert not labeled
            return dataset

    @property
    def data_shape(self):
        return 4, 133, 133

    @property
    def fid_stat(self):
        return f"assets/fid_stats/fid_stats_imagenet256_guided_diffusion.npz"

    def sample_label(self, n_samples, device):
        return torch.randint(0, 1000, (n_samples,), device=device)


class MajorTOM_Tuples_Features(DatasetFactory):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, paths, cfg=False, p_uncond=None):
        super().__init__()
        print("Prepare dataset...")
        # transform_train = [transforms.ToTensor()]
        transform_train = []
        self.train = MajorTOM_Tuples_FeatureDataset(
            paths, transform=transforms.Compose(transform_train)
        )
        self.paths = paths
        print("Prepare dataset ok")
        self.K = 1000

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f"prepare the dataset for classifier free guidance with p_uncond={p_uncond}")
            self.train = CFGDataset(self.train, p_uncond, self.K)

    def get_split(self, split, labeled=False):
        if split == "train":
            dataset = self.train
        elif split == "test":
            dataset = self.test
        else:
            raise ValueError

        if self.has_label:
            return dataset if labeled else UnlabeledDataset(dataset)
        else:
            assert not labeled
            return dataset

    @property
    def data_shape(self):
        return "blablabla"

    @property
    def fid_stat(self):
        return f"assets/fid_stats/fid_stats_imagenet256_guided_diffusion.npz"

    def sample_label(self, n_samples, device):
        raise NotImplementedError
        return torch.randint(0, 1000, (n_samples,), device=device)
    
    
class MajorTOM_Lmdb_FeatureDataset(Dataset):
    def __init__(self, path, transform=None, return_filename=False):
        super().__init__()
        
        self.transform = transform
        self.path = path  # Store the path instead of the environment
        self.return_filename = return_filename
        
        # Create a temporary environment just to get the stats and keys
        env = lmdb.open(
            path,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        
        # Get total number of entries
        with env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]
            
        # Load or create cache of keys
        root_split = path.split("/")
        cache_file = os.path.join("/".join(root_split[:-1]), f"_cache_{root_split[-1]}")
        if os.path.isfile(cache_file):
            # Fast load: read raw bytes and split on newline (much quicker than unpickling)
            with open(cache_file, "rb") as f:
                self.keys = f.read().split(b"\n")
        else:
            with env.begin(write=False) as txn:
                cursor = txn.cursor()
                # key-only iteration is ~2-3× faster than fetching values as well
                self.keys = [k for k in cursor.iternext(values=False)]
            # Store as a single byte blob to avoid pickle per-object overhead
            with open(cache_file, "wb") as f:
                f.write(b"\n".join(self.keys))
            
        # Close the temporary environment
        env.close()
        
        # Create environment lazily in each worker
        self._env = None

    def _init_db(self):
        """Initialize LMDB environment"""
        import lmdb
        self._env = lmdb.open(
            self.path,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        
    @property
    def env(self):
        """Get LMDB environment, creating it if necessary"""
        if self._env is None:
            self._init_db()
        return self._env

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Get data from LMDB
        
        key = self.keys[idx]
        filename = key.decode('utf-8') if isinstance(key, bytes) else key
        filename = os.path.basename(filename) # get filename without path
        filename = os.path.splitext(filename)[0] # remove .npy extension
        
        with self.env.begin(write=False) as txn:
            data = pickle.loads(txn.get(key))
            
        # Convert bytes to data for each modality
        decoded_data = {}
        for k, bytes_data in data.items():
            # Convert bytes back to numpy array with the expected shape (8, 32, 32).
            # TODO: This is currently hardcoded.
            features = np.frombuffer(bytes_data, dtype=np.float32).reshape(8, 32, 32).copy()
            decoded_data[k] = features
            
        # Apply transforms if any
        if self.transform is not None:
            decoded_data = {k: self.transform(v) for k, v in decoded_data.items()}
            
        # Convert the dictionary values to a list in a consistent order
        moments = [decoded_data[k] for k in sorted(decoded_data.keys())]
        
        if self.return_filename:
            return moments, filename
        return moments
        
    def __del__(self):
        if self._env is not None:
            self._env.close()


class MajorTOM_Lmdb_Features(DatasetFactory):
    def __init__(self, path, cfg=False, p_uncond=None, return_filename=False):
        super().__init__()
        print("Prepare dataset...")
        transform_train = []
        self.return_filename = return_filename
        self.train = MajorTOM_Lmdb_FeatureDataset(
            path, transform=transforms.Compose(transform_train), return_filename=return_filename
        )
        self.path = path
        print("Prepare dataset ok")
        self.K = 1000

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f"prepare the dataset for classifier free guidance with p_uncond={p_uncond}")
            self.train = CFGDataset(self.train, p_uncond, self.K)

    def get_split(self, split, labeled=False):
        if split == "train":
            dataset = self.train
        elif split == "test":
            dataset = self.test
        else:
            raise ValueError

        if self.has_label:
            return dataset if labeled else UnlabeledDataset(dataset)
        else:
            assert not labeled
            return dataset

    @property
    def data_shape(self):
        return "blablabla"

    @property
    def fid_stat(self):
        return f"assets/fid_stats/fid_stats_imagenet256_guided_diffusion.npz"

    def sample_label(self, n_samples, device):
        raise NotImplementedError
        return torch.randint(0, 1000, (n_samples,), device=device)


class COPGEN_LMDB_FeatureDataset(Dataset):
    def __init__(self, path, transform=None, return_filename=False, z_input_shapes=None,
                 crop_shapes=None, patches_size=None, random_flip=False,
                 patches_per_side: int | None = None,
                 lat_lon_encode: str | None = None,
                 precomputed_lat_lon_path: str | None = None,
                 time_encode: str | None = None,
                 precomputed_mean_timestamps_path: str | None = None,
                 min_db: dict | None = None,
                 max_db: dict | None = None,
                 min_positive: dict | None = None,
                 normalize_to_neg_one_to_one: bool = False):
        super().__init__()
        # Lat-lon encoding / patch-grid parameters
        self.patches_per_side = patches_per_side
        self.lat_lon_encode = lat_lon_encode
        self.precomputed_mean_timestamps_path = precomputed_mean_timestamps_path
        self.time_encode = time_encode
        self._mean_timestamps_cache = None
        
        print(f"Loading precomputed mean timestamps cache...")
        if self.time_encode:
            if not os.path.exists(self.precomputed_mean_timestamps_path):
                raise FileNotFoundError(
                    f"Pre-computed mean timestamps cache not found at '{self.precomputed_mean_timestamps_path}'. "
                    "Generate it with precompute_time_mean.py before training."
                )
            with open(self.precomputed_mean_timestamps_path, "rb") as f:
                self._mean_timestamps_cache = pickle.load(f)
        
        print(f"Precomputed mean timestamps cache loaded")
        print(f"Loading precomputed lat/lon cache...")
        if self.lat_lon_encode:
            if not os.path.exists(precomputed_lat_lon_path):
                raise FileNotFoundError(
                    f"Pre-computed lat/lon cache not found at '{precomputed_lat_lon_path}'. "
                    "Generate it with scripts/precompute_lat_lon.py before training."
                )
            # Fast path: directory with npy files (memmap)
            if os.path.isdir(precomputed_lat_lon_path) and \
               os.path.exists(os.path.join(precomputed_lat_lon_path, "cells.npy")) and \
               os.path.exists(os.path.join(precomputed_lat_lon_path, "latlon_cells.npy")) and \
               os.path.exists(os.path.join(precomputed_lat_lon_path, "latlon_patches.npy")):
                self._lat_lon_cache = _LatLonMemmapCache(precomputed_lat_lon_path)
            else:
                # Fallback to pickle (backward compatible)
                with open(precomputed_lat_lon_path, "rb") as f:
                    self._lat_lon_cache = pickle.load(f)
        print(f"Precomputed lat/lon cache loaded")
        
        self.transform = transform
        self.path = path  # Store the path instead of the environment
        self.return_filename = return_filename
        self.z_input_shapes = z_input_shapes
        self.crop_shapes = crop_shapes
        self.random_flip = random_flip
        self.patches_size = patches_size
        
        print(f"Creating temporary LMDB environment...")
        
        # Create a temporary environment just to get the stats and keys
        env = lmdb.open(
            path,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        
        print(f"Opened LMDB environment at {path}")
        print(f"Getting total number of entries...")
        
        # Get total number of entries
        with env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]
         
        print(f"Total number of entries: {self.length}")
         
        print(f"Loading or creating cache of keys...")
        # Load or create cache of keys
        root_split = path.split("/")
        cache_file = os.path.join("/".join(root_split[:-1]), f"_cache_{root_split[-1]}")
        if os.path.isfile(cache_file):
            # Fast load: read raw bytes and split on newline instead of unpickling
            with open(cache_file, "rb") as f:
                self.keys = f.read().split(b"\n")
        else:
            with env.begin(write=False) as txn:
                cursor = txn.cursor()
                # key-only iteration avoids reading the values, giving a big speed-up
                self.keys = [k for k in cursor.iternext(values=False)]
            # Save as a single byte blob (newline-delimited) – much cheaper than pickle
            with open(cache_file, "wb") as f:
                f.write(b"\n".join(self.keys))
        print(f"Cache of keys loaded or created")
        
        # Close the temporary environment
        env.close()
        
        # Create environment lazily in each worker
        self._env = None

        # self._profile = collections.defaultdict(float)  # holds cumulated timings
        # self._profile['cnt'] = 0                       # number of calls

    def _init_db(self):
        """Initialize LMDB environment"""
        import lmdb
        print(f"[LMDB INIT] Process {os.getpid()} initializing LMDB at {self.path}")
        self._env = lmdb.open(
            self.path,
            max_readers=32,
            # max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        
    @property
    def env(self):
        """Get LMDB environment, creating it if necessary"""
        if self._env is None:
            self._init_db()
        return self._env

    def __len__(self):
        return self.length

    def _random_cropping(self, decoded_data):
        """
        Perform random cropping on all modalities, maintaining spatial correspondence.
        
        Args:
            decoded_data: Dictionary of modality features with their original shapes
            
        Returns:
            Dictionary of modality features with cropped shapes
        """
        # Calculate scaling ratios between modalities
        modality_keys = self.z_input_shapes.keys()
        
        # Find the smallest modality to use as reference
        ref_key = min(modality_keys, key=lambda k: self.z_input_shapes[k][1])
        ref_input_shape = self.z_input_shapes[ref_key]
        ref_crop_shape = self.crop_shapes[ref_key]
        
        # Calculate scale factors between modalities
        scale_factors = {k: self.z_input_shapes[k][1] // ref_input_shape[1] 
                         for k in modality_keys}
        
        for k in modality_keys: # CHECK
            if scale_factors[k] < 1: # Check that the scale factor is greater or equal to 1
                raise ValueError(f"Scale factor for modality {k} is not greater than 1")
            if scale_factors[k] % 1 != 0: # Check that the scale factor is an integer
                raise ValueError(f"Scale factor for modality {k} is not an integer")
        
        # Select a random position for the reference modality
        max_ref_h_start = ref_input_shape[1] - ref_crop_shape[1]
        max_ref_v_start = ref_input_shape[2] - ref_crop_shape[2]
        ref_h_start = random.randint(0, max_ref_h_start)
        ref_v_start = random.randint(0, max_ref_v_start)
        
        # Apply crop to each modality using the scale factors
        for k in modality_keys:
            scale = scale_factors[k]
            crop_shape = self.crop_shapes[k]
            
            # Scale the reference position for this modality
            h_start = ref_h_start * scale
            v_start = ref_v_start * scale
            
            # Crop the data
            h_end = h_start + crop_shape[1]
            v_end = v_start + crop_shape[2]
            decoded_data[k] = decoded_data[k][:, h_start:h_end, v_start:v_end]
        
        return decoded_data

    def _random_flipping(self, decoded_data):
        """
        Perform random horizontal flipping on all modalities, with 50% probability.
        
        Args:
            decoded_data: Dictionary of modality features
            
        Returns:
            Dictionary of modality features, potentially flipped
        """
        if not self.random_flip or random.random() >= 0.5:
            return decoded_data
            
        # Apply horizontal flip to all modalities (flip the last dimension)
        for k in decoded_data.keys():
            # Flip along the last dimension (width)
            decoded_data[k] = np.flip(decoded_data[k], axis=2).copy()
            
        return decoded_data
    
    def _get_row_col_patch(self, filename):
        # Parse grid-cell / patch information from filename
        grid_cell = filename
        patch_row = patch_col = None
        if "_patch_" in filename:
            grid_cell, patch_str = filename.split("_patch_")
            try:
                patch_id = int(patch_str)
            except ValueError:
                patch_id = None
            if patch_id is not None and self.patches_per_side is not None:
                patch_row = patch_id // self.patches_per_side
                patch_col = patch_id % self.patches_per_side
        
        return grid_cell, patch_row, patch_col

    def _to_torch_tensor(self, data):
        """Convert various data types to torch tensor."""
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).float()
        elif isinstance(data, (list, tuple)):
            return torch.tensor(data, dtype=torch.float32)
        else:
            return data  # Assume it's already a torch tensor

    def _add_encoded_feature(self, moments, cache, cache_key, feature_name, error_msg_prefix, add_spatial_dims=True):
        """Helper method to add encoded features from cache to moments."""
        try:
            feature_data = cache[cache_key]
            feature_tensor = self._to_torch_tensor(feature_data)
            if add_spatial_dims:
                feature_tensor = feature_tensor.unsqueeze(0).unsqueeze(0)
            moments[feature_name] = feature_tensor
        except KeyError as e:
            raise KeyError(f"{error_msg_prefix} for key {cache_key} not in cache") from e

    # def _get_lat_lon(self, grid_cell, patch_row, patch_col):
    #     key = (grid_cell, patch_row, patch_col)
    #     if key in self._patch_center_cache:
    #         return self._patch_center_cache[key]

    #     # --------- Get bottom-left, top-right bounds of the full grid cell
    #     if grid_cell not in self._cell_bounds_cache:
    #         try:
    #             row_label, col_label = grid_cell.split("_")
    #             # bottom-left point geometry
    #             point_gdf = self._grid.points[(self._grid.points.row == row_label) & (self._grid.points.col == col_label)]
    #             if len(point_gdf) == 0:
    #                 raise ValueError(f"Grid labels not found: {grid_cell}")
    #             point = point_gdf.iloc[0]
    #             poly = self._grid.get_bounded_footprint(point, buffer_ratio=0)
    #             minx, miny, maxx, maxy = poly.bounds  # lon_min, lat_min, lon_max, lat_max
    #             self._cell_bounds_cache[grid_cell] = (minx, miny, maxx, maxy)
    #         except Exception as e:
    #             raise RuntimeError(f"Failed to compute bounds for grid cell {grid_cell}: {e}")

    #     minx, miny, maxx, maxy = self._cell_bounds_cache[grid_cell]

    #     # --------- If patch is provided, refine to that patch
    #     if patch_row is not None and patch_col is not None:
    #         # Determine physical patch size (degrees) in this grid cell
    #         cell_width = maxx - minx
    #         cell_height = maxy - miny

    #         if self.patches_size is not None:
    #             # Explicit patch size (in same units as cell bounds)
    #             patch_width, patch_height = self.patches_size
    #         else:
    #             raise ValueError("patches_size must be provided if patches_per_side is provided")

    #         # Horizontal placement – right-align the final patch like _extract_patch
    #         if patch_col == self.patches_per_side - 1:
    #             patch_minx = maxx - patch_width
    #         else:
    #             patch_minx = minx + patch_col * patch_width
    #         patch_maxx = patch_minx + patch_width

    #         # Vertical placement – top-origin (row 0 = northern-most)
    #         if patch_row == self.patches_per_side - 1:
    #             patch_miny = miny  # bottom-align last row
    #         else:
    #             patch_miny = maxy - (patch_row + 1) * patch_height
    #         patch_maxy = patch_miny + patch_height
    #     else:
    #         # Center of full grid cell
    #         lon = (minx + maxx) / 2.0
    #         lat = (miny + maxy) / 2.0
        
    #     self._patch_center_cache[key] = (lat, lon)
    #     return lat, lon

    def __getitem__(self, idx):
        # Get data from LMDB
        key = self.keys[idx]
        filename = key.decode('utf-8') if isinstance(key, bytes) else key
        filename = os.path.basename(filename) # get filename without path
        filename = os.path.splitext(filename)[0] # remove .npy extension
        
        with self.env.begin(write=False) as txn:
            data = pickle.loads(txn.get(key))
            
        # Convert bytes to data for each modality
        decoded_data = {}
        for k, bytes_data in data.items():
            # features = np.frombuffer(bytes_data, dtype=np.float32).reshape(self.z_input_shapes[k]).copy()
            # decoded_data[k] = features
            decoded_data[k] = np.frombuffer(bytes_data, dtype=np.float32).reshape(self.z_input_shapes[k])
            
        # Sort in the order of the config file. This is important for the order of the modalities in the network.
        decoded_data = {modality: decoded_data[modality] for modality in self.z_input_shapes.keys() if modality in decoded_data}

        # Perform random cropping
        if self.crop_shapes is not None:
            decoded_data = self._random_cropping(decoded_data)
        
        # Perform random flipping
        decoded_data = self._random_flipping(decoded_data)
                       
        # Apply transforms if any
        if self.transform is not None:
            decoded_data = {k: self.transform(v) for k, v in decoded_data.items()}
            
        # # Convert the dictionary values to a list in a consistent order
        # moments = [decoded_data[k] for k in sorted(decoded_data.keys())]
        moments = decoded_data
        
        if self.lat_lon_encode:
            # t0 = time.perf_counter()
            grid_cell, patch_row, patch_col = self._get_row_col_patch(filename)
            if patch_row is None:
                patch_row = -1
            if patch_col is None:
                patch_col = -1
            # t1 = time.perf_counter()
            
            cache_key = (grid_cell, patch_row, patch_col)
            self._add_encoded_feature(
                moments=moments,
                cache=self._lat_lon_cache, 
                cache_key=cache_key, 
                feature_name=self.lat_lon_encode, 
                error_msg_prefix="Lat/lon"
            )
            
            # lat, lon = self._get_lat_lon(grid_cell, patch_row, patch_col)
            # t2 = time.perf_counter()

            # --- accumulate profile info
            # self._profile['row_col'] += (t1 - t0)
            # self._profile['lat_lon'] += (t2 - t1)
            # self._profile['cnt']    += 1

            # dump every 1 000 samples
            # if self._profile['cnt'] % 5 == 0:
            #     n = self._profile['cnt']
            #     print(
            #         f"[lat-lon profile] n={n}  "
            #         f"row_col={self._profile['row_col']/n:.6f}s  "
            #         f"lat_lon={self._profile['lat_lon']/n:.6f}s  "
            #     )
        
        if self.time_encode:
            grid_cell, patch_row, patch_col = self._get_row_col_patch(filename)
            
            self._add_encoded_feature(
                moments=moments,
                cache=self._mean_timestamps_cache,
                cache_key=grid_cell,
                feature_name=self.time_encode,
                error_msg_prefix="Mean timestamps"
            )
            
        # Final sort of moments in the order of the config file. This is important for the order of the modalities in the network.
        moments = {modality: moments[modality] for modality in self.z_input_shapes.keys() if modality in moments}
        
        if self.return_filename:
            return moments, filename
        return moments
        
    def __del__(self):
        if self._env is not None:
            self._env.close()


class COPGEN_Features(DatasetFactory):
    def __init__(self, path, cfg=False, p_uncond=None, return_filename=False, z_input_shapes=None, crop_shapes=None, patches_size=None,
                 random_flip=False, patches_per_side: int | None = None, lat_lon_encode: str | None = None,
                 precomputed_lat_lon_path: str | None = None,
                 time_encode: str | None = None,
                 precomputed_mean_timestamps_path: str | None = None,
                 min_db: dict | None = None,
                 max_db: dict | None = None,
                 min_positive: dict | None = None,
                 normalize_to_neg_one_to_one: bool = False):
        super().__init__()
        print("Prepare dataset...")
        transform_train = []
        self.return_filename = return_filename
        self.train = COPGEN_LMDB_FeatureDataset(
            path,
            transform=transforms.Compose(transform_train),
            return_filename=return_filename,
            z_input_shapes=z_input_shapes,
            crop_shapes=crop_shapes,
            patches_size=patches_size,
            random_flip=random_flip,
            patches_per_side=patches_per_side,
            lat_lon_encode=lat_lon_encode,
            precomputed_lat_lon_path=precomputed_lat_lon_path,
            time_encode=time_encode,
            precomputed_mean_timestamps_path=precomputed_mean_timestamps_path,
            min_db=min_db,
            max_db=max_db,
            min_positive=min_positive,
            normalize_to_neg_one_to_one=normalize_to_neg_one_to_one,
        )
        self.path = path
        print("Prepare dataset ok")
        self.K = 1000

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f"prepare the dataset for classifier free guidance with p_uncond={p_uncond}")
            self.train = CFGDataset(self.train, p_uncond, self.K)

    def get_split(self, split, labeled=False):
        if split == "train":
            dataset = self.train
        elif split == "test":
            dataset = self.test
        else:
            raise ValueError

        if self.has_label:
            return dataset if labeled else UnlabeledDataset(dataset)
        else:
            assert not labeled
            return dataset

    @property
    def data_shape(self):
        raise NotImplementedError
        return "blablabla"

    @property
    def fid_stat(self):
        return f"assets/fid_stats/fid_stats_imagenet256_guided_diffusion.npz"

    def sample_label(self, n_samples, device):
        raise NotImplementedError
        return torch.randint(0, 1000, (n_samples,), device=device)


class ImageNet256Features(DatasetFactory):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print('Prepare dataset...')
        self.train = FeatureDataset(path)
        print('Prepare dataset ok')
        self.K = 1000

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
            self.train = CFGDataset(self.train, p_uncond, self.K)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_imagenet256_guided_diffusion.npz'

    def sample_label(self, n_samples, device):
        return torch.randint(0, 1000, (n_samples,), device=device)


class ImageNet512Features(DatasetFactory):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print('Prepare dataset...')
        self.train = FeatureDataset(path)
        print('Prepare dataset ok')
        self.K = 1000

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
            self.train = CFGDataset(self.train, p_uncond, self.K)

    @property
    def data_shape(self):
        return 4, 64, 64

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_imagenet512_guided_diffusion.npz'

    def sample_label(self, n_samples, device):
        return torch.randint(0, 1000, (n_samples,), device=device)


class ImageNet(DatasetFactory):
    def __init__(self, path, resolution, random_crop=False, random_flip=True):
        super().__init__()

        print(f'Counting ImageNet files from {path}')
        train_files = _list_image_files_recursively(os.path.join(path, 'train'))
        class_names = [os.path.basename(path).split("_")[0] for path in train_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        train_labels = [sorted_classes[x] for x in class_names]
        print('Finish counting ImageNet files')

        self.train = ImageDataset(resolution, train_files, labels=train_labels, random_crop=random_crop, random_flip=random_flip)
        self.resolution = resolution
        if len(self.train) != 1_281_167:
            print(f'Missing train samples: {len(self.train)} < 1281167')

        self.K = max(self.train.labels) + 1
        cnt = dict(zip(*np.unique(self.train.labels, return_counts=True)))
        self.cnt = torch.tensor([cnt[k] for k in range(self.K)]).float()
        self.frac = [self.cnt[k] / len(self.train.labels) for k in range(self.K)]
        print(f'{self.K} classes')
        print(f'cnt[:10]: {self.cnt[:10]}')
        print(f'frac[:10]: {self.frac[:10]}')

    @property
    def data_shape(self):
        return 3, self.resolution, self.resolution

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_imagenet{self.resolution}_guided_diffusion.npz'

    def sample_label(self, n_samples, device):
        return torch.multinomial(self.cnt, n_samples, replacement=True).to(device)

    def label_prob(self, k):
        return self.frac[k]

class MajorTOMThumbnail(DatasetFactory):
    def __init__(self, path, resolution):
        super().__init__()

        print(f'Counting MajorTOM thumbnail files from {path}')
        files_list = _list_image_files_recursively(path)
        print('Finish counting MajorTOM thumbnail files')

        self.dataset = MajorTOMThumbnailDataset(resolution, files_list)
        self.resolution = resolution
        if len(self.dataset) != 1_281_167:
            print(f'Missing train samples: {len(self.dataset)} < 1281167')

    @property
    def data_shape(self):
        return 3, self.resolution, self.resolution
    
    @property
    def has_label(self):
        return False

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_imagenet{self.resolution}_guided_diffusion.npz'


class MajorTOMThumbnailDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
    ):
        super().__init__()
        self.resolution = resolution
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        filename = os.path.basename(path).split('.')[0]
        pil_image = Image.open(path)
        pil_image.load()
        pil_image = pil_image.convert("RGB")

        # check that the image has the correct resolution
        if pil_image.size != (self.resolution, self.resolution):
            raise ValueError(f"Image at {path} has size {pil_image.size}, expected {self.resolution}x{self.resolution}")

        arr = np.array(pil_image).astype(np.float32) / 127.5 - 1

        return np.transpose(arr, [2, 0, 1]), filename

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif os.listdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        labels,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.image_paths = image_paths
        self.labels = labels
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        pil_image = Image.open(path)
        pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        label = np.array(self.labels[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), label


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


# CelebA


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )


class CelebA(DatasetFactory):
    r""" train: 162,770
         val:   19,867
         test:  19,962
         shape: 3 * width * width
    """

    def __init__(self, path, resolution=64):
        super().__init__()

        self.resolution = resolution

        cx = 89
        cy = 121
        x1 = cy - 64
        x2 = cy + 64
        y1 = cx - 64
        y2 = cx + 64

        transform = transforms.Compose([Crop(x1, x2, y1, y2), transforms.Resize(self.resolution),
                                        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                        transforms.Normalize(0.5, 0.5)])
        self.train = datasets.CelebA(root=path, split="train", target_type=[], transform=transform, download=True)
        self.train = UnlabeledDataset(self.train)

    @property
    def data_shape(self):
        return 3, self.resolution, self.resolution

    @property
    def fid_stat(self):
        return 'assets/fid_stats/fid_stats_celeba64_train_50000_ddim.npz'

    @property
    def has_label(self):
        return False


# MS COCO


def center_crop(width, height, img):
    resample = {'box': Image.BOX, 'lanczos': Image.LANCZOS}['lanczos']
    crop = np.min(img.shape[:2])
    img = img[(img.shape[0] - crop) // 2: (img.shape[0] + crop) // 2,
          (img.shape[1] - crop) // 2: (img.shape[1] + crop) // 2]
    try:
        img = Image.fromarray(img, 'RGB')
    except:
        img = Image.fromarray(img)
    img = img.resize((width, height), resample)

    return np.array(img).astype(np.uint8)


class MSCOCODatabase(Dataset):
    def __init__(self, root, annFile, size=None):
        from pycocotools.coco import COCO
        self.root = root
        self.height = self.width = size

        self.coco = COCO(annFile)
        self.keys = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, key: int):
        path = self.coco.loadImgs(key)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, key: int):
        return self.coco.loadAnns(self.coco.getAnnIds(key))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        image = self._load_image(key)
        image = np.array(image).astype(np.uint8)
        image = center_crop(self.width, self.height, image).astype(np.float32)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = einops.rearrange(image, 'h w c -> c h w')

        anns = self._load_target(key)
        target = []
        for ann in anns:
            target.append(ann['caption'])

        return image, target


def get_feature_dir_info(root):
    files = glob.glob(os.path.join(root, '*.npy'))
    files_caption = glob.glob(os.path.join(root, '*_*.npy'))
    num_data = len(files) - len(files_caption)
    n_captions = {k: 0 for k in range(num_data)}
    for f in files_caption:
        name = os.path.split(f)[-1]
        k1, k2 = os.path.splitext(name)[0].split('_')
        n_captions[int(k1)] += 1
    return num_data, n_captions


class MSCOCOFeatureDataset(Dataset):
    # the image features are got through sample
    def __init__(self, root):
        self.root = root
        self.num_data, self.n_captions = get_feature_dir_info(root)

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        z = np.load(os.path.join(self.root, f'{index}.npy'))
        k = random.randint(0, self.n_captions[index] - 1)
        c = np.load(os.path.join(self.root, f'{index}_{k}.npy'))
        return z, c


class MSCOCO256Features(DatasetFactory):  # the moments calculated by Stable Diffusion image encoder & the contexts calculated by clip
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print('Prepare dataset...')
        self.train = MSCOCOFeatureDataset(os.path.join(path, 'train'))
        self.test = MSCOCOFeatureDataset(os.path.join(path, 'val'))
        assert len(self.train) == 82783
        assert len(self.test) == 40504
        print('Prepare dataset ok')

        self.empty_context = np.load(os.path.join(path, 'empty_context.npy'))

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
            self.train = CFGDataset(self.train, p_uncond, self.empty_context)

        # text embedding extracted by clip
        # for visulization in t2i
        self.prompts, self.contexts = [], []
        for f in sorted(os.listdir(os.path.join(path, 'run_vis')), key=lambda x: int(x.split('.')[0])):
            prompt, context = np.load(os.path.join(path, 'run_vis', f), allow_pickle=True)
            self.prompts.append(prompt)
            self.contexts.append(context)
        self.contexts = np.array(self.contexts)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_mscoco256_val.npz'


def get_dataset(name, **kwargs):
    if name == 'cifar10':
        return CIFAR10(**kwargs)
    elif name == 'imagenet':
        return ImageNet(**kwargs)
    elif name == 'imagenet256_features':
        return ImageNet256Features(**kwargs)
    elif name == 'imagenet512_features':
        return ImageNet512Features(**kwargs)
    elif name == "majorTOM_S2_256_features":
        return MajorTOM_S2_Features(**kwargs)
    elif name == "majorTOM_tuples_256_features":
        return MajorTOM_Tuples_Features(**kwargs)
    elif name == "majorTOM_lmdb_256_features":
        return MajorTOM_Lmdb_Features(**kwargs)
    elif name == 'celeba':
        return CelebA(**kwargs)
    elif name == 'mscoco256_features':
        return MSCOCO256Features(**kwargs)
    elif name == "copgen_lmdb_features":
        return COPGEN_Features(**kwargs)
    else:
        raise NotImplementedError(name)
