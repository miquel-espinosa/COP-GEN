from pathlib import Path
from libs.copgen import CopgenModel

model_path = Path('models/copgen/world_12_modalities_large/300000.ckpt/nnet_ema.pth')
config_path = Path('configs/copgen/discrete/cop_gen_large.py')

model = CopgenModel(model_path, config_path)

# Generate 5 samples for DEM and S2L2A_B02_B03_B04_B08
batch_size = 1
n_samples = 5
samples = model.generate(["DEM_DEM", "S2L2A_B02_B03_B04_B08"], n_samples=n_samples, batch_size=batch_size)
for modality, sample in samples.items():
    print(modality, sample.shape, sample.min(), sample.max())

# Generate 5 samples for 3d_cartesian_lat_lon and mean_timestamps
samples = model.generate(["3d_cartesian_lat_lon", "mean_timestamps"], n_samples=n_samples, batch_size=batch_size)
for modality, sample in samples.items():
    print(modality, sample.shape, sample.min(), sample.max())