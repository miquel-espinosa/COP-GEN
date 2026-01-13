# COPGEN Clean Inference Interface

A simplified, user-friendly interface for the COPGEN multimodal generative model that hides implementation complexity while maintaining full flexibility.

## Quick Start

```python
from copgen_inference import CopgenModel

# Load model
model = CopgenModel("path/to/model.pt", "path/to/config.yaml")

# Generate samples unconditionally
dem = model.generate("DEM_DEM")

# Generate with conditions
conditions = {"S2L1C_cloud_mask": cloud_mask_tensor}
s2_imagery = model.generate("S2L2A_B02_B03_B04_B08", conditions=conditions)

# Generate multiple modalities
samples = model.generate(["DEM_DEM", "S1RTC_vh_vv"], n_samples=5)
```

## Features

### Simple API
- **One-line model loading**: No need to manage devices, autoencoders, or noise schedules
- **Intuitive generation**: Clear separation between what to generate and what to condition on
- **Automatic batching**: Handles batch processing transparently
- **Clean return values**: Single tensor for one modality, dictionary for multiple

### Flexible Architecture
- **Multiple modalities**: Generate any combination of supported modalities
- **Conditional generation**: Condition on any subset of modalities
- **Batch processing**: Efficient processing of multiple inputs
- **Latent space access**: Optional access to latent representations
- **Device management**: Automatic GPU/CPU selection

### Separation of Concerns
- **Model inference only**: No data loading, preprocessing, or visualisation
- **Clean inputs/outputs**: Works with PyTorch tensors
- **No hidden preprocessing**: User controls data preparation
- **Modular design**: Easy to integrate into existing pipelines

## API Reference

### CopgenModel

The main class for COPGEN inference.

#### Constructor
```python
model = CopgenModel(
    model_path="path/to/model.pt",      # Path to trained model
    config_path="path/to/config.yaml",   # Path to configuration
    device="cuda",                       # Optional: device selection
    seed=42                             # Optional: random seed
)
```

#### Properties
```python
# Get list of available modalities
modalities = model.available_modalities
```

#### Methods

##### generate()
Main generation method supporting both unconditional and conditional generation.

```python
samples = model.generate(
    modalities=["DEM_DEM", "S2L2A_B02_B03_B04_B08"],  # What to generate
    conditions={"S2L1C_cloud_mask": mask_tensor},       # Optional conditions
    n_samples=5,                                        # Samples per condition
    batch_size=2,                                       # Batch size
    sample_steps=50,                                    # Diffusion steps
    return_latents=False                                # Return latents too?
)
```

##### encode()
Encode data into latent space.

```python
data = {
    "DEM_DEM": dem_tensor,
    "S2L1C_cloud_mask": mask_tensor
}
latents = model.encode(data)
```

##### decode()
Decode latents back to data space.

```python
decoded = model.decode(latents)
```

### Convenience Functions

```python
# Load model
model = load_copgen_model(model_path, config_path, device=None)

# Unconditional generation
samples = generate_unconditional(model, modalities, n_samples=1)

# Conditional generation
samples = generate_conditional(model, generate_modalities, condition_data, n_samples=1)
```

## Usage Examples

### 1. Simple Generation

```python
# Generate a single DEM
dem = model.generate("DEM_DEM")

# Generate multiple DEMs
dems = model.generate("DEM_DEM", n_samples=10)

# Generate multiple modalities
samples = model.generate(["DEM_DEM", "S2L2A_B02_B03_B04_B08"])
dem = samples["DEM_DEM"]
s2 = samples["S2L2A_B02_B03_B04_B08"]
```

### 2. Conditional Generation

```python
# Condition on cloud mask to generate clear imagery
conditions = {"S2L1C_cloud_mask": cloud_mask}
clear_s2 = model.generate("S2L2A_B02_B03_B04_B08", conditions=conditions)

# Multiple conditions
conditions = {
    "S2L1C_cloud_mask": cloud_mask,
    "DEM_DEM": elevation_data
}
samples = model.generate(
    ["S2L2A_B02_B03_B04_B08", "S1RTC_vh_vv"],
    conditions=conditions
)
```

### 3. Batch Processing

```python
# Process multiple locations at once
batch_size = 8
conditions = {
    "S2L1C_cloud_mask": torch.stack([mask1, mask2, ..., mask8])
}
batch_results = model.generate(
    ["S2L2A_B02_B03_B04_B08", "DEM_DEM"],
    conditions=conditions,
    batch_size=batch_size
)
```

### 4. Working with Latents

```python
# Get both decoded and latent representations
decoded, latents = model.generate("DEM_DEM", return_latents=True)

# Manipulate in latent space
data = {"DEM_DEM": dem_tensor}
latent = model.encode(data)
# ... modify latent ...
modified_dem = model.decode({"DEM_DEM": latent})
```

### 5. Production Pipeline Example

```python
def process_region(model, region_data):
    """Process a region with COPGEN"""
    
    # Prepare conditions from available data
    conditions = {}
    if "cloud_mask" in region_data:
        conditions["S2L1C_cloud_mask"] = region_data["cloud_mask"]
    if "elevation" in region_data:
        conditions["DEM_DEM"] = region_data["elevation"]
    
    # Generate missing modalities
    to_generate = ["S2L2A_B02_B03_B04_B08", "S1RTC_vh_vv"]
    
    # Generate multiple samples for uncertainty
    samples = model.generate(
        to_generate,
        conditions=conditions,
        n_samples=3,
        sample_steps=100  # Higher quality
    )
    
    # Post-process results (user-defined)
    results = {}
    for modality, tensor in samples.items():
        # Apply domain-specific post-processing
        processed = postprocess_modality(tensor, modality)
        results[modality] = processed
    
    return results
```

## Supported Modalities

The available modalities depend on the trained model. Common modalities include:

- `DEM_DEM`: Digital Elevation Model
- `S2L2A_B02_B03_B04_B08`: Sentinel-2 Level 2A RGB+NIR bands  
- `S2L1C_cloud_mask`: Sentinel-2 Level 1C cloud mask
- `S1RTC_vh_vv`: Sentinel-1 RTC VH/VV polarizations
- `LULC_LULC`: Land Use Land Cover classification
- And more...

Use `model.available_modalities` to see what's available for your model.

## Best Practices

1. **Load model once**: Model loading is expensive; reuse the model instance
2. **Batch when possible**: Process multiple inputs together for efficiency
3. **Handle preprocessing separately**: Apply normalization/transformations before passing to model
4. **Check modality names**: Use `available_modalities` to verify correct names
5. **Set seed for reproducibility**: Use the `seed` parameter when consistent results are needed

## Migration from Original Interface

If you're migrating from the original COPGEN interface:

**Before:**
```python
ctx = CopgenSamplingContext(config)
z_cond_map = {}
for m in ctx.modality_names:
    if m in ctx.condition_set:
        z_lat = ctx.encode_for_condition(moments[m], m)
        z_cond_map[m] = z_lat
_zs = ctx.sample_batch(batch_size, z_cond_map)
samples = [ctx.decode(_z, m) for m, _z in zip(ctx.modality_names, _zs)]
```

**After:**
```python
model = CopgenModel(model_path, config_path)
samples = model.generate(
    modalities_to_generate,
    conditions=condition_data
)
```

## Technical Notes

- The interface handles all device management internally
- Autoencoders are loaded lazily and cached
- The model runs in evaluation mode by default
- Mixed precision (AMP) is used automatically where beneficial
- The underlying diffusion process uses DPM-Solver++ for efficiency

## Requirements

- PyTorch
- NumPy
- ml_collections
- einops
- Custom modules: `libs.autoencoder`, `utils`, `dpm_solver_pp`

## License

See main project license.
