import ml_collections
def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)

def modality_configs():
    """
        Edit this function to add new modalities.
        NOTE: The order of the modalities matters.
        It is the order that the network will use.
    """
    all_modality_configs = {
        # ====== SCALAR: lat_lon and timestamps ======
        "3d_cartesian_lat_lon": d(
            modality_type="scalar",
            latent_channels=1,
            latent_input_resolution=(1, 3),
            img_input_resolution=(1, 3),
            img_bands=1,
        ),
        "mean_timestamps": d(
            modality_type="scalar",
            latent_channels=1,
            latent_input_resolution=(1, 3),
            img_input_resolution=(1, 3),
            img_bands=1,
        ),
        # ==================== DEM ====================
        "DEM_DEM": d(
            modality_type="image",
            bands=['DEM'],
            autoencoder_path="models/vae/DEM_64x64_DEM_latent_8/model-50-ema.pth",
            latent_channels=8,                # latent z_channels (8, *, *)
            latent_input_resolution=(8, 8),   # latent input size (8, 8, 8)
            img_bands=1,                      # number of bands   (1)
            img_input_resolution=(64, 64),    # input pixel space (1, 64, 64)
            scale_factor=0.404247, # Computed with 500 samples
            min_db=0.000006,
            max_db=9.088,
            min_positive=5.899129519093549e-06,
        ),
        # ==================== S1RTC ====================
        "S1RTC_vh_vv": d(
            modality_type="image",
            bands=['vv', 'vh'],
            autoencoder_path="models/vae/S1RTC_192x192_VV_VH_latent_8/model-50-ema.pth",
            latent_channels=8,                  # latent z_channels (8, *, *)
            latent_input_resolution=(24, 24),   # latent input size (8, 24, 24)
            img_bands=2,                        # number of bands   (2)
            img_input_resolution=(192, 192),    # input pixel space (2, 192, 192)
            scale_factor=0.633804, # Computed with 500 samples
            min_db=-55.787186,
            max_db=40.225800,
            min_positive=2.6380407689430285e-06,
        ),
        # ==================== S2L1C ====================
        "S2L1C_B02_B03_B04_B08": d(
            modality_type="image",
            bands=['B04', 'B03', 'B02', 'B08'],
            autoencoder_path="models/vae/S2L1C_192x192_B4_3_2_8_latent_8/model-50-ema.pth",
            latent_channels=8,                # latent z_channels (8, *, *)
            latent_input_resolution=(24, 24), # latent input size (8, 24, 24)
            img_bands=4,                      # number of bands   (4)
            img_input_resolution=(192, 192),  # input pixel space (4, 192, 192)
            scale_factor=0.772514, # Computed with 500 samples
        ),
        "S2L1C_B05_B06_B07_B11_B12_B8A": d(
            modality_type="image",
            bands=['B05', 'B06', 'B07', 'B8A', 'B11', 'B12'],
            autoencoder_path="models/vae/S2L1C_96x96_B5_6_7_8A_11_12_latent_8/model-50-ema.pth",
            latent_channels=8,                # latent z_channels (8, *, *)
            latent_input_resolution=(12, 12), # latent input size (8, 12, 12)
            img_bands=6,                      # number of bands   (6)
            img_input_resolution=(96, 96),    # input pixel space (6, 96, 96)
            scale_factor=0.765817, # Computed with 500 samples
        ),
        "S2L1C_B01_B09_B10": d(
            modality_type="image",
            bands=['B01', 'B09', 'B10'],
            autoencoder_path="models/vae/S2L1C_32x32_B1_9_10_latent_8/model-50-ema.pth",
            latent_channels=8,                # latent z_channels (8, *, *)
            latent_input_resolution=(4, 4),   # latent input size (8, 4, 4)
            img_bands=3,                      # number of bands   (3)
            img_input_resolution=(32, 32),    # input pixel space (3, 32, 32)
            scale_factor=0.565812, # Computed with 500 samples
        ),
        # ==================== S2L2A ====================
        "S2L2A_B02_B03_B04_B08": d(
            modality_type="image",
            bands=['B04', 'B03', 'B02', 'B08'],
            autoencoder_path="models/vae/S2L2A_192x192_B4_3_2_8_latent_8/model-50-ema.pth",
            latent_channels=8,                # latent z_channels (8, *, *)
            latent_input_resolution=(24, 24), # latent input size (8, 24, 24)
            img_bands=4,                      # number of bands   (4)
            img_input_resolution=(192, 192),  # input pixel space (4, 192, 192)
            scale_factor=0.853416, # Computed with 500 samples
        ),
        "S2L2A_B05_B06_B07_B11_B12_B8A": d(
            modality_type="image",
            bands=['B05', 'B06', 'B07', 'B8A', 'B11', 'B12'],
            autoencoder_path="models/vae/S2L2A_96x96_B5_6_7_8A_11_12_latent_8/model-50-ema.pth",
            latent_channels=8,                # latent z_channels (8, *, *)
            latent_input_resolution=(12, 12), # latent input size (8, 12, 12)
            img_bands=6,                      # number of bands   (6)
            img_input_resolution=(96, 96),    # input pixel space (6, 96, 96)
            scale_factor=0.567267, # Computed with 500 samples
        ),
        "S2L2A_B01_B09": d(
            modality_type="image",
            bands=['B01', 'B09'],
            autoencoder_path="models/vae/S2L2A_32x32_B1_9_latent_8/model-50-ema.pth",
            latent_channels=8,                # latent z_channels (8, *, *)
            latent_input_resolution=(4, 4),   # latent input size (8, 4, 4)
            img_bands=2,                      # number of bands   (2)
            img_input_resolution=(32, 32),    # input pixel space (2, 32, 32)
            scale_factor=0.571367, # Computed with 500 samples
        ),
        # ==================== LULC ====================
        "LULC_LULC": d(
            modality_type="image",
            bands=['LULC'],
            autoencoder_path="models/vae/LULC_192x192_LULC_latent_8/model-50-ema.pth",
            latent_channels=8,                # latent z_channels (8, *, *)
            latent_input_resolution=(24, 24), # latent input size (8, 24, 24)
            img_bands=10,                     # number of bands   (10 – one-hot classes)
            img_input_resolution=(192, 192),  # input pixel space (1, 192, 192)
            scale_factor=0.464794, # Computed with 500 samples
        ),
        # ==================== CLOUD_MASK ====================
        "S2L1C_cloud_mask": d(
            modality_type="image",
            bands=['cloud_mask'],
            autoencoder_path="models/vae/S2L1C_192x192_cloud_mask_latent_8/model-50-ema.pth",
            latent_channels=8,                # latent z_channels (8, *, *)
            latent_input_resolution=(24, 24), # latent input size (8, 24, 24)
            img_bands=4,                      # number of bands   (4 – one-hot classes)
            img_input_resolution=(192, 192),  # input pixel space (1, 192, 192)
            scale_factor=0.557045, # Computed with 500 samples
        ),
    }
    return all_modality_configs

def get_config():
    config = ml_collections.ConfigDict()
    config.all_modality_configs = modality_configs()
    z_channels = {modality: mod_config.latent_channels for modality, mod_config in config.all_modality_configs.items()}
    config.seed = 1234
    config.pred = "noise_pred"
    config.z_shapes = {modality: (mod_config.latent_channels,           # E.g. 8
                                  mod_config.latent_input_resolution[0],    # E.g. 4
                                  mod_config.latent_input_resolution[1])    # E.g. 4
                       for modality, mod_config in config.all_modality_configs.items()}
    config.input_resolutions = {modality: (mod_config.img_bands, mod_config.img_input_resolution[0], mod_config.img_input_resolution[1])
                     for modality, mod_config in config.all_modality_configs.items()}
    # AUTOENCODERS CONFIG (auto-fill)
    config.autoencoders = {}
    for modality, mod_config in config.all_modality_configs.items():
        if 'image' in mod_config.modality_type:
            config.autoencoders[modality] = d(
                pretrained_path=mod_config.autoencoder_path,
                ddconfig=dict(
                    double_z=True,
                    z_channels=mod_config.latent_channels,          # E.g. 8
                    resolution=mod_config.img_input_resolution[0],  # E.g. 32
                    in_channels=mod_config.img_bands,               # E.g. 2
                    out_ch=mod_config.img_bands,                    # E.g. 2
                    ch=128,
                    ch_mult=[1, 2, 4, 4],
                    num_res_blocks=2,
                    attn_resolutions=[],
                    dropout=0.0
                ),
                scale_factor=mod_config.scale_factor,
            )
    
    config.train = d(
        n_steps=500000, # Increase to 500000
        batch_size=16,  # Increase for larger datasets
        gradient_accumulation_steps=2, # Accumulate gradients over 2 batches
        num_workers=8,
        mode="uncond",
        log_interval=200,
        eval_interval=1000,
        save_interval=1000,
        multi_modal=True,
    )
    config.optimizer = d(
        name="adamw",
        lr=0.0002,
        weight_decay=0.03,
        betas=(0.99, 0.99),
    )
    config.lr_scheduler = d(name="customized", warmup_steps=5000)
    config.nnet = d(
        name="copgen_multi_post_ln",
        # Input size for the nnet => Latent crop sizes for each modality. E.g. img_sizes = {"S2L2A_B01_B09": 4...)
        img_sizes={modality: mod_config.latent_input_resolution[0]
                   for modality, mod_config in config.all_modality_configs.items()},
        in_chans=z_channels, # All modalities have the same latent_channels
        patch_size={modality: (1 if 'scalar' in mod_config.modality_type else 2) for modality, mod_config in config.all_modality_configs.items()},
        embed_dim=1024, # Increase to 1152
        depth=20, # Increase to 28
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=False,
        pos_drop_rate=0.,
        drop_rate=0.,
        attn_drop_rate=0.,
        mlp_time_embed=False,
        modalities=config.all_modality_configs,
        use_checkpoint=True,
    )
    config.dataset = d(
        name="copgen_lmdb_features",
        path="./data/majorTOM/edinburgh/latents_merged/train",
        cfg=False,
        p_uncond=0.1, # 0.15
        patches_per_side=6,
        patches_size={modality: (mod_config.img_input_resolution[0],
                                mod_config.img_input_resolution[1])
                     for modality, mod_config in config.all_modality_configs.items() if 'image' in mod_config.modality_type},
        lat_lon_encode='3d_cartesian_lat_lon',
        precomputed_lat_lon_path="./data/majorTOM/edinburgh/3d_cartesian_lat_lon_cache",
        time_encode='mean_timestamps',
        precomputed_mean_timestamps_path="./data/majorTOM/edinburgh/mean_timestamps_cache.pkl",
        random_flip=False,
        normalize_to_neg_one_to_one=True,
        # Full size of the encoded latents. E.g. (2*8, 22, 22)
        z_input_shapes={modality: (2*mod_config.latent_channels if 'image' in mod_config.modality_type else mod_config.latent_channels,
                             mod_config.latent_input_resolution[0],
                             mod_config.latent_input_resolution[1])
                        for modality, mod_config in config.all_modality_configs.items()},
        crop_shapes=None, # No cropping on the latent space
        min_db={modality: mod_config.get('min_db', None) for modality, mod_config in config.all_modality_configs.items()},
        max_db={modality: mod_config.get('max_db', None) for modality, mod_config in config.all_modality_configs.items()},
        min_positive={modality: mod_config.get('min_positive', None) for modality, mod_config in config.all_modality_configs.items()},
    )
    config.sample = d(
        sample_steps=50,
        n_samples=50000,
        n_images_to_log=5,
        mini_batch_size=50,  # the decoder is large
        algorithm="dpm_solver",
        # cfg=True,
        cfg=False,
        scale=0.4,
        path="",
    )
    return config
