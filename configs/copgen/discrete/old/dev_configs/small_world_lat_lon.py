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
        "3d_cartesian_lat_lon": d(
            modality_type="scalar",
            latent_channels=1,
            latent_input_resolution=(1, 3),
            img_input_resolution=(1, 3),
            img_bands=1,
        ),
        "S2L2A_B01_B09": d(
            modality_type="image",
            autoencoder_path="models/vae/S2L2A_32x32_B1_9_latent_8/model-50-ema.pt",
            latent_channels=8,          # latent z_channels (8, *, *)
            latent_input_resolution=(4, 4),  # latent input size (8, 4, 4)
            img_bands=2,                # number of bands   (2)
            img_input_resolution=(32, 32),    # input pixel space (2, 32, 32)
            scale_factor=0.407005,
        ),
        "S2L2A_B02_B03_B04_B08": d(
            modality_type="image",
            autoencoder_path="models/vae/S2L2A_192x192_B4_3_2_8_latent_8/model-50-ema.pt",
            latent_channels=8,           # latent z_channels (8, *, *)
            latent_input_resolution=(24, 24),  # latent input size (8, 24, 24)
            img_bands=4,                 # number of bands   (4)
            img_input_resolution=(192, 192),    # input pixel space (4, 192, 192)
            scale_factor=0.331742,
        ),
        "S2L2A_B05_B06_B07_B11_B12_B8A": d(
            modality_type="image",
            autoencoder_path="models/vae/S2L2A_96x96_B5_6_7_8A_11_12_latent_8/model-50-ema.pt",
            latent_channels=8,          # latent z_channels (8, *, *)
            latent_input_resolution=(12, 12), # latent input size (8, 12, 12)
            img_bands=6,                # number of bands   (6)
            img_input_resolution=(96, 96),    # input pixel space (6, 96, 96)
            scale_factor=0.364526,
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
        batch_size=16,  # Change to 512
        num_workers=16,
        mode="uncond",
        log_interval=2, # Change to 100
        eval_interval=2, # Change to 5000
        save_interval=20, # Change to 5000
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
        path="data/majorTOM/small_world/latents_S2L2A_lmdb/train",
        cfg=False,
        p_uncond=0.1, # 0.15
        patches_per_side=6,
        patches_size={modality: (mod_config.img_input_resolution[0],
                                mod_config.img_input_resolution[1])
                     for modality, mod_config in config.all_modality_configs.items() if 'image' in mod_config.modality_type},
        # lat_lon_encode='spherical_harmonics',
        # precomputed_lat_lon_path="assets/s2_test_spherical_harmonics_cache.pkl",
        # lat_lon_encode='lat_lon',
        # precomputed_lat_lon_path="assets/s2_test_lat_lon_cache.pkl",
        lat_lon_encode='3d_cartesian_lat_lon',
        precomputed_lat_lon_path="assets/s2_test_3d_cartesian_lat_lon_cache.pkl",
        random_flip=False,
        # Full size of the encoded latents. E.g. (2*8, 22, 22)
        z_input_shapes={modality: (2*mod_config.latent_channels if 'image' in mod_config.modality_type else mod_config.latent_channels,
                             mod_config.latent_input_resolution[0],
                             mod_config.latent_input_resolution[1])
                        for modality, mod_config in config.all_modality_configs.items()},
        crop_shapes=None, # No cropping on the latent space
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
