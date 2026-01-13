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
        "S2L2A_B01_B09": d(
            autoencoder_path="models/vae/S2L2A_40x40_B1_9_latent_8/model-35-ema.pth",
            latent_channels=8,          # latent z_channels (8, *, *)
            latent_input_resolution=4,  # latent input size (8, 4, 4)
            latent_crop_resolution=4,   # latent crop  size (8, 4, 4)
            img_bands=2,                # number of bands   (2)
            img_input_resolution=32,    # input pixel space (2, 32, 32)
            scale_factor=0.407005,
        ),
        "S2L2A_B02_B03_B04_B08": d(
            autoencoder_path="models/vae/S2L2A_240x240_B4_3_2_8_latent_8/model-22-ema.pth",
            latent_channels=8,           # latent z_channels (8, *, *)
            latent_input_resolution=24,  # latent input size (8, 24, 24)
            latent_crop_resolution=24,   # latent crop  size (8, 24, 24)
            img_bands=4,                 # number of bands   (4)
            img_input_resolution=192,    # input pixel space (4, 192, 192)
            scale_factor=0.331742,
        ),
        "S2L2A_B05_B06_B07_B11_B12_B8A": d(
            autoencoder_path="models/vae/S2L2A_120x120_B5_6_7_8A_11_12_latent_8/model-20-ema.pth",
            latent_channels=8,          # latent z_channels (8, *, *)
            latent_input_resolution=12, # latent input size (8, 12, 12)
            latent_crop_resolution=12,  # latent crop  size (8, 12, 12)
            img_bands=6,                # number of bands   (6)
            img_input_resolution=96,    # input pixel space (6, 96, 96)
            scale_factor=0.364526,
        ),
    }
    return all_modality_configs

def get_config():
    all_modality_configs = modality_configs()
    z_channels = {modality: mod_config.latent_channels for modality, mod_config in all_modality_configs.items()}
    config = ml_collections.ConfigDict()
    config.seed = 1234
    config.pred = "noise_pred"
    config.z_shapes = {modality: (mod_config.latent_channels,           # E.g. 8
                                  mod_config.latent_crop_resolution,    # E.g. 4
                                  mod_config.latent_crop_resolution)    # E.g. 4
                       for modality, mod_config in all_modality_configs.items()}
    # AUTOENCODERS CONFIG (auto-fill)
    config.autoencoders = {}
    for modality, mod_config in all_modality_configs.items():
        config.autoencoders[modality] = d(
            pretrained_path=mod_config.autoencoder_path,
            ddconfig=dict(
                double_z=True,
                z_channels=mod_config.latent_channels,       # E.g. 8
                resolution=mod_config.img_input_resolution,  # E.g. 32
                in_channels=mod_config.img_bands,            # E.g. 2
                out_ch=mod_config.img_bands,                 # E.g. 2
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
        batch_size=256,  # Change to 512
        mode="uncond",
        log_interval=100, # Change to 100
        eval_interval=5000, # Change to 1500
        save_interval=5000,
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
        img_sizes={modality: mod_config.latent_crop_resolution
                   for modality, mod_config in all_modality_configs.items()},
        in_chans=z_channels, # All modalities have the same latent_channels
        patch_size=2,
        embed_dim=1024, # Increase to 1152
        depth=20, # Increase to 28
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=False,
        pos_drop_rate=0.,
        drop_rate=0.,
        attn_drop_rate=0.,
        mlp_time_embed=False,
        modalities=tuple(all_modality_configs.keys()),
        use_checkpoint=True,
    )
    config.dataset = d(
        name="copgen_lmdb_features",
        path="data/majorTOM/small_world/latents_S2L2A_lmdb/train",
        cfg=False,
        p_uncond=0.1, # 0.15
        random_flip=True,
        # Full size of the encoded latents. E.g. (2*8, 22, 22)
        z_input_shapes={modality: (2*mod_config.latent_channels,
                             mod_config.latent_input_resolution,
                             mod_config.latent_input_resolution)
                        for modality, mod_config in all_modality_configs.items()},
        # Crop size of the encoded latents. E.g. (2*8, 4, 4)
        crop_shapes={modality: (2*mod_config.latent_channels,
                                mod_config.latent_crop_resolution,
                                mod_config.latent_crop_resolution)
                     for modality, mod_config in all_modality_configs.items()},
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
