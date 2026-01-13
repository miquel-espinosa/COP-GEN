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
        "DEM_thumbnail": d(
            autoencoder_path="assets/stable-diffusion/autoencoder_kl_ema.pth",
            latent_channels=4,           # latent z_channels (4, *, *)
            latent_input_resolution=32,  # latent input size (4, 32, 32)
            latent_crop_resolution=32,   # latent crop  size (4, 32, 32)
            img_bands=3,                 # number of bands   (3)
            img_input_resolution=256,    # input pixel space (3, 256, 256)
            scale_factor='sd',
        ),
        "S1RTC_thumbnail": d(
            autoencoder_path="assets/stable-diffusion/autoencoder_kl_ema.pth",
            latent_channels=4,           # latent z_channels (4, *, *)
            latent_input_resolution=32,  # latent input size (4, 32, 32)
            latent_crop_resolution=32,   # latent crop  size (4, 32, 32)
            img_bands=3,                 # number of bands   (3)
            img_input_resolution=256,    # input pixel space (3, 256, 256)
            scale_factor='sd',
        ),
        "S2L1C_thumbnail": d(
            autoencoder_path="assets/stable-diffusion/autoencoder_kl_ema.pth",
            latent_channels=4,           # latent z_channels (4, *, *)
            latent_input_resolution=32,  # latent input size (4, 32, 32)
            latent_crop_resolution=32,   # latent crop  size (4, 32, 32)
            img_bands=3,                 # number of bands   (3)
            img_input_resolution=256,    # input pixel space (3, 256, 256)
            scale_factor='sd',
        ),
        "S2L2A_B02_B03_B04": d(
            autoencoder_path="assets/stable-diffusion/autoencoder_kl_ema.pth",
            latent_channels=4,           # latent z_channels (8, *, *)
            latent_input_resolution=32,  # latent input size (8, 24, 24)
            latent_crop_resolution=32,   # latent crop  size (8, 24, 24)
            img_bands=3,                 # number of bands   (4)
            img_input_resolution=256,    # input pixel space (4, 192, 192)
            scale_factor='sd',
        ),
        "S2L2A_B08_B08_B08": d(
            autoencoder_path="assets/stable-diffusion/autoencoder_kl_ema.pth",
            latent_channels=4,           # latent z_channels (8, *, *)
            latent_input_resolution=32,  # latent input size (8, 24, 24)
            latent_crop_resolution=32,   # latent crop  size (8, 24, 24)
            img_bands=3,                 # number of bands   (4)
            img_input_resolution=256,    # input pixel space (4, 192, 192)
            scale_factor='sd',
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
        batch_size=128,  # Change to 512
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
        path="data/majorTOM/small_world/latents_copgenbeta_hybrid_ultimate_test_lmdb/train",
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
