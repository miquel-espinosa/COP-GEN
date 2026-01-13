#!/bin/bash

python3 scripts/extract_ema_convert_model.py models/vae/DEM_64x64_DEM_latent_8/model-50.pt
python3 scripts/extract_ema_convert_model.py models/vae/LULC_192x192_LULC_latent_8/model-50.pt
python3 scripts/extract_ema_convert_model.py models/vae/S1RTC_192x192_VV_VH_latent_8/model-50.pt
python3 scripts/extract_ema_convert_model.py models/vae/S2L1C_192x192_B4_3_2_8_latent_8/model-50.pt
python3 scripts/extract_ema_convert_model.py models/vae/S2L1C_192x192_cloud_mask_latent_8/model-50.pt
python3 scripts/extract_ema_convert_model.py models/vae/S2L1C_32x32_B1_9_10_latent_8/model-50.pt
python3 scripts/extract_ema_convert_model.py models/vae/S2L1C_96x96_B5_6_7_8A_11_12_latent_8/model-50.pt
python3 scripts/extract_ema_convert_model.py models/vae/S2L2A_192x192_B4_3_2_8_latent_8/model-50.pt
python3 scripts/extract_ema_convert_model.py models/vae/S2L2A_32x32_B1_9_latent_8/model-50.pt
python3 scripts/extract_ema_convert_model.py models/vae/S2L2A_96x96_B5_6_7_8A_11_12_latent_8/model-50.pt
