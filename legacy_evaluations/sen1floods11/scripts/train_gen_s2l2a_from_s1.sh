#!/bin/bash

echo -e "\033[31mTraining model on generated S2L2A ONLY (generated from S1) data\033[0m"

CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=sen1floods11_192_s2_only_without_b10 \
   dataset.root_path=./data/sen1floods11_v1.1/v1.1/outputs/GEN_S2L2A_FROM_S1 \
   encoder=unet_encoder \
   decoder=seg_unet \
   preprocessing=seg_default \
   criterion=cross_entropy \
   task=segmentation \
   finetune=True


echo -e "\033[31mTraining model on generated S2L2A and real S1 data (generated from S1) data\033[0m"

CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=sen1floods11_192_without_b10 \
   dataset.root_path=./data/sen1floods11_v1.1/v1.1/outputs/GEN_S2L2A_FROM_S1 \
   encoder=unet_encoder \
   decoder=seg_unet \
   preprocessing=seg_default \
   criterion=cross_entropy \
   task=segmentation \
   finetune=True

