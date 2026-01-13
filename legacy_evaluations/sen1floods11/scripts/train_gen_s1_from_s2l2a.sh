#!/bin/bash

echo -e "\033[31mTraining model on generated S1 ONLY (generated from S2 L2A) data\033[0m"

CUDA_VISIBLE_DEVICES=3 torchrun --nnodes=1 --nproc_per_node=1 --master_port=29503 pangaea/run.py \
   --config-name=train \
   dataset=sen1floods11_192_s1_only \
   dataset.root_path=./data/sen1floods11_v1.1/v1.1/outputs/GEN_S1_FROM_S2L2A \
   encoder=unet_encoder \
   decoder=seg_unet \
   preprocessing=seg_default \
   criterion=cross_entropy \
   task=segmentation \
   finetune=True


echo -e "\033[31mTraining model on generated S1 and real S2 data (generated from S2 L2A) data\033[0m"

CUDA_VISIBLE_DEVICES=3 torchrun --nnodes=1 --nproc_per_node=1 --master_port=29503 pangaea/run.py \
   --config-name=train \
   dataset=sen1floods11_192 \
   dataset.root_path=./data/sen1floods11_v1.1/v1.1/outputs/GEN_S1_FROM_S2L2A \
   encoder=unet_encoder \
   decoder=seg_unet \
   preprocessing=seg_default \
   criterion=cross_entropy \
   task=segmentation \
   finetune=True

