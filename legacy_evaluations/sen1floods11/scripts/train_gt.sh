#!/bin/bash

echo -e "\033[31mTraining model on S1 Ground Truth and S2 Ground Truth data\033[0m"

CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=sen1floods11_192 \
   encoder=unet_encoder \
   decoder=seg_unet \
   preprocessing=seg_default \
   criterion=cross_entropy \
   task=segmentation \
   finetune=True


echo -e "\033[31mTraining model on S1 Ground Truth data ONLY\033[0m"

CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=sen1floods11_192_s1_only \
   encoder=unet_encoder \
   decoder=seg_unet \
   preprocessing=seg_default \
   criterion=cross_entropy \
   task=segmentation \
   finetune=True

echo -e "\033[31mTraining model on S2 Ground Truth data ONLY\033[0m"

CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=sen1floods11_192_s2_only \
   encoder=unet_encoder \
   decoder=seg_unet \
   preprocessing=seg_default \
   criterion=cross_entropy \
   task=segmentation \
   finetune=True
