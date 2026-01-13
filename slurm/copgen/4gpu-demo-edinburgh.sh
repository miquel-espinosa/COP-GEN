#!/bin/bash

export NUM_GPUS=4

# Print the name of the host
export HOSTNAME=$(hostname)
echo $HOSTNAME

# Clean up checkpoints
# ./slurm/clean_checkpoints.sh workdir/copgen/discrete/demo_edinburgh/default/ckpts

export OMP_NUM_THREADS=4
accelerate launch \
            --multi_gpu \
            --num_processes 4 \
            --mixed_precision fp16 \
            train_copgen_discrete.py \
                --config="configs/copgen/discrete/demo_edinburgh.py"
