#!/bin/bash

# Execute the Python script with the provided arguments
python prune_gpu.py \
--config "celeba.yml" \
--timesteps "100" \
--eta "0" \
--ni \
--doc "post_training" \
--skip_type "quad" \
--pruning_ratio "$2" \
--use_ema \
--restore_from "pretrained/celeba_ddpm_ckpt.pth" \
--pruner "$3" \
--save_pruned_model "run/pruned_final/celeba_$3_$2.pth" \
--taylor_batch_size "64" \
--thr "0.05" \
--device cuda:$1 \