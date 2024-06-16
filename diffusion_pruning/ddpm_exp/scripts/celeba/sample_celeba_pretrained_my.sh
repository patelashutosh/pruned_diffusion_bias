python prune_gpu.py \
--config celeba.yml \
--exp run/sample/ddim_celeba_official_my \
--sample \
--restore_from "run/pretrain_final/celeba_trained/logs/training/ckpt_100000.pth" \
--timesteps 100 \
--eta 0 \
--ni \
--doc official \
--skip_type uniform  \
--pruning_ratio 0.0 \
--fid \
--use_ema \
--device cuda:2 \