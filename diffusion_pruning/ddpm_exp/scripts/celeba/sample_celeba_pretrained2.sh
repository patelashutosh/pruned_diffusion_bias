python prune_gpu.py \
--config celeba.yml \
--exp run/sample/ddim_celeba_official \
--sample \
--restore_from "pretrained/celeba_ddpm_ckpt.pth" \
--timesteps 100 \
--eta 0 \
--ni \
--doc official \
--skip_type uniform  \
--pruning_ratio 0.0 \
--fid \
--use_ema \
--device cuda:4 \