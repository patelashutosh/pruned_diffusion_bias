python finetune_gpu.py \
--config celeba.yml \
--exp run/sample/ddim_celeba_pruned_$3_$2 \
--sample \
--timesteps 100 \
--eta 0 \
--ni \
--doc sample \
--skip_type uniform  \
--pruning_ratio 0.0 \
--fid \
--use_ema \
--restore_from run/finetune_final/celeba_$3_$2_finetuned/logs/post_training/ckpt_200000.pth \
--device cuda:$1 \

python fid_score2.py run/sample/ddim_celeba_pruned_$3_$2 run/fid_stats_celeba_64_cropped_png.npz \
--device cuda:$1 \
--batch-size 256