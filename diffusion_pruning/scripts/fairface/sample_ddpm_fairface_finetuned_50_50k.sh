python ddpm_sample.py \
 --output_dir run/sample/ddpm_fairface_finetuned_50_100k_60 \
 --batch_size 128 \
 --pruned_model_ckpt /raid/akshay/ashutosh/Diff-Pruning/run/finetuned/ddpm_fairface_pruned_post_training_50_100k/pruned/unet_ema_pruned-60000.pth \
 --model_path run/finetuned/ddpm_fairface_pruned_post_training_50_100k \
 --skip_type uniform \
 --total_samples 50000 \
 --seed 42 \