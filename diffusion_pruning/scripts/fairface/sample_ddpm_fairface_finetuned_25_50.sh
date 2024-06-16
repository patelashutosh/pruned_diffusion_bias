python ddpm_sample.py \
 --output_dir run/sample/ddpm_fairface_finetuned_25_50 \
 --batch_size 128 \
 --pruned_model_ckpt run/finetuned/ddpm_fairface_pruned_post_training_25_50k/pruned/unet_ema_pruned.pth \
 --model_path run/finetuned/ddpm_fairface_pruned_post_training_25_50k \
 --skip_type uniform \
 --total_samples 50000 \
 --seed 42 \