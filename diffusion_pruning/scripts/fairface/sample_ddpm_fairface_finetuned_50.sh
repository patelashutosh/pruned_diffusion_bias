python ddpm_sample.py \
 --output_dir run/sample/ddpm_fairface_finetuned_50 \
 --batch_size 128 \
 --pruned_model_ckpt run/finetuned/ddpm_fairface_pruned_post_training_50/pruned/unet_ema_pruned.pth \
 --model_path run/finetuned/ddpm_fairface_pruned_post_training_50 \
 --skip_type uniform \
 --total_samples 50000 \
 --seed 42 \