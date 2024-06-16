python ddpm_sample.py \
 --output_dir run/sample/ddpm_fairface_finetuned_50_350 \
 --batch_size 128 \
 --pruned_model_ckpt run/finetuned/ddpm_fairface_pruned_post_training_50/pruned/unet_ema_pruned-350000.pth \
 --model_path run/finetuned/ddpm_fairface_pruned_post_training_50 \
 --skip_type uniform \
 --total_samples 500 \
 --seed 10 \