python ddpm_sample.py \
 --output_dir run/sample/ddpm_fairface_finetuned_custom3_4l_64_dp_125 \
 --batch_size 128 \
 --pruned_model_ckpt run/finetuned/ddpm_fairface_pruned_post_training_custom3_4lsteps/pruned/unet_ema_pruned-300000.pth \
 --model_path run/finetuned/ddpm_fairface_pruned_post_training_custom3_4lsteps \
 --skip_type uniform \
 --total_samples 50000 \
 --seed 42 \