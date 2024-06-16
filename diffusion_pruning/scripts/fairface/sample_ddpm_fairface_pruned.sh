python ddpm_sample.py \
 --output_dir run/sample/ddpm_fairface_pruned_64_dp_125 \
 --batch_size 128 \
 --pruned_model_ckpt run/pruned/ddpm_fairface_pruned_64_dp_125/pruned/unet_pruned.pth \
 --model_path run/pruned/ddpm_fairface_pruned_64_dp_125 \
 --skip_type uniform \
 --total_samples 500 \