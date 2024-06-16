python ddpm_sample.py \
 --output_dir run/sample/ddpm_fairface_pruned_finetuned_my_l1norm_875_200k \
 --batch_size 64 \
 --pruned_model_ckpt run/finetuned/ddpm_fairface_pruned_finetuned_my_l1norm_875/pruned/unet_ema_pruned-200000.pth \
 --model_path run/finetuned/ddpm_fairface_pruned_finetuned_my_l1norm_875 \
 --skip_type uniform \
 --total_samples 50000 \
 --seed 2875 \