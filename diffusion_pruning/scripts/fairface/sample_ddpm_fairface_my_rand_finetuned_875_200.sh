python ddpm_sample1.py \
 --output_dir run/sample/ddpm_fairface_pruned_finetuned_my_rand_875_200k \
 --batch_size 32 \
 --pruned_model_ckpt run/finetuned/ddpm_fairface_pruned_finetuned_my_rand_875/pruned/unet_ema_pruned-200000.pth \
 --model_path run/finetuned/ddpm_fairface_pruned_finetuned_my_rand_875 \
 --skip_type uniform \
 --total_samples 50000 \
 --seed 1875 \