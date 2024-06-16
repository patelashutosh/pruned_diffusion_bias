python ddpm_sample.py \
 --output_dir run/sample/ddpm_fairface_pruned_finetuned_my_rand_25_200k \
 --batch_size 64 \
 --pruned_model_ckpt run/finetuned/ddpm_fairface_pruned_finetuned_my_rand_25_2/pruned/unet_ema_pruned-80000.pth \
 --model_path run/finetuned/ddpm_fairface_pruned_finetuned_my_rand_25_2 \
 --skip_type uniform \
 --total_samples 50000 \
 --seed 225 \