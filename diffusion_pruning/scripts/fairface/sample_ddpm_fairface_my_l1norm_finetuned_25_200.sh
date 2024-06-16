python ddpm_sample7.py \
 --output_dir run/sample/ddpm_fairface_pruned_finetuned_my_l1norm_25_200k \
 --batch_size 128 \
 --pruned_model_ckpt run/finetuned/ddpm_fairface_pruned_finetuned_my_l1norm_25/pruned/unet_ema_pruned-200000.pth \
 --model_path run/finetuned/ddpm_fairface_pruned_finetuned_my_l1norm_25 \
 --skip_type uniform \
 --total_samples 50000 \
 --seed 125 \