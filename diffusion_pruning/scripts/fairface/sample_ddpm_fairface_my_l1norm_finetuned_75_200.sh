python ddpm_sample6.py \
 --output_dir run/sample/ddpm_fairface_pruned_finetuned_my_l1norm_75_200k \
 --batch_size 64 \
 --pruned_model_ckpt run/finetuned/ddpm_fairface_pruned_finetuned_my_l1norm_75/pruned/unet_ema_pruned-200000.pth \
 --model_path run/finetuned//ddpm_fairface_pruned_finetuned_my_l1norm_75 \
 --skip_type uniform \
 --total_samples 50000 \
 --seed 275 \