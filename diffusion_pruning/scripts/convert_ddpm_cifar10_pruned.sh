python ddpm_convert.py \
 --output_dir run/converted_model/ddpm_cifar10_pruned \
 --batch_size 128 \
 --pruned_model_ckpt run/finetuned/ddpm_cifar10_pruned_post_training/pruned/unet_ema_pruned.pth \
 --model_path run/finetuned/ddpm_cifar10_pruned_post_training \
 --skip_type uniform
