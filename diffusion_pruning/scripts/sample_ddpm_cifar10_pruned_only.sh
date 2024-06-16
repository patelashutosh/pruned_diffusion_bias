python ddpm_sample.py \
 --output_dir run/sample/ddpm_cifar10_pruned_only \
 --batch_size 128 \
 --pruned_model_ckpt run/pruned/ddpm_cifar10_pruned/pruned/unet_pruned.pth \
 --model_path run/pruned/ddpm_cifar10_pruned/ \
 --skip_type uniform \
 --total_samples 500 \