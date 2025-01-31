python ddpm_train_ff.py \
  --dataset="/raid/akshay/satyabrat/fairface-img-margin025-trainval/fairface_dataset_64" \
  --model_path="run/pruned/ddpm_fairface_pruned_64_dp_125_custom" \
  --pruned_model_ckpt="run/pruned/ddpm_fairface_pruned_64_dp_125_custom/pruned/unet_pruned.pth" \
  --resolution=64 \
  --output_dir="run/finetuned/ddpm_fairface_pruned_post_training_125_50k" \
  --train_batch_size=128 \
  --num_iters=50000 \
  --gradient_accumulation_steps=1 \
  --learning_rate=0.0001 \
  --lr_warmup_steps=0 \
  --save_model_steps 5000 \
  --dataloader_num_workers 8 \
  --adam_weight_decay 0.00 \
  --ema_max_decay 0.9999 \
  --dropout 0.01 \
  --use_ema \