python ddpm_train_ff.py \
  --dataset="/raid/akshay/satyabrat/fairface-img-margin025-trainval/fairface_dataset_64" \
  --model_path="theunnecessarythings/ddpm-ema-fairface-64" \
  --resolution=64 \
  --output_dir="run/finetuned/ddpm_fairface_pretrained" \
  --train_batch_size=128 \
  --num_iters=400000 \
  --gradient_accumulation_steps=1 \
  --learning_rate=0.0001 \
  --lr_warmup_steps=0 \
  --save_model_steps 10000 \
  --dataloader_num_workers 8 \
  --adam_weight_decay 0.00 \
  --ema_max_decay 0.9999 \
  --dropout 0.01 \
  --use_ema \