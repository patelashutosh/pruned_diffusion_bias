python ddpm_prune2.py \
--dataset /raid/akshay/satyabrat/fairface-img-margin025-trainval/fairface_dataset_64 \
--model_path "theunnecessarythings/ddpm-ema-fairface-64" \
--save_path run/pruned/ddpm_fairface_pruned_64_random_2_50 \
--pruning_ratio 0.50 \
--batch_size 64 \
--pruner random \
--thr 0.05 \
--device cuda:5 \

#"/raid/akshay/ashutosh/Diff-Pruning/pretrained/ddpm_ema_fairface64"