python ddpm_prune.py \
--dataset cifar10 \
--model_path pretrained/ddpm_ema_cifar10 \
--save_path run/pruned/ddpm_cifar10_pruned \
--pruning_ratio 0.1 \
--batch_size 128 \
--pruner random \
--thr 0.05 \
--device cuda:2 \
