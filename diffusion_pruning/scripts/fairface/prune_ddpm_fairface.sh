python ddpm_prune.py \
--dataset /raid/akshay/satyabrat/fairface-img-margin025-trainval/fairface_dataset_64 \
--model_path "theunnecessarythings/ddpm-ema-fairface-64" \
--save_path run/pruned/ddpm_fairface_pruned_64_random_50 \
--pruning_ratio $1 \
--batch_size 64 \
--pruner random \
--thr 0.05 \
--device cuda:5 \

# --dataset /raid/akshay/satyabrat/fairface-img-margin025-trainval/fairface_dataset_64 --model_path "theunnecessarythings/ddpm-ema-fairface-64" --save_path run/pruned/ddpm_fairface_pruned_64_dp_25 --pruning_ratio 0.1 --batch_size 64 --pruner random --thr 0.05 --device cuda:7

# /raid/akshay/ashutosh/Diff-Pruning/ddpm_prune.py(147)<module>()
# /raid/akshay/.local/lib/python3.8/site-packages/torch_pruning/dependency.py(158)prune()
# /raid/akshay/.local/lib/python3.8/site-packages/torch_pruning/dependency.py(180)prune()

# /raid/akshay/.local/lib/python3.8/site-packages/torch_pruning/dependency.py(103)__call__()
# /raid/akshay/.local/lib/python3.8/site-packages/torch_pruning/dependency.py(107)__call__()

# /raid/akshay/.local/lib/python3.8/site-packages/torch_pruning/pruner/function.py(177)prune_out_channels()

