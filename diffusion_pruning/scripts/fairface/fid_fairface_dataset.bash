# pre-compute the stats of CIFAR-10 dataset
python fid_score.py  --save-stats /raid/akshay/satyabrat/fairface-img-margin025-trainval/fairface_dataset_64/ run/fid_stats_fairface_64_full.npz  --device cuda:6 --batch-size 256
