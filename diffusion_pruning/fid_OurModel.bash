# pre-compute the stats of CIFAR-10 dataset
python fid_score.py  --save-stats /raid/akshay/satyabrat/fairface-img-margin025-trainval/train_64 run/fid_stats_fairface_64_50k.npz  --device cuda:0 --batch-size 256
