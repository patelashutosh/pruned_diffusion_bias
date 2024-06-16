# Compute the FID score of sampled images
python fid_score2.py run/sample/ddpm_cifar10_pruned_only  run/fid_stats_fairface_64_50k.npz --device cuda:0 --batch-size 256

# python fid_score.py run/sample/ddpm_cifar10_pruned run/fid_stats_cifar10.npz --device cuda:0 --batch-size 256
