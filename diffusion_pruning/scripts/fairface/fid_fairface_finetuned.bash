# Compute the FID score of sampled images
python fid_score2.py run/sample/ddpm_fairface_pruned_finetuned_my_25_200k_2  run/fid_stats_fairface_64_full.npz --device cuda:6 --batch-size 256

