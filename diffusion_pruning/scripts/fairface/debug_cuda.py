import torch

# Optionally set CUDA_LAUNCH_BLOCKING for better debugging
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Check GPU memory before loading the model
print(torch.cuda.memory_summary())
pruned_model_ckpt = "/raid/akshay/ashutosh/Diff-Pruning/run/finetuned/ddpm_fairface_pruned_finetuned_my_50/pruned/unet_ema_pruned-210000.pth"
# Load model on CPU first, then move to GPU
try:
    print("loading model")
    unet = torch.load(pruned_model_ckpt, map_location='cuda:6').eval()
    # unet.to('cuda')
    print("model loaded")
except RuntimeError as e:
    print(f"Failed to load model: {e}")
    torch.cuda.empty_cache()
    print("Cleared cache, trying to load again...")
    unet = torch.load(pruned_model_ckpt, map_location='cpu').eval()
    unet.to('cuda')

# Check GPU memory after loading the model
print(torch.cuda.memory_summary())
