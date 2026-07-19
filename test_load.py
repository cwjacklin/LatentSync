import torch
from omegaconf import OmegaConf

# Load your 1.6 config
config = OmegaConf.load("configs/unet/stage2_512.yaml")

# Fix: Read the correct keys from ByteDance's YAML structure
target_resolution = config.data.resolution
print(f"Config target resolution: {target_resolution}x{target_resolution}")

# Test loading the model onto your system
print("Loading model weights...")
try:
    state_dict = torch.load("checkpoints/latentsync_unet.pt", map_location="cpu")
    print("Success! Model weights file parsed into memory smoothly.")
    
    # Check the actual weight key count
    actual_keys = len(state_dict.keys()) if "state_dict" not in state_dict else len(state_dict["state_dict"].keys())
    print(f"Total parameter keys parsed: {actual_keys}")
    
except Exception as e:
    print(f"Failed to parse weights: {e}")


