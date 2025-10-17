import os
import sys
import tempfile
import shutil
from safetensors import safe_open
from safetensors.torch import save_file
import torch

def reshape_weight_scale(file_path: str):
    """Update all tensors ending with 'weight_scale' from [out_features] to [out_features, 1]."""
    new_tensors = {}
    updated = False

    with safe_open(file_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if k.endswith("weight_scale") and t.ndim == 1:
                print(f"  -> Reshaping {k}: {list(t.shape)} -> {[t.shape[0], 1]}")
                new_tensors[k] = t.unsqueeze(1)
                updated = True
            else:
                new_tensors[k] = t

    if updated:
        # Save to a temporary file first, then replace the original
        tmp_file = file_path + ".tmp"
        save_file(new_tensors, tmp_file)  # torch version
        shutil.move(tmp_file, file_path)
        print(f"✅ Updated file saved: {file_path}")
    else:
        print(f"ℹ️  No changes needed for {file_path}")


def process_folder(folder: str):
    """Process all .safetensors files in a folder."""
    files = [f for f in os.listdir(folder) if f.endswith(".safetensors")]

    if not files:
        print("No .safetensors files found in the folder.")
        return

    for fname in files:
        file_path = os.path.join(folder, fname)
        print(f"\nProcessing {fname} ...")
        reshape_weight_scale(file_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python update_weight_scale.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid folder.")
        sys.exit(1)

    process_folder(folder_path)
