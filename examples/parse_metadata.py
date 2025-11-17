import os
import csv
import argparse
from safetensors import safe_open

def parse_safetensors(folder: str, output_csv: str):
    rows = []
    if output_csv == "":
        # parse model base name from folder
        model_base_name = os.path.basename(os.path.normpath(folder))
        output_csv = f"{model_base_name}_safetensors_info.csv"
        print(f"No output file specified. Using default: {output_csv}")

    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".safetensors"):
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")
                try:
                    with safe_open(file_path, framework="pt") as f:
                        keys = list(f.keys())
                        print(f"  Found {len(keys)} tensors")
                        for tensor_name in keys:
                            tensor = f.get_tensor(tensor_name)
                            dtype = str(tensor.dtype)
                            shape = list(tensor.shape)
                            tensor_min, tensor_max = "N/A", "N/A"
                            try:
                                tensor_min, tensor_max = tensor.min().item(), tensor.max().item()
                            except:
                                pass
                            tensor_min, tensor_max = str(tensor_min), str(tensor_max)
                            rows.append({
                                "file": os.path.relpath(file_path, folder),
                                "tensor": tensor_name,
                                "dtype": dtype,
                                "shape": str(shape),
                                "min": tensor_min,
                                "max": tensor_max
                            })
                except Exception as e:
                    print(f"  Error reading {file_path}: {e}")

    print(f"\nSaving results to {output_csv} ...")
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["file", "tensor", "dtype", "shape"
                                                     , "min", "max"
                                                     ])
        writer.writeheader()
        writer.writerows(rows)
    print("Done ✅")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse tensor dtype/shape from safetensors files.")
    parser.add_argument("folder", help="Path to the folder containing safetensors files")
    parser.add_argument("-o", "--output", default="",
                        help="Output CSV file name (default: safetensors_info.csv)")
    args = parser.parse_args()

    parse_safetensors(args.folder, args.output)
