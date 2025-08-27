import os
import hashlib
import json
import time

output_dir = "./data/DikeDataset-ValTest"

def save_metadata(metadata_file, samples):
  if not os.path.exists(output_dir):
      os.makedirs(output_dir)
  with open(metadata_file, "w") as f:
    json.dump(samples, f, indent=2)

def build_metadata(directory, is_malware):
  samples = {}
  for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    if os.path.isfile(filepath):
      samples[filename] = { "is_malware": is_malware }
  return samples

if __name__ == "__main__":
    benign = build_metadata(f"{output_dir}/benign/binaries", False)
    save_metadata(f"{output_dir}/benign/metadata.json", benign)
    malware = build_metadata(f"{output_dir}/malware/binaries", True)
    save_metadata(f"{output_dir}/malware/metadata.json", malware)
