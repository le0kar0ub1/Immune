import os
import hashlib
import json
import time

output_dir = "./data/assemblagePE"
samples = {}

def sha256_file(filepath):
    """Compute SHA-256 hash of a file's contents."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def rename_files_to_sha256(directory):
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            # Compute hash
            file_hash = sha256_file(filepath)

            new_path = os.path.join(directory, file_hash)

            # Rename only if the new name is different
            if filepath != new_path:
                os.rename(filepath, new_path)
                print(f"Renamed: {filename} -> {file_hash}")
            else:
                print(f"Skipped: {filename} -> {file_hash} (already hash named)")

def save_metadata(metadata_file):
  if not os.path.exists(output_dir):
      os.makedirs(output_dir)
  with open(metadata_file, "w") as f:
    json.dump(samples, f, indent=2)

def build_metadata(directory):
  for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    if os.path.isfile(filepath):
      samples[filename] = { "is_malware": False }

if __name__ == "__main__":
    # rename_files_to_sha256(f"{output_dir}/binaries")
    build_metadata(f"{output_dir}/binaries")
    save_metadata(f"{output_dir}/metadata.json")
