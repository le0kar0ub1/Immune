import boto3
import zlib
import os
import sqlite3
import hashlib
import json
import time

s3 = boto3.client("s3")
s3_bucket = "sorel-20m"
s3_prefix = "09-DEC-2020/binaries"

meta_db_http_url = "https://sorel-20m.s3.amazonaws.com/09-DEC-2020/processed-data/meta.db"
meta_db_s3_uri = "09-DEC-2020/processed-data/meta.db"

sqlite_db_path = "data/sorel20M/meta.db"

output_dir="data/sorel20M"

samples = {}

start_time = time.time()

def sample_balanced_hashes(offset=0, total_samples=20000):
    """
    Randomly select samples from SOREL-20M meta.db with balanced malware/benign.

    Args:
        sqlite_db_path (str): Path to SQLite meta database.
        total_samples (int): Total number of samples desired.

    Returns:
        list[str]: List of SHA256 hashes.
    """
    half = total_samples // 2

    conn = sqlite3.connect(sqlite_db_path)
    cursor = conn.cursor()

    # Sample malware
    cursor.execute(f"""
        SELECT sha256 FROM meta
        WHERE is_malware=1
        LIMIT {half}
        OFFSET {offset}
    """)
    for row in cursor.fetchall():
      samples[row[0]] = { "is_malware": 1 }

    # Sample benign
    # cursor.execute(f"""
    #     SELECT sha256 FROM meta
    #     WHERE is_malware=0
    #     LIMIT {half}
    #     OFFSET {offset}
    # """)
    # for row in cursor.fetchall():
    #   samples[row[0]] = { "is_malware": 0 }

    conn.close()

# def download_sorel_binaries(num_files=1000, continuationToken=None, output_dir="data/sorel20M/binaries"):
#     """
#     Download and decompress binaries from the SOREL-20M dataset.
    
#     Args:
#         num_files (int): Number of files to download.
#         continuationToken (str): Offset in listing (skip first N files).
#         output_dir (str): Local directory to save binaries.
#     """
#     paginator = s3.get_paginator("list_objects_v2")

#     os.makedirs(output_dir, exist_ok=True)

#     files_downloaded = 0

#     # Paginate through S3 keys
#     for page in paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix, PaginationConfig={'MaxItems': num_files, 'StartingToken': continuationToken}):
#         if 'NextContinuationToken' in page:
#           continuationToken = page['NextContinuationToken']
#         else:
#           continuationToken = None
#         for obj in page.get("Contents", []):
#             if files_downloaded >= num_files:
#                 return  # done

#             key = obj["Key"]
#             hashes = key.split("/")[-1]

#             samples[hashes] = {}

#             if os.path.exists(os.path.join(output_dir, os.path.basename(key))):
#                 print(f"Skipping {key}, already exists.")
#                 continue

#             print(f"Downloading {key}...")

#             # Fetch compressed object
#             response = s3.get_object(Bucket=s3_bucket, Key=key)
#             compressed_data = response["Body"].read()

#             try:
#                 binary_data = zlib.decompress(compressed_data)
#             except zlib.error:
#                 print(f"⚠️ Skipping {key}, decompression failed.")
#                 continue

#             # Save locally
#             filename = os.path.join(output_dir, os.path.basename(key))
#             with open(filename, "wb") as f:
#                 f.write(binary_data)

#             print(f"Saved: {filename}")
#             files_downloaded += 1

#     return continuationToken

def save_metadata():
  if not os.path.exists(output_dir):
      os.makedirs(output_dir)
  with open(f"{output_dir}/metadata.json", "w") as f:
    json.dump(samples, f, indent=2)
  print(f"Saved metadata to {output_dir}/metadata.json in {time.time() - start_time} seconds")

def download_sorel_binaries():
    keys = list(samples.keys())
    print(f"Downloading {len(keys)} samples...")
    for i, h in enumerate(keys):
      try:
        key = f"{s3_prefix}/{h}"
        if os.path.exists(os.path.join(output_dir, "binaries", os.path.basename(key))):
            print(f"  Skipping {key} ({i+1}/{len(keys)}), already exists.")
            continue
        response = s3.get_object(Bucket=s3_bucket, Key=key)
        compressed_data = response["Body"].read()
        binary_data = zlib.decompress(compressed_data)
        with open(os.path.join(output_dir, "binaries", os.path.basename(key)), "wb") as f:
          f.write(binary_data)
        print(f"  -> Downloaded {key} ({i+1}/{len(keys)})")
        if i % 100 == 0:
          save_metadata()
      except Exception as e:
        del samples[h]
        print(f"  -> Error downloading {h}: {e}, deleting from samples")


# def label_files(sqlite_db_path, batch_size=1000):
#     """
#     Label files as malware (1) or benign (0) using SOREL-20M SQLite database.

#     Args:
#         file_names (list[str]): List of local file paths.
#         sqlite_db_path (str): Path to SQLite database (with `meta` table).

#     Returns:
#         dict: {file_path: label} where label ∈ {0,1,None}
#               (None if hash not found in DB).
#     """
#     if not os.path.exists(sqlite_db_path):
#         print(f"SQLite database not found at {sqlite_db_path}, downloading...")
#         s3.download_file(s3_bucket, meta_db_s3_uri, sqlite_db_path)
#         print(f"Downloaded SQLite database to {sqlite_db_path}")
    
#     # Connect to SQLite DB
#     conn = sqlite3.connect(sqlite_db_path)
#     cursor = conn.cursor()

#     # PRAGMA table_info(meta);
#     # 0|sha256|TEXT|0||1
#     # 1|is_malware|SMALLINT|0||0
#     # 2|rl_fs_t|DOUBLE|0||0
#     # 3|rl_ls_const_positives|INTEGER|0||0
#     # 4|adware|INTEGER|0||0
#     # 5|flooder|INTEGER|0||0
#     # 6|ransomware|INTEGER|0||0
#     # 7|dropper|INTEGER|0||0
#     # 8|spyware|INTEGER|0||0
#     # 9|packed|INTEGER|0||0
#     # 10|crypto_miner|INTEGER|0||0
#     # 11|file_infector|INTEGER|0||0
#     # 12|installer|INTEGER|0||0
#     # 13|worm|INTEGER|0||0
#     # 14|downloader|INTEGER|0||0
#     for i in range(0, len(samples.keys()), batch_size):
#         batch = list(samples.keys())[i:i+batch_size]
#         placeholders = ",".join("?" * len(batch))
#         query = f"SELECT sha256, is_malware, rl_fs_t, rl_ls_const_positives, adware, flooder, ransomware, dropper, spyware, packed, crypto_miner, file_infector, installer, worm, downloader FROM meta WHERE sha256 IN ({placeholders})"
#         cursor.execute(query, batch)
#         for sha256, is_malware, rl_fs_t, rl_ls_const_positives, adware, flooder, ransomware, dropper, spyware, packed, crypto_miner, file_infector, installer, worm, downloader in cursor.fetchall():
#             samples[sha256] = {
#                 "is_malware": int(is_malware),
#                 "rl_fs_t": float(rl_fs_t),
#                 "rl_ls_const_positives": int(rl_ls_const_positives),
#                 "adware": int(adware),
#                 "flooder": int(flooder),
#                 "ransomware": int(ransomware),
#                 "dropper": int(dropper),
#                 "spyware": int(spyware),
#                 "packed": int(packed),
#                 "crypto_miner": int(crypto_miner),
#                 "file_infector": int(file_infector),
#                 "installer": int(installer),
#                 "worm": int(worm),
#                 "downloader": int(downloader),
#             }
#     malware_count = sum(1 for x in samples.values() if x["is_malware"] == 1)
#     benign_count = sum(1 for x in samples.values() if x["is_malware"] == 0)
#     print(f"Malware count: {malware_count}")
#     print(f"Benign count: {benign_count}")
#     print(f"Total count: {len(samples.keys())}")
#     print(f"Malware ratio: {malware_count / len(samples.keys())}")
#     print(f"Benign ratio: {benign_count / len(samples.keys())}")
#     conn.close()

sample_balanced_hashes(offset=0, total_samples=20000)
download_sorel_binaries()

print(f"Total time taken: {time.time() - start_time} seconds")
print(f"Total samples: {len(samples.keys())}")
print(f"Malware samples: {sum(1 for x in samples.values() if x['is_malware'] == 1)}")
print(f"Benign samples: {sum(1 for x in samples.values() if x['is_malware'] == 0)}")

save_metadata()
