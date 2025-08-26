import logging
import requests
import json
import os
import pyzipper
import io
from pathlib import Path

logger = logging.getLogger(__name__)

class MalwareScraper:
    def __init__(self, output_dir="samples", limit=1000):
        self.output_dir = output_dir
        self.limit = limit
        self.query_url = "https://mb-api.abuse.ch/api/v1/"
        self.api_key = os.getenv("MALWAREBAZAAR_API_KEY")
        if not self.api_key:
            raise ValueError("MALWAREBAZAAR_API_KEY is not set")

    def download_malware_samples(self):
        """Download diverse malware samples from MalwareBazaar API"""

        # Create output directory
        Path(self.output_dir).mkdir(exist_ok=True)

        # Get recent samples
        payload = {
            "query": "get_file_type",
            "file_type": "exe",
            "limit": self.limit
        }

        logger.info(f"Fetching metadata for {self.limit} samples...")
        response = requests.post(self.query_url, data=payload, headers={"Auth-Key": self.api_key})
        data = response.json()

        if data["query_status"] != "ok":
            logger.error(f"Query failed: {data}")
            return

        downloaded = 0

        for sample in data["data"]:
            sha256 = sample["sha256_hash"]
            family = sample.get("signature", "unknown")

            if os.path.exists(os.path.join(self.output_dir, sha256 + ".bin")):
                logger.info(f"  Skipping {sha256} because it already exists")
                continue

            download_payload = {
                "query": "get_file",
                "sha256_hash": sha256
            }

            logger.info(f"  Downloading {sha256} ({family})")
            dl_response = requests.post(self.query_url, data=download_payload, headers={"Auth-Key": self.api_key})

            if dl_response.status_code == 200:
                try:
                    with pyzipper.AESZipFile(io.BytesIO(dl_response.content), 'r') as zip_ref:
                        zip_ref.extractall(self.output_dir, pwd=b"infected")
                    
                    # Find the extracted binary file
                    extracted_files = [f for f in os.listdir(self.output_dir) if f.endswith('.exe')]
                    
                    if extracted_files:
                        # Rename the extracted file to use SHA256
                        extracted_file = extracted_files[0]
                        old_path = os.path.join(self.output_dir, extracted_file)
                        new_path = os.path.join(self.output_dir, f"{sha256}.bin")
                        os.rename(old_path, new_path)
                        
                        # Save metadata
                        metadata_filename = f"{sha256}.json"
                        metadata_filepath = os.path.join(self.output_dir, metadata_filename)
                        with open(metadata_filepath, "w") as f:
                            json.dump(sample, f)

                        logger.info(f"  Downloaded and extracted: {sha256}.bin ({family})")
                        downloaded += 1
                    else:
                        logger.warning(f"No executable found in ZIP for {sha256}")
                        
                except pyzipper.BadZipFile:
                    logger.error(f"Invalid ZIP file for {sha256}")
                except Exception as e:
                    logger.error(f"Error extracting {sha256}: {e}")

        logger.info(f"Downloaded {downloaded} samples")
