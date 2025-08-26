#!/usr/bin/env python3
"""Feature extraction script using the Immune package."""

import argparse
import concurrent.futures
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.append(str(Path(__file__).parent / "Immune"))

from Immune.features.extractor import FeatureExtractor
from Immune.features.models import BinaryFeatures

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def save_features(input_dir: Path, features: Dict[str, BinaryFeatures]) -> None:
    """Save features to file."""
    logger.info(f"\nSaving features to {input_dir}\n")
    with open(os.path.join(input_dir, "features.json"), "w") as f:
        json.dump({file_path: feature.to_dict() for file_path, feature in features.items()}, f) #, indent=2)

def load_features(input_dir: Path) -> Dict[str, BinaryFeatures]:
    """Load features from file."""
    if not os.path.exists(os.path.join(input_dir, "features.json")):
        return {}
    logger.info(f"\nLoading features from {input_dir}\n")
    with open(os.path.join(input_dir, "features.json"), "r") as f:
        return {file_path: BinaryFeatures.from_dict(feature) for file_path, feature in json.load(f).items()}

def process_single_file(args):
    """Process a single file - extracted to be thread-safe."""
    file_path, is_malware, extractor, i, total_files = args
    
    try:
        logger.info(f"Processing file {i}/{total_files}: {file_path.name}")
        
        # Extract features
        current_features = extractor.extract_features(file_path, is_malware=is_malware)
        
        # Return success result
        return file_path.name, current_features, True, None
        
    except Exception as e:
        logger.error(f"Failed to extract features from {file_path.name}: {e}")
        return file_path.name, None, False, str(e)

def extract_features_from_directory(
    input_dir: Path,
    is_malware: bool,
    output_file: Optional[Path] = None,
    file_extensions: Optional[List[str]] = None,
    max_files: Optional[int] = None,
) -> List[BinaryFeatures]:
    """Extract features from all files in a directory.

    Args:
        input_dir: Directory containing files to analyze
        output_file: Optional path to save results as JSON
        file_extensions: Optional list of file extensions to process (e.g., ['.exe', '.dll'])
        max_files: Optional maximum number of files to process

    Returns:
        Tuple of (file_info_list, features_list)
    """
    binaries_dir = Path(os.path.join(input_dir, "binaries"))

    if not binaries_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if not binaries_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {binaries_dir}")

    features = load_features(input_dir)

    # Initialize feature extractor
    extractor = FeatureExtractor()

    # Get list of files to process
    files_to_process = []
    if file_extensions:
        for ext in file_extensions:
            files_to_process.extend(binaries_dir.glob(f"*{ext}"))
            files_to_process.extend(binaries_dir.glob(f"*{ext.upper()}"))
    else:
        # Process all files
        files_to_process = [f for f in binaries_dir.iterdir() if f.is_file()]

    # Remove duplicates and limit if specified
    files_to_process = list(set(files_to_process))
    if max_files:
        files_to_process = files_to_process[:max_files]

    logger.info(f"Found {len(files_to_process)} files to process")

    # Thread-safe counter for successful and failed extractions
    successful_extractions = 0
    failed_extractions = 0
    
    # Prepare arguments for threading
    thread_args = []
    for i, file_path in enumerate(files_to_process, 1):
        if file_path.name in features:
            logger.info(f"Skipping file {file_path.name} because it already exists")
            continue
            
        thread_args.append((file_path, is_malware, extractor, i, len(files_to_process)))
    
    # Process files in parallel with 6 processes (utilizing multiple CPU cores)
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_single_file, args): args for args in thread_args}
        
        # Process completed tasks
        for future in concurrent.futures.as_completed(future_to_file):
            file_name, current_features, success, error = future.result()
            
            if success:
                features[file_name] = current_features
                successful_extractions += 1
                logger.info(f"Successfully extracted features from {file_name}")
                
                if successful_extractions % 200 == 0:
                    save_features(input_dir, features)
            else:
                failed_extractions += 1

    logger.info(
        f"Feature extraction completed: {successful_extractions} successful, {failed_extractions} failed"
    )
    save_features(input_dir, features)
    return features


def save_results(features_list: List[BinaryFeatures], output_file: Path) -> None:
    """Save extraction results to file.

    Args:
        file_info_list: List of file information dictionaries
        features_list: List of extracted features
        output_file: Path to save the results
    """
    results = []

    for feature in features_list:
        results.append(feature.to_dict())

    # Save to JSON file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {output_file}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Extract features from files using the Immune package"
    )
    parser.add_argument(
        "-im",
        "--input_dir_malware",
        type=Path,
        help="Directory containing malware files to analyze",
        default="data/sorel20M",
    )
    parser.add_argument(
        "-ib",
        "--input_dir_benign",
        type=Path,
        help="Directory containing benign files to analyze",
        default="data/assemblagePE",
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="Output file to save results (JSON format)"
    )
    parser.add_argument(
        "-e",
        "--extensions",
        nargs="+",
        help="File extensions to process (e.g., .exe .dll)",
        default=None,
    )
    parser.add_argument(
        "-m", "--max-files", type=int, help="Maximum number of files to process", default=None
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging", default=False
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Extract features
        malware_features_list = extract_features_from_directory(
            input_dir=args.input_dir_malware,
            is_malware=True,
            output_file=args.output,
            file_extensions=args.extensions,
            max_files=args.max_files,
        )

        benign_features_list = extract_features_from_directory(
            input_dir=args.input_dir_benign,
            is_malware=False,
            output_file=args.output,
            file_extensions=args.extensions,
            max_files=args.max_files,
        )

        features_list = {}
        feature_list.update(benign_feature_list)
        malware_list.update(malware_feature_list)

        logger.info(f"Saving full results to {args.output}")

        save_results(features_list, args.output)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Full results saved to {args.output}")

        logger.info("Feature extraction summary:")
        logger.info(f"  Total files: {len(features_list)}")
        logger.info(f"  Results saved to: {args.output}")

    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
