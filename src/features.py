#!/usr/bin/env python3
"""Feature extraction script using the Immune package."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from Immune.data.feature_extractor import FeatureExtractor
from Immune.data.models import MalwareFeatures

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_features_from_directory(
    input_dir: Path,
    output_file: Optional[Path] = None,
    file_extensions: Optional[List[str]] = None,
    max_files: Optional[int] = None,
) -> Tuple[List[Dict], List[MalwareFeatures]]:
    """Extract features from all files in a directory.

    Args:
        input_dir: Directory containing files to analyze
        output_file: Optional path to save results as JSON
        file_extensions: Optional list of file extensions to process (e.g., ['.exe', '.dll'])
        max_files: Optional maximum number of files to process

    Returns:
        Tuple of (file_info_list, features_list)
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")

    # Initialize feature extractor
    extractor = FeatureExtractor()

    # Get list of files to process
    files_to_process = []
    if file_extensions:
        for ext in file_extensions:
            files_to_process.extend(input_dir.glob(f"*{ext}"))
            files_to_process.extend(input_dir.glob(f"*{ext.upper()}"))
    else:
        # Process all files
        files_to_process = [f for f in input_dir.iterdir() if f.is_file()]

    # Remove duplicates and limit if specified
    files_to_process = list(set(files_to_process))
    if max_files:
        files_to_process = files_to_process[:max_files]

    logger.info(f"Found {len(files_to_process)} files to process")

    # Process files and extract features
    file_info_list = []
    features_list = []
    successful_extractions = 0
    failed_extractions = 0

    for i, file_path in enumerate(files_to_process, 1):
        try:
            logger.info(f"Processing file {i}/{len(files_to_process)}: {file_path.name}")

            # Extract features
            features = extractor.extract_features(file_path)

            # Store file info
            file_info = {
                "filename": file_path.name,
                "filepath": str(file_path),
                "filesize": file_path.stat().st_size,
                "status": "success",
            }

            file_info_list.append(file_info)
            features_list.append(features)
            successful_extractions += 1

            logger.info(f"Successfully extracted features from {file_path.name}")

        except Exception as e:
            logger.error(f"Failed to extract features from {file_path.name}: {e}")

            # Store failed file info
            file_info = {
                "filename": file_path.name,
                "filepath": str(file_path),
                "filesize": file_path.stat().st_size if file_path.exists() else 0,
                "status": "failed",
                "error": str(e),
            }

            file_info_list.append(file_info)
            features_list.append(None)
            failed_extractions += 1

    logger.info(
        f"Feature extraction completed: {successful_extractions} successful, {failed_extractions} failed"
    )

    # Save results if output file specified
    if output_file:
        save_results(file_info_list, features_list, output_file)

    return file_info_list, features_list


def save_results(
    file_info_list: List[Dict], features_list: List[MalwareFeatures], output_file: Path
) -> None:
    """Save extraction results to file.

    Args:
        file_info_list: List of file information dictionaries
        features_list: List of extracted features
        output_file: Path to save the results
    """
    # Prepare data for saving
    results = {
        "extraction_info": {
            "total_files": len(file_info_list),
            "successful_extractions": sum(
                1 for info in file_info_list if info["status"] == "success"
            ),
            "failed_extractions": sum(1 for info in file_info_list if info["status"] == "failed"),
        },
        "files": [],
    }

    for file_info, features in zip(file_info_list, features_list, strict=False):
        file_result = file_info.copy()

        if features is not None:
            # Convert features to serializable format
            file_result["features"] = {
                "pe_features": features.pe_features.to_array().tolist()
                if hasattr(features.pe_features, "to_array")
                else None,
                "byte_histogram": features.byte_histogram.histogram.tolist()
                if hasattr(features.byte_histogram, "histogram")
                else None,
                "api_features": {
                    "file_apis": features.api_features.file_apis
                    if hasattr(features.api_features, "file_apis")
                    else {},
                    "registry_apis": features.api_features.registry_apis
                    if hasattr(features.api_features, "registry_apis")
                    else {},
                    "network_apis": features.api_features.network_apis
                    if hasattr(features.api_features, "network_apis")
                    else {},
                    "process_apis": features.api_features.process_apis
                    if hasattr(features.api_features, "process_apis")
                    else {},
                    "memory_apis": features.api_features.memory_apis
                    if hasattr(features.api_features, "memory_apis")
                    else {},
                    "system_apis": features.api_features.system_apis
                    if hasattr(features.api_features, "system_apis")
                    else {},
                    "crypto_apis": features.api_features.crypto_apis
                    if hasattr(features.api_features, "crypto_apis")
                    else {},
                    "anti_debug_apis": features.api_features.anti_debug_apis
                    if hasattr(features.api_features, "anti_debug_apis")
                    else {},
                    "total_strings": getattr(features.api_features, "total_strings", 0),
                    "avg_string_length": getattr(features.api_features, "avg_string_length", 0.0),
                    "max_string_length": getattr(features.api_features, "max_string_length", 0),
                    "min_string_length": getattr(features.api_features, "min_string_length", 0),
                }
                if hasattr(features, "api_features")
                else {},
            }
        else:
            file_result["features"] = None

        results["files"].append(file_result)

    # Save to JSON file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {output_file}")


def create_feature_matrix(features_list: List[MalwareFeatures]) -> np.ndarray:
    """Create a feature matrix from extracted features.

    Args:
        features_list: List of extracted features

    Returns:
        Feature matrix as numpy array
    """
    feature_vectors = []

    for features in features_list:
        if features is None:
            # Create zero vector for failed extractions
            feature_vector = np.zeros(356, dtype=np.float32)
        else:
            # Combine all feature types into a single vector
            pe_vector = (
                features.pe_features.to_array()
                if hasattr(features.pe_features, "to_array")
                else np.zeros(50, dtype=np.float32)
            )
            byte_vector = (
                features.byte_histogram.histogram
                if hasattr(features.byte_histogram, "histogram")
                else np.zeros(256, dtype=np.float32)
            )

            # For API features, create a simple vector (you can expand this)
            api_vector = np.array(
                [
                    getattr(features.api_features, "total_strings", 0),
                    getattr(features.api_features, "avg_string_length", 0.0),
                    getattr(features.api_features, "max_string_length", 0),
                    getattr(features.api_features, "min_string_length", 0),
                ],
                dtype=np.float32,
            )

            # Pad or truncate to ensure consistent size
            if len(pe_vector) < 50:
                pe_vector = np.pad(pe_vector, (0, 50 - len(pe_vector)), "constant")
            elif len(pe_vector) > 50:
                pe_vector = pe_vector[:50]

            if len(byte_vector) < 256:
                byte_vector = np.pad(byte_vector, (0, 256 - len(byte_vector)), "constant")
            elif len(byte_vector) > 256:
                byte_vector = byte_vector[:256]

            if len(api_vector) < 50:
                api_vector = np.pad(api_vector, (0, 50 - len(api_vector)), "constant")
            elif len(api_vector) > 50:
                api_vector = api_vector[:50]

            # Combine all features
            feature_vector = np.concatenate([pe_vector, byte_vector, api_vector])

        feature_vectors.append(feature_vector)

    return np.array(feature_vectors)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Extract features from files using the Immune package"
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing files to analyze")
    parser.add_argument(
        "-o", "--output", type=Path, help="Output file to save results (JSON format)"
    )
    parser.add_argument(
        "-e", "--extensions", nargs="+", help="File extensions to process (e.g., .exe .dll)"
    )
    parser.add_argument("-m", "--max-files", type=int, help="Maximum number of files to process")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Extract features
        file_info_list, features_list = extract_features_from_directory(
            input_dir=args.input_dir,
            output_file=args.output,
            file_extensions=args.extensions,
            max_files=args.max_files,
        )

        # Create feature matrix
        feature_matrix = create_feature_matrix(features_list)

        logger.info(f"Feature matrix shape: {feature_matrix.shape}")

        # Print summary
        successful = sum(1 for info in file_info_list if info["status"] == "success")
        failed = sum(1 for info in file_info_list if info["status"] == "failed")

        print("\nFeature extraction summary:")
        print(f"  Total files: {len(file_info_list)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Feature matrix shape: {feature_matrix.shape}")

        if args.output:
            print(f"  Results saved to: {args.output}")

    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
