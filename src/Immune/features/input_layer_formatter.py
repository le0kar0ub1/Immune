"""Input layer formatter for transforming BinaryFeatures to model-ready arrays."""

import argparse
import json
from typing import Any, Dict

from models import BinaryFeatures


class InputLayerFormatter:
    """Formats BinaryFeatures for AI model input."""

    def features_from_file(self, file_path: str) -> Dict[str, BinaryFeatures]:
        """
        Create a BinaryFeatures object from a file.
        """
        with open(file_path, "r") as f:
            features = json.load(f)
            for filehash, feature_dict in features.items():
                features[filehash] = BinaryFeatures.from_dict(feature_dict)
            return features

    @staticmethod
    def format_features_dict(features_dict: Dict[str, BinaryFeatures]) -> Dict[str, Dict[str, Any]]:
        """
        Transform a dictionary of {filehash: BinaryFeatures} to
        {filehash: {is_malware: bool, feature_array: [...]} } format.

        Args:
            features_dict: Dictionary mapping file hashes to BinaryFeatures

        Returns:
            Dictionary with file hash and feature array
        """
        formatted_data = {}
        feature_len = None

        for filehash, binary_features in features_dict.items():
            feature_array = binary_features.to_input_layer_format()
            if feature_len is None:
                feature_len = len(feature_array)
            elif feature_len != len(feature_array):
                raise ValueError(f"Feature length mismatch for file {filehash}")

            formatted_data[filehash] = {
                "is_malware": binary_features.is_malware,
                "feature_array": feature_array.tolist(),  # Convert numpy array to list for JSON serialization
            }

        return formatted_data

    @staticmethod
    def validate_features_dict(features_dict: Dict[str, BinaryFeatures]) -> bool:
        """
        Validate that all BinaryFeatures objects in the dictionary are properly formatted.

        Args:
            features_dict: Dictionary to validate

        Returns:
            True if valid, False otherwise
        """
        if not features_dict:
            return False

        expected_features = InputLayerFormatter.get_total_features()

        for _filehash, binary_features in features_dict.items():
            if not isinstance(binary_features, BinaryFeatures):
                return False

            try:
                feature_array = binary_features.to_input_layer_format()
                if len(feature_array) != expected_features:
                    return False
            except Exception:
                return False

        return True


obj = InputLayerFormatter()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format features for input layer")
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Path to the input features file"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Path to the output formatted features file"
    )
    args = parser.parse_args()

    features = obj.features_from_file(args.input)
    formatted_features = obj.format_features_dict(features)
    with open(args.output, "w") as f:
        json.dump(formatted_features, f)
