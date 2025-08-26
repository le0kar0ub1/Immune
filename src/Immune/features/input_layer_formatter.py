"""Input layer formatter for transforming BinaryFeatures to model-ready arrays."""

from typing import Dict, List, Any
import numpy as np

from .models import BinaryFeatures


class InputLayerFormatter:
    """Formats BinaryFeatures for AI model input."""
    
    @staticmethod
    def format_features_dict(features_dict: Dict[str, BinaryFeatures]) -> List[Dict[str, Any]]:
        """
        Transform a dictionary of {filehash: BinaryFeatures} to 
        [{file: filehash, feature_array: [...]}] format.
        
        Args:
            features_dict: Dictionary mapping file hashes to BinaryFeatures
            
        Returns:
            List of dictionaries with file hash and feature array
        """
        formatted_data = []
        
        for filehash, binary_features in features_dict.items():
            feature_array = binary_features.to_array()
            
            formatted_data.append({
                "file": filehash,
                "feature_array": feature_array.tolist()  # Convert numpy array to list for JSON serialization
            })
            
        return formatted_data
    
    @staticmethod
    def format_features_dict_numpy(features_dict: Dict[str, BinaryFeatures]) -> List[Dict[str, Any]]:
        """
        Transform a dictionary of {filehash: BinaryFeatures} to 
        [{file: filehash, feature_array: numpy_array}] format (keeps numpy arrays).
        
        Args:
            features_dict: Dictionary mapping file hashes to BinaryFeatures
            
        Returns:
            List of dictionaries with file hash and numpy feature array
        """
        formatted_data = []
        
        for filehash, binary_features in features_dict.items():
            feature_array = binary_features.to_array()
            
            formatted_data.append({
                "file": filehash,
                "feature_array": feature_array
            })
            
        return formatted_data
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """
        Get the complete list of feature names in the order they appear in the feature array.
        
        Returns:
            List of feature names
        """
        # Create a dummy BinaryFeatures object to get feature names
        # We'll use minimal data since we only need the structure
        dummy_pe = BinaryFeatures(
            file_size=0,
            pe_features=None,
            byte_histogram=None,
            entropy_features=None,
            api_features=None
        )
        
        # This will raise an error since we need actual feature objects
        # Instead, let's create a proper method that doesn't require instantiation
        return ["file_size"] + BinaryFeatures._get_all_feature_names()
    
    @staticmethod
    def get_feature_dimensions() -> Dict[str, int]:
        """
        Get the dimensions of each feature group.
        
        Returns:
            Dictionary mapping feature group names to their dimensions
        """
        return {
            "file_size": 1,
            "pe_features": 42,  # From PEFeatures.get_feature_names()
            "byte_histogram": 256,  # From ByteHistogramFeatures.get_feature_names()
            "entropy_features": 11,  # From EntropyFeatures.get_feature_names()
            "api_features": 47,  # From APIFeatures.get_feature_names()
        }
    
    @staticmethod
    def get_total_features() -> int:
        """
        Get the total number of features.
        
        Returns:
            Total number of features
        """
        dimensions = InputLayerFormatter.get_feature_dimensions()
        return sum(dimensions.values())
    
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
        
        for filehash, binary_features in features_dict.items():
            if not isinstance(binary_features, BinaryFeatures):
                return False
                
            try:
                feature_array = binary_features.to_array()
                if len(feature_array) != expected_features:
                    return False
            except Exception:
                return False
                
        return True


# Add a helper method to BinaryFeatures class for getting all feature names
def _get_all_feature_names(self) -> List[str]:
    """Get all feature names in order without requiring instantiation."""
    pe_names = [
        "size_of_code", "size_of_initialized_data", "size_of_uninitialized_data",
        "address_of_entry_point", "base_of_code", "base_of_data", "image_base",
        "section_alignment", "file_alignment", "major_os_version", "minor_os_version",
        "major_image_version", "minor_image_version", "major_subsystem_version",
        "minor_subsystem_version", "size_of_image", "size_of_headers", "checksum",
        "subsystem", "dll_characteristics", "size_of_stack_reserve", "size_of_stack_commit",
        "size_of_heap_reserve", "size_of_heap_commit", "loader_flags", "number_of_rva_and_sizes",
        "number_of_sections", "total_section_size", "max_section_size", "min_section_size",
        "avg_section_size", "executable_sections", "writable_sections", "suspicious_sections",
        "number_of_imports", "number_of_exports"
    ]
    
    byte_names = [f"byte_{i:02x}" for i in range(256)]
    
    entropy_names = [
        "overall_entropy", "data_section_entropy", "rodata_section_entropy", "text_section_entropy",
        "min_section_entropy", "max_section_entropy", "avg_section_entropy", "std_section_entropy",
        "high_entropy_sections", "low_entropy_sections", "entropy_variance"
    ]
    
    api_names = [
        "CreateFile", "ReadFile", "WriteFile", "DeleteFile", "MoveFile", "CopyFile",
        "RegCreateKey", "RegSetValue", "RegQueryValue", "RegDeleteKey", "RegOpenKey",
        "connect", "send", "recv", "WSAConnect", "HttpOpenRequest",
        "CreateProcess", "CreateThread", "OpenProcess", "TerminateProcess", "LoadLibrary",
        "VirtualAlloc", "VirtualFree", "HeapAlloc", "HeapFree", "LocalAlloc",
        "GetSystemTime", "GetComputerName", "GetUserName", "GetWindowsDirectory",
        "CryptCreateHash", "CryptHashData", "CryptEncrypt", "CryptDecrypt",
        "IsDebuggerPresent", "CheckRemoteDebuggerPresent", "GetTickCount",
        "total_strings", "avg_string_length", "max_string_length", "min_string_length"
    ]
    
    return pe_names + byte_names + entropy_names + api_names


# Monkey patch the BinaryFeatures class to add the helper method
BinaryFeatures._get_all_feature_names = classmethod(_get_all_feature_names)
