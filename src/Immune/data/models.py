"""Data models for malware detection features."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class PEFeatures:
    """PE file header features."""

    # Basic PE characteristics
    size_of_code: int
    size_of_initialized_data: int
    size_of_uninitialized_data: int
    address_of_entry_point: int
    base_of_code: int
    base_of_data: int
    image_base: int
    section_alignment: int
    file_alignment: int

    # Version information
    major_os_version: int
    minor_os_version: int
    major_image_version: int
    minor_image_version: int
    major_subsystem_version: int
    minor_subsystem_version: int

    # Size information
    size_of_image: int
    size_of_headers: int
    checksum: int
    subsystem: int
    dll_characteristics: int

    # Memory configuration
    size_of_stack_reserve: int
    size_of_stack_commit: int
    size_of_heap_reserve: int
    size_of_heap_commit: int
    loader_flags: int
    number_of_rva_and_sizes: int

    # Section information - use all sections intelligently
    number_of_sections: int
    total_section_size: int
    max_section_size: int
    min_section_size: int
    avg_section_size: float
    executable_sections: int  # Number of executable sections
    writable_sections: int  # Number of writable sections
    suspicious_sections: int  # Sections that are both executable and writable

    # Import/Export information
    number_of_imports: int
    number_of_exports: int

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        features = [
            self.size_of_code,
            self.size_of_initialized_data,
            self.size_of_uninitialized_data,
            self.address_of_entry_point,
            self.base_of_code,
            self.base_of_data,
            self.image_base,
            self.section_alignment,
            self.file_alignment,
            self.major_os_version,
            self.minor_os_version,
            self.major_image_version,
            self.minor_image_version,
            self.major_subsystem_version,
            self.minor_subsystem_version,
            self.size_of_image,
            self.size_of_headers,
            self.checksum,
            self.subsystem,
            self.dll_characteristics,
            self.size_of_stack_reserve,
            self.size_of_stack_commit,
            self.size_of_heap_reserve,
            self.size_of_heap_commit,
            self.loader_flags,
            self.number_of_rva_and_sizes,
            self.number_of_sections,
            self.total_section_size,
            self.max_section_size,
            self.min_section_size,
            self.avg_section_size,
            self.executable_sections,
            self.writable_sections,
            self.suspicious_sections,
            self.number_of_imports,
            self.number_of_exports,
        ]

        return np.array(features, dtype=np.float32)

    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Get list of feature names in order."""
        return [
            "size_of_code",
            "size_of_initialized_data",
            "size_of_uninitialized_data",
            "address_of_entry_point",
            "base_of_code",
            "base_of_data",
            "image_base",
            "section_alignment",
            "file_alignment",
            "major_os_version",
            "minor_os_version",
            "major_image_version",
            "minor_image_version",
            "major_subsystem_version",
            "minor_subsystem_version",
            "size_of_image",
            "size_of_headers",
            "checksum",
            "subsystem",
            "dll_characteristics",
            "size_of_stack_reserve",
            "size_of_stack_commit",
            "size_of_heap_reserve",
            "size_of_heap_commit",
            "loader_flags",
            "number_of_rva_and_sizes",
            "number_of_sections",
            "total_section_size",
            "max_section_size",
            "min_section_size",
            "avg_section_size",
            "executable_sections",
            "writable_sections",
            "suspicious_sections",
            "number_of_imports",
            "number_of_exports",
        ]


@dataclass
class ByteHistogramFeatures:
    """Byte frequency histogram features."""

    # 256-dimensional byte histogram (normalized)
    histogram: np.ndarray  # Shape: (256,)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return self.histogram.astype(np.float32)

    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Get list of feature names in order."""
        return [f"byte_{i:02x}" for i in range(256)]


@dataclass
class APIFeatures:
    """API call and string-based features."""

    # Windows API function counts
    file_apis: Dict[str, int]  # CreateFile, ReadFile, WriteFile, etc.
    registry_apis: Dict[str, int]  # RegCreateKey, RegSetValue, etc.
    network_apis: Dict[str, int]  # connect, send, recv, etc.
    process_apis: Dict[str, int]  # CreateProcess, CreateThread, etc.
    memory_apis: Dict[str, int]  # VirtualAlloc, HeapAlloc, etc.
    system_apis: Dict[str, int]  # GetSystemTime, GetComputerName, etc.
    crypto_apis: Dict[str, int]  # CryptCreateHash, CryptEncrypt, etc.
    anti_debug_apis: Dict[str, int]  # IsDebuggerPresent, etc.

    # String statistics
    total_strings: int
    avg_string_length: float
    max_string_length: int
    min_string_length: int

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        # API counts (convert dict values to list in consistent order)
        api_features = []

        # File APIs
        file_api_names = [
            "CreateFile",
            "ReadFile",
            "WriteFile",
            "DeleteFile",
            "MoveFile",
            "CopyFile",
        ]
        api_features.extend([self.file_apis.get(name, 0) for name in file_api_names])

        # Registry APIs
        registry_api_names = [
            "RegCreateKey",
            "RegSetValue",
            "RegQueryValue",
            "RegDeleteKey",
            "RegOpenKey",
        ]
        api_features.extend([self.registry_apis.get(name, 0) for name in registry_api_names])

        # Network APIs
        network_api_names = ["connect", "send", "recv", "WSAConnect", "HttpOpenRequest"]
        api_features.extend([self.network_apis.get(name, 0) for name in network_api_names])

        # Process APIs
        process_api_names = [
            "CreateProcess",
            "CreateThread",
            "OpenProcess",
            "TerminateProcess",
            "LoadLibrary",
        ]
        api_features.extend([self.process_apis.get(name, 0) for name in process_api_names])

        # Memory APIs
        memory_api_names = ["VirtualAlloc", "VirtualFree", "HeapAlloc", "HeapFree", "LocalAlloc"]
        api_features.extend([self.memory_apis.get(name, 0) for name in memory_api_names])

        # System APIs
        system_api_names = [
            "GetSystemTime",
            "GetComputerName",
            "GetUserName",
            "GetWindowsDirectory",
        ]
        api_features.extend([self.system_apis.get(name, 0) for name in system_api_names])

        # Crypto APIs
        crypto_api_names = ["CryptCreateHash", "CryptHashData", "CryptEncrypt", "CryptDecrypt"]
        api_features.extend([self.crypto_apis.get(name, 0) for name in crypto_api_names])

        # Anti-debug APIs
        anti_debug_api_names = ["IsDebuggerPresent", "CheckRemoteDebuggerPresent", "GetTickCount"]
        api_features.extend([self.anti_debug_apis.get(name, 0) for name in anti_debug_api_names])

        # String statistics
        api_features.extend(
            [
                self.total_strings,
                self.avg_string_length,
                self.max_string_length,
                self.min_string_length,
            ]
        )

        return np.array(api_features, dtype=np.float32)

    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Get list of feature names in order."""
        file_api_names = [
            "CreateFile",
            "ReadFile",
            "WriteFile",
            "DeleteFile",
            "MoveFile",
            "CopyFile",
        ]
        registry_api_names = [
            "RegCreateKey",
            "RegSetValue",
            "RegQueryValue",
            "RegDeleteKey",
            "RegOpenKey",
        ]
        network_api_names = ["connect", "send", "recv", "WSAConnect", "HttpOpenRequest"]
        process_api_names = [
            "CreateProcess",
            "CreateThread",
            "OpenProcess",
            "TerminateProcess",
            "LoadLibrary",
        ]
        memory_api_names = ["VirtualAlloc", "VirtualFree", "HeapAlloc", "HeapFree", "LocalAlloc"]
        system_api_names = [
            "GetSystemTime",
            "GetComputerName",
            "GetUserName",
            "GetWindowsDirectory",
        ]
        crypto_api_names = ["CryptCreateHash", "CryptHashData", "CryptEncrypt", "CryptDecrypt"]
        anti_debug_api_names = ["IsDebuggerPresent", "CheckRemoteDebuggerPresent", "GetTickCount"]

        return (
            file_api_names
            + registry_api_names
            + network_api_names
            + process_api_names
            + memory_api_names
            + system_api_names
            + crypto_api_names
            + anti_debug_api_names
            + ["total_strings", "avg_string_length", "max_string_length", "min_string_length"]
        )


@dataclass
class MalwareFeatures:
    """Complete feature set for malware detection."""

    pe_features: PEFeatures
    byte_histogram: ByteHistogramFeatures
    api_features: APIFeatures

    def to_array(self) -> np.ndarray:
        """Convert all features to a single concatenated array."""
        return np.concatenate(
            [
                self.pe_features.to_array(),
                self.byte_histogram.to_array(),
                self.api_features.to_array(),
            ]
        )

    def get_feature_names(self) -> List[str]:
        """Get all feature names in order."""
        return (
            self.pe_features.get_feature_names()
            + self.byte_histogram.get_feature_names()
            + self.api_features.get_feature_names()
        )

    def get_feature_sizes(self) -> Dict[str, int]:
        """Get the size of each feature group."""
        return {
            "pe_features": len(self.pe_features.get_feature_names()),
            "byte_histogram": len(self.byte_histogram.get_feature_names()),
            "api_features": len(self.api_features.get_feature_names()),
        }

    def get_feature_ranges(self) -> Dict[str, tuple]:
        """Get the start and end indices for each feature group."""
        pe_size = len(self.pe_features.get_feature_names())
        byte_size = len(self.byte_histogram.get_feature_names())
        api_size = len(self.api_features.get_feature_names())

        return {
            "pe_features": (0, pe_size),
            "byte_histogram": (pe_size, pe_size + byte_size),
            "api_features": (pe_size + byte_size, pe_size + byte_size + api_size),
        }
