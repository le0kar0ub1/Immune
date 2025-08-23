"""Simple feature extractor for malware detection."""

import logging
from pathlib import Path

import numpy as np
import pefile

from .model import APIFeatures, ByteHistogramFeatures, MalwareFeatures, PEFeatures

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract features from binary files for malware detection."""

    def __init__(self):
        """Initialize the feature extractor."""
        pass

    def extract_features(self, file_path: Path) -> MalwareFeatures:
        """Extract all features from a binary file.

        Args:
            file_path: Path to the binary file

        Returns:
            MalwareFeatures object containing all extracted features
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Extracting features from {file_path}")

        with open(file_path, "rb") as f:
            self.bdata = f.read()
        # Extract each feature type
        pe_features = self._extract_pe_features(file_path)
        byte_histogram = self._extract_byte_histogram(file_path)
        api_features = self._extract_api_features(file_path)

        return MalwareFeatures(
            pe_features=pe_features, byte_histogram=byte_histogram, api_features=api_features
        )

    def _extract_pe_features(self, file_path: Path) -> PEFeatures:
        """Extract PE file features."""
        try:
            pe = pefile.PE(file_path)

            # Analyze all sections intelligently
            section_sizes = []
            executable_sections = 0
            writable_sections = 0
            suspicious_sections = 0

            for section in pe.sections:
                size = section.SizeOfRawData
                section_sizes.append(size)

                # Check section characteristics
                characteristics = section.Characteristics
                is_executable = bool(characteristics & 0x20000000)  # IMAGE_SCN_MEM_EXECUTE
                is_writable = bool(characteristics & 0x80000000)  # IMAGE_SCN_MEM_WRITE

                if is_executable:
                    executable_sections += 1
                if is_writable:
                    writable_sections += 1
                if is_executable and is_writable:
                    suspicious_sections += 1  # Both executable and writable is suspicious

            # Calculate section statistics
            total_section_size = sum(section_sizes) if section_sizes else 0
            max_section_size = max(section_sizes) if section_sizes else 0
            min_section_size = min(section_sizes) if section_sizes else 0
            avg_section_size = float(np.mean(section_sizes)) if section_sizes else 0.0

            pe_features = PEFeatures(
                # Basic PE characteristics
                size_of_code=pe.OPTIONAL_HEADER.SizeOfCode,
                size_of_initialized_data=pe.OPTIONAL_HEADER.SizeOfInitializedData,
                size_of_uninitialized_data=pe.OPTIONAL_HEADER.SizeOfUninitializedData,
                address_of_entry_point=pe.OPTIONAL_HEADER.AddressOfEntryPoint,
                base_of_code=pe.OPTIONAL_HEADER.BaseOfCode,
                base_of_data=pe.OPTIONAL_HEADER.BaseOfData,
                image_base=pe.OPTIONAL_HEADER.ImageBase,
                section_alignment=pe.OPTIONAL_HEADER.SectionAlignment,
                file_alignment=pe.OPTIONAL_HEADER.FileAlignment,
                # Version information
                major_os_version=pe.OPTIONAL_HEADER.MajorOperatingSystemVersion,
                minor_os_version=pe.OPTIONAL_HEADER.MinorOperatingSystemVersion,
                major_image_version=pe.OPTIONAL_HEADER.MajorImageVersion,
                minor_image_version=pe.OPTIONAL_HEADER.MinorImageVersion,
                major_subsystem_version=pe.OPTIONAL_HEADER.MajorSubsystemVersion,
                minor_subsystem_version=pe.OPTIONAL_HEADER.MinorSubsystemVersion,
                # Size information
                size_of_image=pe.OPTIONAL_HEADER.SizeOfImage,
                size_of_headers=pe.OPTIONAL_HEADER.SizeOfHeaders,
                checksum=pe.OPTIONAL_HEADER.CheckSum,
                subsystem=pe.OPTIONAL_HEADER.Subsystem,
                dll_characteristics=pe.OPTIONAL_HEADER.DllCharacteristics,
                # Memory configuration
                size_of_stack_reserve=pe.OPTIONAL_HEADER.SizeOfStackReserve,
                size_of_stack_commit=pe.OPTIONAL_HEADER.SizeOfStackCommit,
                size_of_heap_reserve=pe.OPTIONAL_HEADER.SizeOfHeapReserve,
                size_of_heap_commit=pe.OPTIONAL_HEADER.SizeOfHeapCommit,
                loader_flags=pe.OPTIONAL_HEADER.LoaderFlags,
                number_of_rva_and_sizes=pe.OPTIONAL_HEADER.NumberOfRvaAndSizes,
                # Intelligent section analysis
                number_of_sections=len(pe.sections),
                total_section_size=total_section_size,
                max_section_size=max_section_size,
                min_section_size=min_section_size,
                avg_section_size=avg_section_size,
                executable_sections=executable_sections,
                writable_sections=writable_sections,
                suspicious_sections=suspicious_sections,
                # Import/Export information
                number_of_imports=len(pe.DIRECTORY_ENTRY_IMPORT)
                if hasattr(pe, "DIRECTORY_ENTRY_IMPORT")
                else 0,
                number_of_exports=len(pe.DIRECTORY_ENTRY_EXPORT)
                if hasattr(pe, "DIRECTORY_ENTRY_EXPORT")
                else 0,
            )

            pe.close()
            return pe_features

        except Exception as e:
            logger.warning(f"Failed to extract PE features from {file_path}: {e}")
            # Return default PE features with zeros
            return PEFeatures(
                size_of_code=0,
                size_of_initialized_data=0,
                size_of_uninitialized_data=0,
                address_of_entry_point=0,
                base_of_code=0,
                base_of_data=0,
                image_base=0,
                section_alignment=0,
                file_alignment=0,
                major_os_version=0,
                minor_os_version=0,
                major_image_version=0,
                minor_image_version=0,
                major_subsystem_version=0,
                minor_subsystem_version=0,
                size_of_image=0,
                size_of_headers=0,
                checksum=0,
                subsystem=0,
                dll_characteristics=0,
                size_of_stack_reserve=0,
                size_of_stack_commit=0,
                size_of_heap_reserve=0,
                size_of_heap_commit=0,
                loader_flags=0,
                number_of_rva_and_sizes=0,
                number_of_sections=0,
                total_section_size=0,
                max_section_size=0,
                min_section_size=0,
                avg_section_size=0.0,
                executable_sections=0,
                writable_sections=0,
                suspicious_sections=0,
                number_of_imports=0,
                number_of_exports=0,
            )

    def _extract_byte_histogram(self, file_path: Path) -> ByteHistogramFeatures:
        """Extract byte histogram features."""
        try:
            # Calculate byte frequency histogram
            histogram = np.zeros(256, dtype=np.uint32)
            for byte in self.bdata:
                histogram[byte] += 1

            # Normalize histogram
            if len(self.bdata) > 0:
                histogram = histogram.astype(np.float32) / len(self.bdata)

            return ByteHistogramFeatures(histogram=histogram)

        except Exception as e:
            logger.warning(f"Failed to extract byte histogram from {file_path}: {e}")
            return ByteHistogramFeatures(histogram=np.zeros(256, dtype=np.float32))

    def _extract_api_features(self, file_path: Path) -> APIFeatures:
        """Extract API call and string features."""
        try:
            # Convert data to string for API search
            data_str = self.bdata.decode("utf-8", errors="ignore")

            # Define API categories
            file_apis = [
                "CreateFile",
                "ReadFile",
                "WriteFile",
                "DeleteFile",
                "MoveFile",
                "CopyFile",
            ]
            registry_apis = [
                "RegCreateKey",
                "RegSetValue",
                "RegQueryValue",
                "RegDeleteKey",
                "RegOpenKey",
            ]
            network_apis = ["connect", "send", "recv", "WSAConnect", "HttpOpenRequest"]
            process_apis = [
                "CreateProcess",
                "CreateThread",
                "OpenProcess",
                "TerminateProcess",
                "LoadLibrary",
            ]
            memory_apis = ["VirtualAlloc", "VirtualFree", "HeapAlloc", "HeapFree", "LocalAlloc"]
            system_apis = ["GetSystemTime", "GetComputerName", "GetUserName", "GetWindowsDirectory"]
            crypto_apis = ["CryptCreateHash", "CryptHashData", "CryptEncrypt", "CryptDecrypt"]
            anti_debug_apis = ["IsDebuggerPresent", "CheckRemoteDebuggerPresent", "GetTickCount"]

            # Count API calls
            api_counts = {
                "file_apis": {api: data_str.count(api) for api in file_apis},
                "registry_apis": {api: data_str.count(api) for api in registry_apis},
                "network_apis": {api: data_str.count(api) for api in network_apis},
                "process_apis": {api: data_str.count(api) for api in process_apis},
                "memory_apis": {api: data_str.count(api) for api in memory_apis},
                "system_apis": {api: data_str.count(api) for api in system_apis},
                "crypto_apis": {api: data_str.count(api) for api in crypto_apis},
                "anti_debug_apis": {api: data_str.count(api) for api in anti_debug_apis},
            }

            # Extract strings and calculate statistics
            strings = self._extract_strings(self.bdata)
            string_lengths = [len(s) for s in strings] if strings else [0]

            return APIFeatures(
                file_apis=api_counts["file_apis"],
                registry_apis=api_counts["registry_apis"],
                network_apis=api_counts["network_apis"],
                process_apis=api_counts["process_apis"],
                memory_apis=api_counts["memory_apis"],
                system_apis=api_counts["system_apis"],
                crypto_apis=api_counts["crypto_apis"],
                anti_debug_apis=api_counts["anti_debug_apis"],
                total_strings=len(strings),
                avg_string_length=float(np.mean(string_lengths)),
                max_string_length=max(string_lengths),
                min_string_length=min(string_lengths),
            )

        except Exception as e:
            logger.warning(f"Failed to extract API features from {file_path}: {e}")
            # Return default API features
            return APIFeatures(
                file_apis={},
                registry_apis={},
                network_apis={},
                process_apis={},
                memory_apis={},
                system_apis={},
                crypto_apis={},
                anti_debug_apis={},
                total_strings=0,
                avg_string_length=0.0,
                max_string_length=0,
                min_string_length=0,
            )

    def _extract_strings(self, data: bytes) -> list:
        """Extract printable strings from binary data."""
        strings = []
        current_string = ""

        for byte in data:
            if 32 <= byte <= 126:  # Printable ASCII
                current_string += chr(byte)
            else:
                if len(current_string) >= 3:  # Only count strings of length 3 or more
                    strings.append(current_string)
                current_string = ""

        if current_string and len(current_string) >= 3:
            strings.append(current_string)

        return strings
