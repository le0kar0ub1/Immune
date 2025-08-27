"""Simple feature extractor for malware detection."""

import logging
import re
from pathlib import Path

import numpy as np
import pefile

from .models import APIFeatures, BinaryFeatures, ByteHistogramFeatures, EntropyFeatures, PEFeatures

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract features from binary files for malware detection."""

    def __init__(self):
        """Initialize the feature extractor."""
        pass

    def __del__(self):
        """Close the PE file."""
        if hasattr(self, "pe"):
            self.pe.close()

    def extract_features(self, file_path: Path, is_malware: bool) -> BinaryFeatures:
        """Extract all features from a binary file.

        Args:
            file_path: Path to the binary file

        Returns:
            BinaryFeatures object containing all extracted features
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Extracting features from {file_path}")

        with open(file_path, "rb") as f:
            self.bdata = f.read()

        # Try to parse as PE file, but don't fail if it's not a PE
        self.pe = None
        try:
            self.pe = pefile.PE(file_path)
            logger.debug(f"Successfully parsed {file_path} as PE file")
        except Exception as e:
            logger.warning(f"Failed to parse {file_path} as PE file: {e}")
            self.pe = None
            raise e

        # Extract each feature type
        logger.debug(f"Extracting PE features from {file_path}")
        pe_features = self._extract_pe_features(file_path)
        logger.debug(f"Extracting byte histogram from {file_path}")
        byte_histogram = self._extract_byte_histogram(file_path)
        logger.debug(f"Extracting entropy features from {file_path}")
        entropy_features = self._extract_entropy_features(file_path)
        logger.debug(f"Extracting API features from {file_path}")
        api_features = self._extract_api_features(file_path)

        return BinaryFeatures(
            is_malware=is_malware,
            file_size=file_path.stat().st_size,
            pe_features=pe_features,
            byte_histogram=byte_histogram,
            entropy_features=entropy_features,
            api_features=api_features,
        )

    def _extract_pe_features(self, file_path: Path) -> PEFeatures:
        """Extract PE file features."""
        # Check if we have a valid PE file
        if not hasattr(self, "pe") or self.pe is None:
            logger.warning(f"No PE file available for {file_path}, returning default features")
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

        try:
            # Analyze all sections intelligently
            section_sizes = []
            executable_sections = 0
            writable_sections = 0
            suspicious_sections = 0

            for section in self.pe.sections:
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
                size_of_code=getattr(self.pe.OPTIONAL_HEADER, "SizeOfCode", 0),
                size_of_initialized_data=getattr(
                    self.pe.OPTIONAL_HEADER, "SizeOfInitializedData", 0
                ),
                size_of_uninitialized_data=getattr(
                    self.pe.OPTIONAL_HEADER, "SizeOfUninitializedData", 0
                ),
                address_of_entry_point=getattr(self.pe.OPTIONAL_HEADER, "AddressOfEntryPoint", 0),
                base_of_code=getattr(self.pe.OPTIONAL_HEADER, "BaseOfData", 0),
                base_of_data=getattr(self.pe.OPTIONAL_HEADER, "BaseOfData", 0),
                image_base=getattr(self.pe.OPTIONAL_HEADER, "ImageBase", 0),
                section_alignment=getattr(self.pe.OPTIONAL_HEADER, "SectionAlignment", 0),
                file_alignment=getattr(self.pe.OPTIONAL_HEADER, "FileAlignment", 0),
                # Version information
                major_os_version=getattr(self.pe.OPTIONAL_HEADER, "MajorOperatingSystemVersion", 0),
                minor_os_version=getattr(self.pe.OPTIONAL_HEADER, "MinorOperatingSystemVersion", 0),
                major_image_version=getattr(self.pe.OPTIONAL_HEADER, "MajorImageVersion", 0),
                minor_image_version=getattr(self.pe.OPTIONAL_HEADER, "MinorImageVersion", 0),
                major_subsystem_version=getattr(
                    self.pe.OPTIONAL_HEADER, "MajorSubsystemVersion", 0
                ),
                minor_subsystem_version=getattr(
                    self.pe.OPTIONAL_HEADER, "MinorSubsystemVersion", 0
                ),
                # Size information
                size_of_image=getattr(self.pe.OPTIONAL_HEADER, "SizeOfImage", 0),
                size_of_headers=getattr(self.pe.OPTIONAL_HEADER, "SizeOfHeaders", 0),
                checksum=getattr(self.pe.OPTIONAL_HEADER, "CheckSum", 0),
                subsystem=getattr(self.pe.OPTIONAL_HEADER, "Subsystem", 0),
                dll_characteristics=getattr(self.pe.OPTIONAL_HEADER, "DllCharacteristics", 0),
                # Memory configuration
                size_of_stack_reserve=getattr(self.pe.OPTIONAL_HEADER, "SizeOfStackReserve", 0),
                size_of_stack_commit=getattr(self.pe.OPTIONAL_HEADER, "SizeOfStackCommit", 0),
                size_of_heap_reserve=getattr(self.pe.OPTIONAL_HEADER, "SizeOfHeapReserve", 0),
                size_of_heap_commit=getattr(self.pe.OPTIONAL_HEADER, "SizeOfHeapCommit", 0),
                loader_flags=getattr(self.pe.OPTIONAL_HEADER, "LoaderFlags", 0),
                number_of_rva_and_sizes=getattr(self.pe.OPTIONAL_HEADER, "NumberOfRvaAndSizes", 0),
                # Intelligent section analysis
                number_of_sections=len(self.pe.sections),
                total_section_size=total_section_size,
                max_section_size=max_section_size,
                min_section_size=min_section_size,
                avg_section_size=avg_section_size,
                executable_sections=executable_sections,
                writable_sections=writable_sections,
                suspicious_sections=suspicious_sections,
                # Import/Export information
                number_of_imports=sum(
                    len(entry.imports) for entry in self.pe.DIRECTORY_ENTRY_IMPORT
                )
                if hasattr(self.pe, "DIRECTORY_ENTRY_IMPORT")
                else 0,
                number_of_exports=len(self.pe.DIRECTORY_ENTRY_EXPORT.symbols)
                if hasattr(self.pe, "DIRECTORY_ENTRY_EXPORT")
                else 0,
            )

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
            data = np.frombuffer(self.bdata, dtype=np.uint8)
            histogram = np.bincount(data, minlength=256)
            histogram = histogram.astype(np.float32) / len(self.bdata)
            return ByteHistogramFeatures(histogram=histogram)
        except Exception as e:
            logger.warning(f"Failed to extract byte histogram from {file_path}: {e}")
            return ByteHistogramFeatures(histogram=np.zeros(256, dtype=np.float32))

    def _calculate_shanon_entropy(self, data: bytes) -> float:
        if not data:
            return 0.0
        arr = np.frombuffer(data, dtype=np.uint8)
        counts = np.bincount(arr, minlength=256)  # histogram of all 256 possible byte values
        probs = counts / arr.size  # normalize to probabilities
        probs = probs[probs > 0]  # drop zero entries
        entropy = -np.sum(probs * np.log2(probs))  # Shannon entropy in bits
        return float(entropy)

    def _extract_entropy_features(self, file_path: Path) -> EntropyFeatures:
        """Extract entropy features from binary file, handling both PE and non-PE files."""
        try:
            overall_entropy = self._calculate_shanon_entropy(self.bdata)

            section_entropies = {}
            data_section_entropy = 0.0
            rodata_section_entropy = 0.0
            text_section_entropy = 0.0

            # Try to extract PE section entropies if available
            if hasattr(self, "pe") and self.pe is not None:
                try:
                    for section in self.pe.sections:
                        if section.SizeOfRawData > 0:
                            section_data = section.get_data()
                            section_entropy = self._calculate_shanon_entropy(section_data)
                            logger.debug(f"Section {section.Name}: entropy = {section_entropy}")

                            # Store section entropy
                            section_entropies[section.Name] = section_entropy

                            # Identify specific section types
                            if re.match(r"\.data.*", section.Name.decode("utf-8", errors="ignore")):
                                data_section_entropy = section_entropy
                            elif re.match(
                                r"\.r(o)?data.*", section.Name.decode("utf-8", errors="ignore")
                            ):
                                rodata_section_entropy = section_entropy
                            elif re.match(
                                r"\.text.*", section.Name.decode("utf-8", errors="ignore")
                            ):
                                text_section_entropy = section_entropy
                except Exception as e:
                    logger.warning(f"Failed to parse PE sections for entropy: {e}")
                    # Fall through to chunk-based entropy

            # If no PE sections or parsing failed, use chunk-based entropy
            if not section_entropies:
                chunk_size = 4096  # 4 KB chunks
                for i in range(0, len(self.bdata), chunk_size):
                    chunk = self.bdata[i : i + chunk_size]
                    if len(chunk) >= 256:  # Only calculate entropy for chunks with sufficient data
                        chunk_entropy = self._calculate_shanon_entropy(chunk)
                        section_entropies[f"chunk_{i // chunk_size}"] = chunk_entropy

            # Calculate section-level entropy statistics
            if section_entropies:
                min_section_entropy = min(section_entropies.values())
                max_section_entropy = max(section_entropies.values())
                avg_section_entropy = sum(section_entropies.values()) / len(section_entropies)
                std_section_entropy = np.std(list(section_entropies.values()))
                high_entropy_sections = sum(e > 7.0 for e in section_entropies.values())
                low_entropy_sections = sum(e < 4.0 for e in section_entropies.values())
                entropy_variance = np.var(list(section_entropies.values()))
            else:
                # Fallback values if no sections/chunks available
                min_section_entropy = max_section_entropy = avg_section_entropy = overall_entropy
                std_section_entropy = high_entropy_sections = low_entropy_sections = (
                    entropy_variance
                ) = 0.0

            return EntropyFeatures(
                overall_entropy=overall_entropy,
                data_section_entropy=data_section_entropy,
                rodata_section_entropy=rodata_section_entropy,
                text_section_entropy=text_section_entropy,
                min_section_entropy=min_section_entropy,
                max_section_entropy=max_section_entropy,
                avg_section_entropy=avg_section_entropy,
                std_section_entropy=std_section_entropy,
                high_entropy_sections=high_entropy_sections,
                low_entropy_sections=low_entropy_sections,
                entropy_variance=entropy_variance,
            )

        except Exception as e:
            logger.warning(f"Failed to extract entropy features from {file_path}: {e}")
            return EntropyFeatures(
                overall_entropy=0.0,
                data_section_entropy=0.0,
                rodata_section_entropy=0.0,
                text_section_entropy=0.0,
                min_section_entropy=0.0,
                max_section_entropy=0.0,
                avg_section_entropy=0.0,
                std_section_entropy=0.0,
                high_entropy_sections=0,
                low_entropy_sections=0,
                entropy_variance=0.0,
            )

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
