"""Data models for EMBER dataset malware detection features."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

input_dir = Path("data/ember/")


@dataclass
class EmberBinaryFeature:
    """Complete EMBER dataset feature set for malware detection in a single class."""

    # Basic identifiers
    sha256: str
    md5: str
    appeared: str
    label: int
    avclass: str

    # Byte histogram and entropy (256-dimensional each)
    histogram: np.ndarray  # Shape: (256,)
    byteentropy: np.ndarray  # Shape: (256,)

    # String features
    numstrings: int
    avlength: float
    printabledist: List[int]
    printables: int
    string_entropy: float
    paths: int
    urls: int
    registry: int
    mz_count: int

    # General PE features
    size: int
    vsize: int
    has_debug: int
    exports: int
    imports: int
    has_relocations: int
    has_resources: int
    has_signature: int
    has_tls: int
    symbols: int

    # PE header features
    timestamp: int
    machine: str
    characteristics: List[str]
    subsystem: str
    dll_characteristics: List[str]
    magic: str
    major_image_version: int
    minor_image_version: int
    major_linker_version: int
    minor_linker_version: int
    major_operating_system_version: int
    minor_operating_system_version: int
    major_subsystem_version: int
    minor_subsystem_version: int
    sizeof_code: int
    sizeof_headers: int
    sizeof_heap_commit: int

    # Section features
    section_entry: str
    sections: List[Dict[str, Any]]

    # Import features
    import_data: Dict[str, List[str]]

    # Export and data directory features
    exports: List[str]
    datadirectories: List[Dict[str, Any]]

    def __init__(
        self,
        sha256: str,
        md5: str,
        appeared: str,
        label: int,
        avclass: str,
        histogram: List[int],
        byteentropy: List[int],
        strings: Dict[str, Any],
        general: Dict[str, Any],
        header: Dict[str, Any],
        section: Dict[str, Any],
        imports: Dict[str, Any],
        exports: List[str],
        datadirectories: List[Dict[str, Any]],
    ):
        self.sha256 = sha256
        self.md5 = md5
        self.appeared = appeared
        self.label = label
        self.avclass = avclass

        # Convert histogram and byteentropy to numpy arrays
        self.histogram = np.array(histogram, dtype=np.float32)
        self.byteentropy = np.array(byteentropy, dtype=np.float32)

        # Extract string features
        self.numstrings = strings.get("numstrings", 0)
        self.avlength = strings.get("avlength", 0.0)
        self.printabledist = strings.get("printabledist", [])
        self.printables = strings.get("printables", 0)
        self.string_entropy = strings.get("entropy", 0.0)
        self.paths = strings.get("paths", 0)
        self.urls = strings.get("urls", 0)
        self.registry = strings.get("registry", 0)
        self.mz_count = strings.get("MZ", 0)

        # Extract general features
        self.size = general.get("size", 0)
        self.vsize = general.get("vsize", 0)
        self.has_debug = general.get("has_debug", 0)
        self.exports = general.get("exports", 0)
        self.imports = general.get("imports", 0)
        self.has_relocations = general.get("has_relocations", 0)
        self.has_resources = general.get("has_resources", 0)
        self.has_signature = general.get("has_signature", 0)
        self.has_tls = general.get("has_tls", 0)
        self.symbols = general.get("symbols", 0)

        # Extract header features
        self.timestamp = header.get("coff", {}).get("timestamp", 0)
        self.machine = header.get("coff", {}).get("machine", "")
        self.characteristics = header.get("coff", {}).get("characteristics", [])
        self.subsystem = header.get("optional", {}).get("subsystem", "")
        self.dll_characteristics = header.get("optional", {}).get("dll_characteristics", [])
        self.magic = header.get("optional", {}).get("magic", "")
        self.major_image_version = header.get("optional", {}).get("major_image_version", 0)
        self.minor_image_version = header.get("optional", {}).get("minor_image_version", 0)
        self.major_linker_version = header.get("optional", {}).get("major_linker_version", 0)
        self.minor_linker_version = header.get("optional", {}).get("minor_linker_version", 0)
        self.major_operating_system_version = header.get("optional", {}).get(
            "major_operating_system_version", 0
        )
        self.minor_operating_system_version = header.get("optional", {}).get(
            "minor_operating_system_version", 0
        )
        self.major_subsystem_version = header.get("optional", {}).get("major_subsystem_version", 0)
        self.minor_subsystem_version = header.get("optional", {}).get("minor_subsystem_version", 0)
        self.sizeof_code = header.get("optional", {}).get("sizeof_code", 0)
        self.sizeof_headers = header.get("optional", {}).get("sizeof_headers", 0)
        self.sizeof_heap_commit = header.get("optional", {}).get("sizeof_heap_commit", 0)

        # Extract section features
        self.section_entry = section.get("entry", "")
        self.sections = section.get("sections", [])

        # Extract import features
        self.import_data = imports

        # Extract export and data directory features
        self.exports = len(exports) if exports else 0
        self.datadirectories = len(datadirectories) if datadirectories else 0

    def to_input_layer_format(self) -> np.ndarray:
        """Convert all features to a single concatenated numpy array."""
        # Normalize histogram to probabilities
        hist_sum = np.sum(self.histogram)
        normalized_hist = self.histogram / hist_sum if hist_sum > 0 else self.histogram

        # Normalize byteentropy
        byteent_sum = np.sum(self.byteentropy)
        normalized_byteent = self.byteentropy / byteent_sum if byteent_sum > 0 else self.byteentropy

        # String features - convert all to numbers
        print_dist = self.printabledist[:100]  # Use first 100 values
        while len(print_dist) < 100:
            print_dist.append(0)

        string_features = [
            self.numstrings,
            self.avlength,
            self.printables,
            self.string_entropy,
            self.paths,
            self.urls,
            self.registry,
            self.mz_count,
        ] + print_dist

        # General features
        general_features = [
            self.size,
            self.vsize,
            self.has_debug,
            self.exports,
            self.imports,
            self.has_relocations,
            self.has_resources,
            self.has_signature,
            self.has_tls,
            self.symbols,
        ]

        # Header features - convert all to numbers
        char_features = [
            1 if "CHARA_32BIT_MACHINE" in self.characteristics else 0,
            1 if "RELOCS_STRIPPED" in self.characteristics else 0,
            1 if "EXECUTABLE_IMAGE" in self.characteristics else 0,
            1 if "LINE_NUMS_STRIPPED" in self.characteristics else 0,
            1 if "LOCAL_SYMS_STRIPPED" in self.characteristics else 0,
        ]

        dll_char_features = [
            1 if "DYNAMIC_BASE" in self.dll_characteristics else 0,
            1 if "FORCE_INTEGRITY" in self.dll_characteristics else 0,
            1 if "NX_COMPAT" in self.dll_characteristics else 0,
            1 if "NO_ISOLATION" in self.dll_characteristics else 0,
            1 if "NO_SEH" in self.dll_characteristics else 0,
        ]

        # Add array length features
        char_count = len(self.characteristics) if self.characteristics else 0
        dll_char_count = len(self.dll_characteristics) if self.dll_characteristics else 0

        # Convert string fields to numerical representations
        machine_numeric = hash(self.machine) % 10000 if self.machine else 0
        subsystem_numeric = hash(self.subsystem) % 10000 if self.subsystem else 0
        magic_numeric = hash(self.magic) % 10000 if self.magic else 0

        header_features = (
            [
                self.timestamp,
                machine_numeric,
                subsystem_numeric,
                magic_numeric,
                self.major_image_version,
                self.minor_image_version,
                self.major_linker_version,
                self.minor_linker_version,
                self.major_operating_system_version,
                self.minor_operating_system_version,
                self.major_subsystem_version,
                self.minor_subsystem_version,
                self.sizeof_code,
                self.sizeof_headers,
                self.sizeof_heap_commit,
                char_count,
                dll_char_count,
            ]
            + char_features
            + dll_char_features
        )

        # Section features - convert all to numbers
        if not self.sections:
            section_features = [0] * 25
        else:
            section_sizes = [s.get("size", 0) for s in self.sections]
            section_entropies = [s.get("entropy", 0.0) for s in self.sections]
            section_vsizes = [s.get("vsize", 0) for s in self.sections]

            # Count section properties
            executable_sections = sum(
                1 for s in self.sections if "MEM_EXECUTE" in s.get("props", [])
            )
            writable_sections = sum(1 for s in self.sections if "MEM_WRITE" in s.get("props", []))
            readable_sections = sum(1 for s in self.sections if "MEM_READ" in s.get("props", []))

            # Section statistics
            total_sections = len(self.sections)
            total_size = sum(section_sizes)
            max_size = max(section_sizes) if section_sizes else 0
            min_size = min(section_sizes) if section_sizes else 0
            avg_size = total_size / total_sections if total_sections > 0 else 0

            # Entropy statistics
            max_entropy = max(section_entropies) if section_entropies else 0
            min_entropy = min(section_entropies) if section_entropies else 0
            avg_entropy = sum(section_entropies) / total_sections if total_sections > 0 else 0

            # Virtual size statistics
            total_vsize = sum(section_vsizes)
            max_vsize = max(section_vsizes) if section_vsizes else 0
            min_vsize = min(section_vsizes) if section_vsizes else 0
            avg_vsize = total_vsize / total_sections if total_sections > 0 else 0

            # Convert section entry string to numeric
            section_entry_numeric = hash(self.section_entry) % 10000 if self.section_entry else 0

            section_features = [
                total_sections,
                total_size,
                max_size,
                min_size,
                avg_size,
                executable_sections,
                writable_sections,
                readable_sections,
                max_entropy,
                min_entropy,
                avg_entropy,
                total_vsize,
                max_vsize,
                min_vsize,
                avg_vsize,
                section_entry_numeric,
            ]

            # Pad with zeros to get 25 features
            while len(section_features) < 25:
                section_features.append(0)

        # Import features
        dll_names = [
            "KERNEL32.dll",
            "USER32.dll",
            "GDI32.dll",
            "SHELL32.dll",
            "ADVAPI32.dll",
            "COMCTL32.dll",
            "ole32.dll",
            "VERSION.dll",
            "snmpapi.dll",
            "ws2_32.dll",
            "iphlpapi.dll",
            "wininet.dll",
        ]

        file_apis = ["CreateFile", "ReadFile", "WriteFile", "DeleteFile", "MoveFile", "CopyFile"]
        registry_apis = [
            "RegCreateKey",
            "RegSetValue",
            "RegQueryValue",
            "RegDeleteKey",
            "RegOpenKey",
        ]
        network_apis = [
            "connect",
            "send",
            "recv",
            "WSAConnect",
            "HttpOpenRequest",
            "InternetConnect",
        ]
        process_apis = [
            "CreateProcess",
            "CreateThread",
            "OpenProcess",
            "TerminateProcess",
            "LoadLibrary",
        ]
        memory_apis = ["VirtualAlloc", "VirtualFree", "HeapAlloc", "HeapFree", "LocalAlloc"]

        dll_imports = [len(self.import_data.get(dll, [])) for dll in dll_names]

        # Count API categories
        file_count = 0
        registry_count = 0
        network_count = 0
        process_count = 0
        memory_count = 0

        for api_list in self.import_data.values():
            for api in api_list:
                if any(fapi in api for fapi in file_apis):
                    file_count += 1
                if any(rapi in api for rapi in registry_apis):
                    registry_count += 1
                if any(napi in api for napi in network_apis):
                    network_count += 1
                if any(papi in api for papi in process_apis):
                    process_count += 1
                if any(mapi in api for mapi in memory_apis):
                    memory_count += 1

        total_imports = sum(len(apis) for apis in self.import_data.values())

        import_features = dll_imports + [
            file_count,
            registry_count,
            network_count,
            process_count,
            memory_count,
            total_imports,
        ]
        # Concatenate all features
        return np.concatenate(
            [
                normalized_hist,
                normalized_byteent,
                np.array(string_features, dtype=np.float32),
                np.array(general_features, dtype=np.float32),
                np.array(header_features, dtype=np.float32),
                np.array(section_features, dtype=np.float32),
                np.array(import_features, dtype=np.float32),
            ]
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "sha256": self.sha256,
            "md5": self.md5,
            "appeared": self.appeared,
            "label": self.label,
            "avclass": self.avclass,
            "histogram": self.histogram.tolist(),
            "byteentropy": self.byteentropy.tolist(),
            "strings": {
                "numstrings": self.numstrings,
                "avlength": self.avlength,
                "printabledist": self.printabledist,
                "printables": self.printables,
                "entropy": self.string_entropy,
                "paths": self.paths,
                "urls": self.urls,
                "registry": self.registry,
                "MZ": self.mz_count,
            },
            "general": {
                "size": self.size,
                "vsize": self.vsize,
                "has_debug": self.has_debug,
                "exports": self.exports,
                "imports": self.imports,
                "has_relocations": self.has_relocations,
                "has_resources": self.has_resources,
                "has_signature": self.has_signature,
                "has_tls": self.has_tls,
                "symbols": self.symbols,
            },
            "header": {
                "coff": {
                    "timestamp": self.timestamp,
                    "machine": self.machine,
                    "characteristics": self.characteristics,
                },
                "optional": {
                    "subsystem": self.subsystem,
                    "dll_characteristics": self.dll_characteristics,
                    "magic": self.magic,
                    "major_image_version": self.major_image_version,
                    "minor_image_version": self.minor_image_version,
                    "major_linker_version": self.major_linker_version,
                    "minor_linker_version": self.minor_linker_version,
                    "major_operating_system_version": self.major_operating_system_version,
                    "minor_operating_system_version": self.minor_operating_system_version,
                    "major_subsystem_version": self.major_subsystem_version,
                    "minor_subsystem_version": self.minor_subsystem_version,
                    "sizeof_code": self.sizeof_code,
                    "sizeof_headers": self.sizeof_headers,
                    "sizeof_heap_commit": self.sizeof_heap_commit,
                },
            },
            "section": {
                "entry": self.section_entry,
                "sections": self.sections,
            },
            "imports": self.import_data,
            "exports": self.exports,
            "datadirectories": self.datadirectories,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmberBinaryFeature":
        """Create instance from dictionary."""
        return cls(
            sha256=data["sha256"],
            md5=data["md5"],
            appeared=data["appeared"],
            label=data["label"],
            avclass=data["avclass"],
            histogram=data["histogram"],
            byteentropy=data["byteentropy"],
            strings=data["strings"],
            general=data["general"],
            header=data["header"],
            section=data["section"],
            imports=data["imports"],
            exports=data["exports"],
            datadirectories=data["datadirectories"],
        )

    def get_feature_names(self) -> List[str]:
        """Get all feature names in order."""
        # Histogram features
        histogram_names = [f"histogram_{i:02x}" for i in range(256)]

        # Byteentropy features
        byteentropy_names = [f"byteentropy_{i:02x}" for i in range(256)]

        # String features
        string_names = [
            "numstrings",
            "avlength",
            "printables",
            "string_entropy",
            "paths",
            "urls",
            "registry",
            "mz_count",
        ] + [f"printable_dist_{i}" for i in range(100)]

        # General features
        general_names = [
            "size",
            "vsize",
            "has_debug",
            "exports",
            "imports",
            "has_relocations",
            "has_resources",
            "has_signature",
            "has_tls",
            "symbols",
        ]

        # Header features
        header_names = [
            "timestamp",
            "machine_numeric",
            "subsystem_numeric",
            "magic_numeric",
            "major_image_version",
            "minor_image_version",
            "major_linker_version",
            "minor_linker_version",
            "major_operating_system_version",
            "minor_operating_system_version",
            "major_subsystem_version",
            "minor_subsystem_version",
            "sizeof_code",
            "sizeof_headers",
            "sizeof_heap_commit",
            "char_count",
            "dll_char_count",
            "char_32bit_machine",
            "char_relocs_stripped",
            "char_executable_image",
            "char_line_nums_stripped",
            "char_local_syms_stripped",
            "dll_dynamic_base",
            "dll_force_integrity",
            "dll_nx_compat",
            "dll_no_isolation",
            "dll_no_seh",
        ]

        # Section features
        section_names = [
            "total_sections",
            "total_size",
            "max_size",
            "min_size",
            "avg_size",
            "executable_sections",
            "writable_sections",
            "readable_sections",
            "max_entropy",
            "min_entropy",
            "avg_entropy",
            "total_vsize",
            "max_vsize",
            "min_vsize",
            "avg_vsize",
            "section_entry_numeric",
        ] + [f"padding_{i}" for i in range(9)]

        # Import features
        dll_names = [
            "KERNEL32.dll",
            "USER32.dll",
            "GDI32.dll",
            "SHELL32.dll",
            "ADVAPI32.dll",
            "COMCTL32.dll",
            "ole32.dll",
            "VERSION.dll",
            "snmpapi.dll",
            "ws2_32.dll",
            "iphlpapi.dll",
            "wininet.dll",
        ]
        import_names = dll_names + [
            "file_apis_count",
            "registry_apis_count",
            "network_apis_count",
            "process_apis_count",
            "memory_apis_count",
            "total_imports",
        ]

        return (
            histogram_names
            + byteentropy_names
            + string_names
            + general_names
            + header_names
            + section_names
            + import_names
        )

    def get_feature_sizes(self) -> Dict[str, int]:
        """Get the size of each feature group."""
        return {
            "histogram": 256,
            "byteentropy": 256,
            "strings": 108,  # 8 base + 100 printable_dist
            "general": 10,
            "header": 27,  # 17 base + 5 char + 5 dll
            "section": 25,
            "imports": 18,  # 12 dll + 6 api categories
        }

    def get_feature_ranges(self) -> Dict[str, tuple]:
        """Get the start and end indices for each feature group."""
        hist_size = 256
        byteent_size = 256
        strings_size = 108
        general_size = 10
        header_size = 27
        section_size = 25
        imports_size = 18

        return {
            "histogram": (0, hist_size),
            "byteentropy": (hist_size, hist_size + byteent_size),
            "strings": (hist_size + byteent_size, hist_size + byteent_size + strings_size),
            "general": (
                hist_size + byteent_size + strings_size,
                hist_size + byteent_size + strings_size + general_size,
            ),
            "header": (
                hist_size + byteent_size + strings_size + general_size,
                hist_size + byteent_size + strings_size + general_size + header_size,
            ),
            "section": (
                hist_size + byteent_size + strings_size + general_size + header_size,
                hist_size + byteent_size + strings_size + general_size + header_size + section_size,
            ),
            "imports": (
                hist_size + byteent_size + strings_size + general_size + header_size + section_size,
                hist_size
                + byteent_size
                + strings_size
                + general_size
                + header_size
                + section_size
                + imports_size,
            ),
        }


def build_features(ifile: str, ofile: str, limit: int = None):
    print(f"Loading features from {ifile} to {ofile}")
    df = pd.read_json(ifile, lines=True)
    if limit:
        df = df[:limit]
    df = df[df["label"].isin([0, 1])]
    features = df.to_dict(orient="records")
    data = {"features": [], "labels": []}
    print(f"Building features from {len(features)} samples")
    for feature in features:
        obj = EmberBinaryFeature.from_dict(feature)
        r = obj.to_input_layer_format()
        data["features"].append(r.tolist())
        data["labels"].append(feature["label"])

    print(f"Saving features to {ofile}")
    json.dump(data, open(ofile, "w"))


build_features(input_dir / "train_features_sample.jsonl", input_dir / "train_features_sample.json")
# build_features(input_dir / "train_features_0.jsonl", input_dir / "train_features_0.json")
# build_features(input_dir / "train_features_1.jsonl", input_dir / "train_features_1.json")
# build_features(input_dir / "train_features_2.jsonl", input_dir / "train_features_2.json")
# build_features(input_dir / "train_features_3.jsonl", input_dir / "train_features_3.json")
build_features(input_dir / "test_features.jsonl", input_dir / "test_features.json", limit=10000)
