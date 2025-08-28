# Immune 🛡️

A chronological journey through building malware detection models, from custom datasets to established benchmarks.

## The Journey

### Phase 1: Building Our Own Dataset (The Hard Way)

#### Initial Ambition
We started with an ambitious goal: build a malware detection system from scratch using our own dataset. We combined two major sources:
- **SOREL-20M**: A massive malware dataset with 20 million samples from [https://github.com/sophos/SOREL-20M/tree/master](https://github.com/sophos/SOREL-20M/tree/master)
- **Assemblage PE**: Benign PE files from [https://huggingface.co/datasets/changliu8541/Assemblage_PE/tree/main](https://huggingface.co/datasets/changliu8541/Assemblage_PE/tree/main)

#### Feature Engineering
We extracted approximately 350 features per sample, including:
- **PE Metadata**: File headers, section information, import/export tables
- **Byte Histogram**: 256-bin frequency distribution of byte values
- **Entropy Analysis**: Overall and per-section entropy calculations
- **Windows API Calls**: Categorized by functionality (file, registry, network, process, memory, system, crypto, anti-debug)

#### First Attempt: Deep Neural Network
Our initial approach used a DNN architecture. The results were... disappointing. Performance was essentially random guessing, suggesting our feature representation or model architecture wasn't capturing the underlying patterns.

#### Pivot to XGBoost
We switched to XGBoost to establish a baseline. The results were surprisingly good... too good. This raised our suspicions about potential dataset bias.

#### Reality Check: External Validation
To validate our results, we brought in a test dataset from another source that combined malware and benign samples. The performance dropped to random guessing levels, revealing a critical insight: **our original dataset was biased because malware and benign samples came from different distributions**.

### Phase 2: Embracing the Ember Dataset

#### Learning from Experience
After the sobering results with our custom dataset, we decided to use the established **Ember dataset** - a well-curated, balanced dataset specifically designed for malware detection research.

#### Building Our Subset
We created a manageable subset of 100K samples with 700 features.

#### XGBoost Success
The results were impressive:
- **Accuracy**: ~87%
- **False Positive Rate**: < 1%
- **Training Time**: Few dozen seconds
- **Data Usage**: 10x less than our original attempt

These results are particularly noteworthy when compared to the [Ember paper](https://arxiv.org/pdf/1804.04637), where we achieved similar performance with significantly less data and training time.

#### DNN Redux
We revisited deep learning with the Ember dataset, but results remained suboptimal:
- **Accuracy**: ~60%
- **False Positive Rate**: < 50%

## Key Lessons Learned

1. **Distribution Matters**: Malware and benign samples must come from similar real-world distributions
2. **Baseline First**: XGBoost provided an excellent baseline that helped us understand what was achievable
3. **Feature Engineering**: Our custom features showed promise but needed proper validation
4. **Established Benchmarks**: The Ember dataset provided a reliable foundation for model development
5. **Accuracy is relative**: for this kind of task you want to look for FPR and TPR

## Sample Data Formats

### Our Custom Feature Extraction Output

```json
{
  "file_size": 27136,
  "pe_features": {
    "size_of_code": 14848,
    "size_of_initialized_data": 324096,
    "size_of_uninitialized_data": 0,
    "address_of_entry_point": 15816,
    "base_of_code": 0,
    "base_of_data": 0,
    "image_base": 5368709120,
    "section_alignment": 4096,
    "file_alignment": 512,
    "major_os_version": 6,
    "minor_os_version": 0,
    "major_image_version": 0,
    "minor_image_version": 0,
    "major_subsystem_version": 6,
    "minor_subsystem_version": 0,
    "size_of_image": 360448,
    "size_of_headers": 1024,
    "checksum": 0,
    "subsystem": 2,
    "dll_characteristics": 33120,
    "size_of_stack_reserve": 1048576,
    "size_of_stack_commit": 4096,
    "size_of_heap_reserve": 1048576,
    "size_of_heap_commit": 4096,
    "loader_flags": 0,
    "number_of_rva_and_sizes": 16,
    "number_of_sections": 7,
    "total_section_size": 26112,
    "max_section_size": 14848,
    "min_section_size": 512,
    "avg_section_size": 3730.285714285714,
    "executable_sections": 1,
    "writable_sections": 1,
    "suspicious_sections": 0,
    "number_of_imports": 79,
    "number_of_exports": 0
  },
  "byte_histogram": {
    "histogram": [/* 256 values representing byte frequency distribution */]
  },
  "entropy_features": {
    "overall_entropy": 5.646893595283354,
    "data_section_entropy": 0.0,
    "rodata_section_entropy": 0.0,
    "text_section_entropy": 0.0,
    "min_section_entropy": 3.4566291559391,
    "max_section_entropy": 6.179212956280428,
    "avg_section_entropy": 5.083369041133698,
    "std_section_entropy": 0.9526227409592255,
    "high_entropy_sections": 0,
    "low_entropy_sections": 1,
    "entropy_variance": 0.9074900865926677
  },
  "api_features": {
    "file_apis": {
      "CreateFile": 0,
      "ReadFile": 0,
      "WriteFile": 0,
      "DeleteFile": 0,
      "MoveFile": 0,
      "CopyFile": 0
    },
    "registry_apis": {
      "RegCreateKey": 0,
      "RegSetValue": 0,
      "RegQueryValue": 0,
      "RegDeleteKey": 0,
      "RegOpenKey": 0
    },
    "network_apis": {
      "connect": 0,
      "send": 0,
      "recv": 0,
      "WSAConnect": 0,
      "HttpOpenRequest": 0
    },
    "process_apis": {
      "CreateProcess": 0,
      "CreateThread": 0,
      "OpenProcess": 0,
      "TerminateProcess": 1,
      "LoadLibrary": 0
    },
    "memory_apis": {
      "VirtualAlloc": 0,
      "VirtualFree": 0,
      "HeapAlloc": 0,
      "HeapFree": 0,
      "LocalAlloc": 0
    },
    "system_apis": {
      "GetSystemTime": 1,
      "GetComputerName": 0,
      "GetUserName": 0,
      "GetWindowsDirectory": 0
    },
    "crypto_apis": {
      "CryptCreateHash": 0,
      "CryptHashData": 0,
      "CryptEncrypt": 0,
      "CryptDecrypt": 0
    },
    "anti_debug_apis": {
      "IsDebuggerPresent": 1,
      "CheckRemoteDebuggerPresent": 0,
      "GetTickCount": 0
    },
    "total_strings": 648,
    "avg_string_length": 7.157407407407407,
    "max_string_length": 112,
    "min_string_length": 3
  }
}
```

### Ember Dataset Sample

The Ember dataset provides a comprehensive set of features including:
- **General**: File size, virtual size, debug info, exports/imports
- **Header**: COFF and optional header information
- **Section**: Detailed section analysis with entropy and properties
- **Imports**: DLL and function import analysis
- **Strings**: String analysis with entropy and distribution
- **Byte Histogram**: 256-bin byte frequency distribution
- **Byte Entropy**: Entropy analysis across byte ranges

## Current Status

We've successfully established a working malware detection system using the Ember dataset with XGBoost, achieving production-ready performance metrics. The journey taught us valuable lessons about dataset quality, distribution bias, and the importance of proper validation.