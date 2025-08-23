# Immune 🛡️

A machine learning-based malware detection system built with PyTorch, featuring a **multi-headed neural network architecture** designed to identify malicious software through specialized feature processing and advanced neural network analysis.

## 🏗️ Multi-Headed Architecture

The system uses a sophisticated multi-headed neural network that processes different types of features separately before fusion:

### **Head 1: PE Header Features** (50 → 64 → 32)
- Processes Portable Executable file headers
- Extracts metadata, section information, and import/export data
- Outputs 32-dimensional latent representation

### **Head 2: Byte Histogram** (256 → 256 → 128)
- Analyzes byte frequency distributions
- Captures statistical patterns in binary data
- Outputs 128-dimensional latent representation

### **Head 3: Opcode Features** (300 → 256 → 128)
- Counts and analyzes assembly opcodes
- Identifies suspicious instruction patterns
- Outputs 128-dimensional latent representation

### **Head 4: API Call Features** (200 → 128 → 64)
- Detects Windows API calls and system functions
- Identifies potentially malicious behaviors
- Outputs 64-dimensional latent representation

### **Fusion Layer** (352 → 256 → 128 → 2)
- Combines all latent representations
- Processes through dense layers with dropout
- Final output: binary classification (Malware vs Benign)

## ✨ Features

- **Advanced Feature Extraction**: PE file analysis, entropy calculation, string analysis, byte histogram, opcode counting, and API call detection
- **Multi-Headed Deep Learning**: Specialized neural networks for each feature type
- **YARA Integration**: Rule-based detection combined with ML predictions
- **Comprehensive Testing**: Full test suite with pytest and coverage reporting
- **Modern Development Tools**: Ruff linter, Black formatter, MyPy type checking, and pre-commit hooks
- **Training & Evaluation**: Complete training pipeline with visualization and reporting

## 🚀 Quick Start

### 1. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and setup the project

```bash
git clone <your-repo-url>
cd Immune
uv sync --extra dev
```

### 3. Run the application

```bash
# Development mode (shows architecture info)
uv run python -m src.main --dev

# Analyze a file (feature extraction only)
uv run python -m src.main /path/to/suspicious/file.exe

# Analyze with trained model
uv run python -m src.main --model models/malware_detector.pt /path/to/file.exe
```

## 🎯 Training Your Own Model

### 1. Prepare your dataset

Organize your samples in the following structure:
```
data/
├── malware/
│   ├── sample1.exe
│   ├── sample2.dll
│   └── ...
└── benign/
    ├── sample1.exe
    ├── sample2.dll
    └── ...
```

### 2. Train the model

```bash
# Basic training
uv run python scripts/train_model.py data/

# Advanced training with custom parameters
uv run python scripts/train_model.py data/ \
    --epochs 100 \
    --learning-rate 0.0005 \
    --device cuda \
    --model-save-path models/custom_detector.pt
```

### 3. Use your trained model

```bash
uv run python -m src.main --model models/custom_detector.pt /path/to/file.exe
```

## 🏗️ Project Structure

```
Immune/
├── src/                    # Source code
│   ├── models/            # ML model implementations
│   │   └── malware_detector.py  # Multi-headed neural network
│   ├── data/              # Data processing and feature extraction
│   │   └── feature_extractor.py # Multi-headed feature extractor
│   ├── utils/             # Utility functions
│   │   └── visualization.py     # Training plots and reports
│   └── main.py            # Main application entry point
├── scripts/                # Training and utility scripts
│   └── train_model.py     # Model training script
├── tests/                  # Test suite
├── pyproject.toml         # Project configuration and dependencies
├── .pre-commit-config.yaml # Pre-commit hooks configuration
├── Makefile               # Development commands
└── README.md              # This file
```

## 🛠️ Development

### Available Commands

```bash
# Install dependencies
make install          # Production dependencies
make install-dev      # Development dependencies

# Code quality
make lint             # Run ruff linter
make format           # Format code with black and ruff
make type-check       # Run MyPy type checking

# Testing
make test             # Run tests
make test-cov         # Run tests with coverage

# Cleanup
make clean            # Remove cache and build files

# Run application
make run              # Run main application
make dev              # Start development server
```

### Pre-commit Hooks

The project includes pre-commit hooks that automatically:
- Format code with Black
- Fix issues with Ruff
- Run type checking with MyPy
- Check for common issues

Install the hooks:
```bash
uv run pre-commit install
```

## 📊 Model Performance

The multi-headed architecture provides several advantages:

- **Specialized Processing**: Each head learns optimal representations for its feature type
- **Better Generalization**: Separate processing reduces overfitting
- **Interpretability**: Can analyze which feature types contribute most to predictions
- **Modularity**: Easy to add/remove feature types or modify individual heads

## 🔍 Feature Extraction Details

### PE Header Features (50 dimensions)
- File size, entry point, base addresses
- Section information (up to 5 sections)
- Import/export counts
- Subsystem and DLL characteristics
- Stack/heap configuration

### Byte Histogram Features (256 dimensions)
- Normalized byte frequency distribution
- Captures statistical patterns in binary data
- Useful for detecting packed/encrypted content

### Opcode Features (300 dimensions)
- Counts of common x86 opcodes
- Entropy analysis across different block sizes
- Pattern recognition in instruction sequences

### API Call Features (200 dimensions)
- Windows API function counts
- System call patterns
- String-based feature extraction
- Suspicious behavior indicators

## 🎓 Training Configuration

### Default Hyperparameters
- **Epochs**: 40 (with early stopping)
- **Learning Rate**: 0.001
- **Batch Size**: 64
- **Optimizer**: Adam
- **Loss Function**: Cross-Entropy
- **Early Stopping**: 10 epochs patience
- **Learning Rate Scheduling**: ReduceLROnPlateau

### Training Features
- Automatic train/validation split (80/20)
- Early stopping to prevent overfitting
- Learning rate scheduling
- Comprehensive training reports
- Visualization of training progress

## 📈 Evaluation Metrics

The system provides comprehensive evaluation including:
- Training and validation loss curves
- Confusion matrices
- ROC curves and AUC scores
- Precision-recall curves
- Feature importance analysis
- Per-head contribution analysis

## 🔧 Configuration

Copy the example configuration and customize:
```bash
cp config.example.env .env
# Edit .env with your settings
```

Key configuration options:
- Model architecture parameters
- Training hyperparameters
- Feature extraction settings
- File size and extension limits

## 🚨 Security Notice

⚠️ **Important**: This tool is for educational and research purposes. Always use in isolated environments when analyzing potentially malicious files. The authors are not responsible for any damage caused by misuse of this software.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make test`
5. Format code: `make format`
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🗺️ Roadmap

- [ ] Support for more file formats (ELF, Mach-O)
- [ ] Real-time monitoring capabilities
- [ ] Cloud-based model serving
- [ ] Advanced evasion technique detection
- [ ] Integration with threat intelligence feeds
- [ ] Web-based user interface
- [ ] Additional feature heads (network traffic, registry changes)
- [ ] Ensemble methods with multiple models
- [ ] Transfer learning from pre-trained models
