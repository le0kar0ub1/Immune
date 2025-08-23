.PHONY: help install install-dev test lint format clean run train train-custom

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	uv sync

install-dev: ## Install development dependencies
	uv sync --extra dev

test: ## Run tests
	uv run pytest

test-cov: ## Run tests with coverage
	uv run pytest --cov=src --cov-report=html

lint: ## Run linting with ruff
	uv run ruff check src tests

format: ## Format code with black and ruff
	uv run black src tests
	uv run ruff format src tests

type-check: ## Run type checking with mypy
	uv run mypy src

clean: ## Clean up cache and build files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf .coverage htmlcov/

run: ## Run the main application
	uv run python -m src.main

dev: ## Start development server
	uv run python -m src.main --dev

train: ## Train the malware detection model (requires data/ directory)
	@if [ ! -d "data" ]; then \
		echo "Error: data/ directory not found. Please create it with malware/ and benign/ subdirectories."; \
		echo "Example structure:"; \
		echo "  data/"; \
		echo "  ├── malware/"; \
		echo "  │   ├── sample1.exe"; \
		echo "  │   └── sample2.dll"; \
		echo "  └── benign/"; \
		echo "      ├── sample1.exe"; \
		echo "      └── sample2.dll"; \
		exit 1; \
	fi
	uv run python scripts/train_model.py data/

train-custom: ## Train with custom parameters (usage: make train-custom EPOCHS=100 LR=0.0005)
	@if [ ! -d "data" ]; then \
		echo "Error: data/ directory not found. Please create it with malware/ and benign/ subdirectories."; \
		exit 1; \
	fi
	uv run python scripts/train_model.py data/ \
		--epochs $(or $(EPOCHS),40) \
		--learning-rate $(or $(LR),0.001) \
		--device $(or $(DEVICE),cpu) \
		--model-save-path $(or $(MODEL_PATH),models/malware_detector.pt)

analyze: ## Analyze a file for malware (usage: make analyze FILE=path/to/file.exe)
	@if [ -z "$(FILE)" ]; then \
		echo "Error: Please specify FILE parameter"; \
		echo "Usage: make analyze FILE=path/to/file.exe"; \
		exit 1; \
	fi
	uv run python -m src.main "$(FILE)"

analyze-with-model: ## Analyze with trained model (usage: make analyze-with-model FILE=path/to/file.exe MODEL=models/model.pt)
	@if [ -z "$(FILE)" ] || [ -z "$(MODEL)" ]; then \
		echo "Error: Please specify both FILE and MODEL parameters"; \
		echo "Usage: make analyze-with-model FILE=path/to/file.exe MODEL=models/model.pt"; \
		exit 1; \
	fi
	uv run python -m src.main --model "$(MODEL)" "$(FILE)"

setup-dirs: ## Create necessary directories for the project
	mkdir -p data/malware data/benign models reports logs

demo: ## Run a quick demo (development mode)
	uv run python -m src.main --dev

pre-commit-install: ## Install pre-commit hooks
	uv run pre-commit install

pre-commit-run: ## Run pre-commit hooks on all files
	uv run pre-commit run --all-files

