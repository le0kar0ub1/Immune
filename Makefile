.PHONY: install train lint format clean test

install:
	uv sync

scrap:
	python src/scrap.py -o data/malware -l 1000

fetch-sorel:
	python src/Immune/data/fetch_sorel20M.py

process-assemblage:
	python src/Immune/data/process_assemblage.py

features-extraction:
	python src/features.py -im data/sorel20M -ib data/assemblagePE --output=data/features.json

features-input-layer-building:
	python src/Immune/features/input_layer_formatter.py

train:
	python src/train.py --device cuda

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

fix:
	ruff check --fix --unsafe-fixes src/ tests/

pre-commit: format fix lint

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -delete
	rm -rf build/ dist/ .pytest_cache/ .ruff_cache/

test:
	python -m pytest tests/ -v

all: format lint test
