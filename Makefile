.PHONY: install train lint format clean test

install:
	uv sync

scrap:
	python src/scrap.py -o data/malware -l 1000

fetch-sorel:
	python src/Immune/data/fetch_sorel20M.py

process-assemblage:
	python src/Immune/data/process_assemblage.py

process-valtest:
	python src/Immune/data/process_dike_valtest.py

features-extraction:
	python src/features.py -im data/sorel20M -ib data/assemblagePE --output=data/features.json

features-extraction-valtest:
	python src/features.py -im data/DikeDataset-ValTest/benign -ib data/DikeDataset-ValTest/malware --output=data/DikeDataset-ValTest/features.json

features-input-layer-building-train-set:
	python src/Immune/features/input_layer_formatter.py -i data/features.json -o data/formatted_features.json

features-input-layer-building-valtest-set:
	python src/Immune/features/input_layer_formatter.py -i data/DikeDataset-ValTest/features.json -o data/DikeDataset-ValTest/formatted_features.json

ember-features-building:
	python src/ember/ember_build.py

train:
	python src/train.py --device cuda --learning-rate 0.0005 --epochs 200 --batch-size 1024

evaluate:
	python src/train.py --device cuda --evaluate

scaling-laws-mode:
	python src/train.py --device cuda --learning-rate 0.0005 --epochs 200 --batch-size 1024 --parallel-setups --max-parallel-workers 5

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
