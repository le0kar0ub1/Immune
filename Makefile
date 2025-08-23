.PHONY: install train lint format clean test

install:
	uv sync

train:
	python src/train.py

features:
	python src/features.py

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
