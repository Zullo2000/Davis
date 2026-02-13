.PHONY: install test lint format data train evaluate sample

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check bd_gen/ tests/

format:
	ruff format bd_gen/ tests/

data:
	python scripts/prepare_data.py

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

sample:
	python scripts/sample.py
