.PHONY: install lint test convert

install:
	pip install -e ".[dev,train]"

lint:
	ruff check .
	ruff format --check .

test:
	pytest

convert:
	python -m roadshift.data.convert --config configs/default.yaml
