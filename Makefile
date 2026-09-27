.PHONY: help setup test lint format quick run

PYTHON ?= python
CONFIG ?= configs/default.yaml

help:
	@echo "make setup   Install Python dependencies"
	@echo "make test    Run the test suite"
	@echo "make lint    Check code style (ruff + black)"
	@echo "make format  Auto-fix code style"
	@echo "make quick   Smoke run of the whole pipeline (configs/quick.yaml)"
	@echo "make run     Full experiment, e.g. make run CONFIG=configs/default.yaml"

setup:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m black --check src tests scripts

format:
	$(PYTHON) -m ruff check --fix src tests scripts
	$(PYTHON) -m black src tests scripts

quick:
	$(PYTHON) scripts/run_all.py --config configs/quick.yaml

run:
	$(PYTHON) scripts/run_all.py --config $(CONFIG)
