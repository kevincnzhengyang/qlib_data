.PHONY: help install install-dev test lint clean

PYTHON ?= python
PIP ?= pip
PKG := qlib_data

help:  ## Show this help message
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:  ## Install the package in editable mode
	$(PIP) install -e .

install-dev:  ## Install the package with dev dependencies
	$(PIP) install -e ".[dev]"

test:  ## Run the test suite
	$(PYTHON) -m pytest tests/ -v

lint:  ## Run pyflakes on the package
	$(PYTHON) -m pyflakes $(PKG)/

clean:  ## Remove build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov/
