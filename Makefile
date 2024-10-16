# Adapted from https://raw.githubusercontent.com/huggingface/optimum/refs/heads/main/Makefile

SHELL := /bin/bash
CURRENT_DIR = $(shell pwd)
# DEFAULT_CLONE_URL := https://github.com/huggingface/optimum.git
# # If CLONE_URL is empty, revert to DEFAULT_CLONE_URL
# REAL_CLONE_URL = $(if $(CLONE_URL),$(CLONE_URL),$(DEFAULT_CLONE_URL))

.PHONY:	style test

# Run code quality checks
style_check:
	black --check .
	ruff check .

style:
	black .
	ruff check . --fix

# Run tests for the library
test:
	python -m pytest tests

# Utilities to release to PyPi
build_dist_install_tools:
	pip install build
	pip install twine

build_dist:
	rm -fr build
	rm -fr dist
	python -m build

