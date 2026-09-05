# Makefile for Nanoscope

CONFIG ?= configs/m0/local-smoke.yaml

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  help          - Show this help"
	@echo "  install       - Install dependencies"
	@echo "  sync          - Sync the workspace with Kaggle"
	@echo "  train         - Train the model"
	@echo "  doctor        - Check the environment"
	@echo "  clean         - Clean the workspace"

.PHONY: install
install:
	uv tool install kaggle
	uv pip install -r requirements.txt
	uv pip install -e .
	

.PHONY: sync
sync:
	python3 kaggle_sync.py

.PHONY: train
train:
	uv run nanoscope train --config $(CONFIG) --resume none

.PHONY: doctor
doctor:
	uv run nanoscope doctor --config $(CONFIG)

.PHONY: clean
clean:
	rm -rf runs/*