.PHONY: install test test-all lint

install:  ## everything, including dev tools
	uv sync --all-extras

test:  ## fast offline tests
	uv run pytest -m "not network and not gpu"

test-all:  ## also the first-notebook timing test (downloads TinyStories once) and GPU tests
	uv run pytest

lint:
	uv run ruff check nanoscope tests
