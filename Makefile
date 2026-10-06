.PHONY: install test test-all test-blocks lint typecheck check

install:  ## everything, including dev tools
	uv sync --all-extras

test:  ## fast offline tests
	uv run pytest -m "not network and not gpu"

test-all:  ## also the first-notebook timing test (downloads TinyStories once) and GPU tests
	uv run pytest

test-blocks:  ## the block library: blocks, references, graph, describe, block stats
	uv run pytest tests/test_blocks.py tests/test_reference.py tests/test_graph.py tests/test_describe.py tests/test_blockstats.py

lint:
	uv run ruff check nanoscope tests

typecheck:
	uv run pyright nanoscope

check: lint typecheck test  ## what CI runs
