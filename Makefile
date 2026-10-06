.PHONY: install test test-all test-blocks test-curricula test-curricula-full lint typecheck check

install:  ## everything, including dev tools
	uv sync --all-extras

test:  ## fast offline tests
	uv run pytest -m "not network and not gpu"

test-all:  ## also the first-notebook timing test (downloads TinyStories once) and GPU tests
	uv run pytest

test-blocks:  ## the block library: blocks, references, graph, describe, block stats
	uv run pytest tests/test_blocks.py tests/test_reference.py tests/test_graph.py tests/test_describe.py tests/test_blockstats.py

test-curricula:  ## every lesson's starter fails and its solution passes (offline, CPU)
	uv run pytest tests/test_curricula.py tests/test_checks.py tests/test_learn.py tests/test_gating.py -m "not network and not gpu"

test-curricula-full:  ## also train every lesson's solution on real TinyStories (minutes; needs network once)
	uv run pytest tests/test_curricula.py

lint:
	uv run ruff check nanoscope tests

typecheck:
	uv run pyright nanoscope

check: lint typecheck test  ## what CI runs
