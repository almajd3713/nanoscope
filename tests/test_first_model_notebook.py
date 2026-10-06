"""Guards the level 0 experience: the first notebook stays short and fast."""

import ast
import importlib.util
import json
import math
import time
from pathlib import Path

import pytest
import torch

NOTEBOOKS = Path(__file__).parent.parent / "notebooks"
NOTEBOOK = NOTEBOOKS / "marimo" / "01_first_model.py"
MAX_CODE_CELLS = 3
MAX_CPU_SECONDS = 120


def _code_cells() -> list[ast.FunctionDef]:
    """The marimo cells that hold code; `hide_code=True` marks the text-only ones."""
    tree = ast.parse(NOTEBOOK.read_text(encoding="utf-8"))
    cells = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.decorator_list:
            call = node.decorator_list[0]
            hidden = isinstance(call, ast.Call) and any(
                k.arg == "hide_code" and isinstance(k.value, ast.Constant) and k.value.value
                for k in call.keywords)
            if not hidden:
                cells.append(node)
    return cells


def test_first_notebook_has_at_most_three_code_cells():
    assert len(_code_cells()) <= MAX_CODE_CELLS


@pytest.mark.network
@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_first_notebook_trains_on_cpu_in_under_two_minutes(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    from nanoscope.dataset import load_data
    from nanoscope.presets import get_preset

    load_data(get_preset("tinystories-5min"))  # one-time data prep is not part of the budget
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    spec = importlib.util.spec_from_file_location("first_model_notebook", NOTEBOOK)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    start = time.perf_counter()
    _, defs = module.app.run()
    elapsed = time.perf_counter() - start

    result = defs["result"]
    assert elapsed < MAX_CPU_SECONDS, f"notebook took {elapsed:.0f}s on CPU"
    assert result.final_step == result.preset.max_steps
    assert result.val_losses[-1][1] < math.log(result.data.tokenizer.vocab_size) - 2
    assert result.samples
    assert "Δ vs GPT2" in str(defs["compare"](result, "gpt2"))


def test_kaggle_notebook_has_at_most_four_code_cells_using_real_commands():
    import shlex

    from nanoscope.cli import build_parser

    path = NOTEBOOKS / "kaggle.ipynb"
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    sources = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    assert len(sources) <= 4
    parser = build_parser()
    study = "studies/m1_ablation.py"
    assert f'STUDY = "{study}"' in sources[0]
    assert (NOTEBOOKS.parent / study).exists()
    commands = [line.removeprefix("!nanoscope ") for s in sources for line in s.splitlines()
                if line.startswith("!nanoscope ")]
    assert len(commands) == 2
    for command in commands:
        filled = command.replace("{STUDY}", study).replace("{RUNS_REPO}", "me/runs")
        args = parser.parse_args(shlex.split(filled))
        if args.command == "study":
            assert args.devices == "cuda:0,cuda:1" and args.push_to_hub == "me/runs"
