"""structure: what any nn.Module builds, with its lines, even while it is unfinished."""

from helpers import assert_valid

from nanoscope.inspect import structure
from nanoscope.learn.loader import curricula_dir, load_lesson
from nanoscope.modelref import load_class

HEAD = """import torch.nn as nn


class Part(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.fc(x)


class Whole(nn.Module):
    def __init__(self, vocab_size, d_model=8):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.parts = nn.ModuleList([Part(d_model) for _ in range(2)])
        self.out: nn.Linear = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        x = self.emb(idx)
        for part in self.parts:
            x = part(x)
        return self.out(x)
"""


def rows(report):
    return {r["path"]: r for r in report["modules"]}


def test_every_module_with_its_line_and_shapes(tmp_path):
    (tmp_path / "whole.py").write_text(HEAD)
    report = structure(load_class(f"{tmp_path / 'whole.py'}:Whole"))
    assert_valid("structure", report)
    assert (report["built"], report["traced"], report["error"]) == (True, True, None)
    assert report["args"] == {"vocab_size": 50, "d_model": 8}  # a small size, then its default
    assert report["inputs"] == [[2, 8]] and report["line"] == 13
    by = rows(report)
    assert by[""]["output_shapes"] == [[2, 8, 50]]
    assert by["emb"]["line"] == 16 and by["out"]["line"] == 18  # annotated too
    assert by["parts"]["line"] == 17 and by["parts.1"]["line"] == 17  # the list's line
    assert by["parts.0.fc"]["line"] == 7  # in the class that owns it
    assert by["parts.0.fc"]["params"] == 72 and by["parts.0.fc"]["input_shapes"] == [[2, 8, 8]]
    assert report["params"] == sum(r["params"] for r in report["modules"] if r["depth"] == 1)


def test_an_unfinished_starter_still_has_a_structure():
    lesson = curricula_dir() / "foundations" / "03-attention-head" / "starter.py"
    report = structure(load_class(f"{lesson}:OneHead"))
    assert_valid("structure", report)
    assert report["built"] and not report["traced"]
    assert report["args"] == {"d_model": 16, "context_length": 8}
    assert report["inputs"] == [[2, 8, 16]]  # no vocab_size: (batch, time, d_model) floats
    error = report["error"]
    assert (error["stage"], error["type"], error["module"]) == (
        "forward", "NotImplementedError", "")
    assert error["message"] == "attention: scores, mask, softmax, mix"
    assert lesson.read_text().splitlines()[error["line"] - 1].strip().startswith("raise")
    assert [r["path"] for r in report["modules"]] == [""]  # no layers written yet


def test_a_lesson_check_gives_the_args_and_inputs():
    lesson = load_lesson("foundations/05-block")
    check = next(c for c in lesson.checks if c.kind == "equivalent")
    solution = "tests/solutions/foundations/05-block.py"
    report = structure(load_class(f"{solution}:MyBlock"), check.args.get("args"),
                       check.args["inputs"])
    assert report["traced"] and report["inputs"] == [[2, 6, 16]]  # the first set of specs
    assert rows(report)["fc"]["output_shapes"] == [[2, 6, 64]]


def test_a_constructor_that_raises(tmp_path):
    (tmp_path / "bad.py").write_text(
        "import torch.nn as nn\n\n\nclass Bad(nn.Module):\n    def __init__(self, d_model):\n"
        "        super().__init__()\n        self.fc = nn.Linear(d_model, d_model)\n"
        "        raise ValueError('not yet')\n")
    report = structure(load_class(f"{tmp_path / 'bad.py'}:Bad"))
    assert_valid("structure", report)
    assert not report["built"] and report["modules"] == []
    assert report["error"] == {"stage": "build", "type": "ValueError", "message": "not yet",
                               "module": None, "line": 8}


def test_a_required_argument_with_no_small_size(tmp_path):
    (tmp_path / "odd.py").write_text(
        "import torch.nn as nn\n\n\nclass Odd(nn.Module):\n    def __init__(self, widthness):\n"
        "        super().__init__()\n")
    report = structure(load_class(f"{tmp_path / 'odd.py'}:Odd"))
    assert not report["built"] and report["inputs"] is None
    assert "needs a value for widthness" in report["error"]["message"]


def test_no_input_to_try(tmp_path):
    (tmp_path / "gate.py").write_text(
        "import torch.nn as nn\n\n\nclass Gate(nn.Module):\n    def __init__(self, n=3):\n"
        "        super().__init__()\n        self.w = nn.Linear(n, n)\n")
    report = structure(load_class(f"{tmp_path / 'gate.py'}:Gate"))
    assert report["built"] and not report["traced"] and report["error"] is None
    assert report["inputs"] is None and rows(report)["w"]["output_shapes"] is None
