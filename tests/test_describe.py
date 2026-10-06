"""describe: what a model is on the meta device, without training or downloading anything."""

import inspect

import pytest
from fakes import tiny

from nanoscope.inspect import describe
from nanoscope.models import GPT2, Bigram, Modern
from nanoscope.sizing import build_on_meta, count_params, flops_per_token

SMALL = dict(n_layers=2, d_model=32, n_heads=4)


def rows_by_path(report):
    return {r["path"]: r for r in report["modules"]}


def test_describe_reports_shapes_params_flops_and_memory():
    report = describe(Modern, tiny(), **SMALL)
    assert report["model"] == "Modern" and report["preset"] == "test-tiny"
    assert report["kwargs"]["n_layers"] == 2 and report["kwargs"]["vocab_size"] == 300
    assert (report["context_length"], report["batch_size"]) == (32, 4)
    rows = rows_by_path(report)
    assert rows[""]["depth"] == 0 and rows["blocks.0"]["depth"] == 2
    attn = rows["blocks.0.attn"]
    assert (attn["type"], attn["block"], attn["family"], attn["tier"]) == (
        "Attention", "Attention", "attention", "composite")
    assert attn["input_shapes"] == [[4, 32, 32]] and attn["output_shapes"] == [[4, 32, 32]]
    assert rows["tok_emb"]["input_shapes"] == [[4, 32]]
    assert rows["tok_emb"]["output_shapes"] == [[4, 32, 32]]
    assert rows["head"]["output_shapes"] == [[4, 32, 300]]
    assert rows["blocks"]["input_shapes"] is None  # a ModuleList is never called
    assert attn["flops_source"] == "analytic" and rows["blocks.0.attn.q"]["flops_source"] == "6N"
    memory = report["memory"]
    assert memory["adamw_state"] == 2 * memory["weights"] == 2 * report["params"]["total"] * 4
    assert memory["total"] == sum(v for k, v in memory.items() if k != "total")
    assert memory["activations"] > 0


def test_describe_totals_equal_count_params_and_flops_per_token():
    for cls, kw in [(Bigram, {}), (GPT2, SMALL), (Modern, SMALL), (Modern, {
            **SMALL, "rope": False, "swiglu": False, "tie_weights": False})]:
        preset = tiny()
        report = describe(cls, preset, **kw)
        given = {k: v for k, v in {"vocab_size": 300, "context_length": 32}.items()
                 if k in inspect.signature(cls).parameters}
        model = build_on_meta(cls, **given, **kw)
        total, non_embedding = count_params(model)
        assert report["params"] == {"total": total, "non_embedding": non_embedding}, cls
        assert report["flops_per_token"] == flops_per_token(model, 32), cls
        assert rows_by_path(report)[""]["params"] == total


def test_describe_bigram_falls_back_to_6n():
    report = describe(Bigram, tiny())
    assert report["flops_source"] == "6N"
    assert report["flops_per_token"] == 6 * report["params"]["non_embedding"]


def test_describe_routes_keywords_like_run_and_refuses_unknown_ones():
    report = describe(Modern, tiny(), context_length=16, batch_size=2, **SMALL)
    assert (report["context_length"], report["batch_size"]) == (16, 2)
    assert rows_by_path(report)["head"]["output_shapes"] == [[2, 16, 300]]
    with pytest.raises(TypeError, match="no_such_knob"):
        describe(Modern, tiny(), no_such_knob=1)


def test_describe_a_composed_model(tmp_path):
    from pathlib import Path

    from nanoscope.modelref import load_class

    fixture = Path(__file__).parent / "fixtures" / "graphs" / "mylm.py"
    cls = load_class(str(fixture) + ":MyLM")
    report = describe(cls, tiny())
    assert report["model"] == "MyLM"
    assert rows_by_path(report)["blocks.3.attn"]["block"] == "Attention"


BAD_MODEL = '''\
import torch.nn as nn

from nanoscope.blocks import Attention, Block, Decoder, RMSNorm, SwiGLU
from nanoscope.blocks.spec import BlockModule


class Wrong(BlockModule):
    def __init__(self, d_model, context_length):
        super().__init__()
        self.fc = nn.Linear(d_model + 1, d_model)

    def forward(self, x):
        return self.fc(x)


class Broken(Decoder):
    def __init__(self, vocab_size: int, context_length: int = 32):
        super().__init__(
            vocab_size, context_length, d_model=32, n_layers=2,
            block=Block(norm=RMSNorm(),
                        attn=Wrong(),
                        mlp=SwiGLU()),
            final_norm=RMSNorm(),
        )
'''


def test_shape_error_line(tmp_path):
    from nanoscope.inspect import ShapeError
    from nanoscope.modelref import load_class

    path = tmp_path / "broken.py"
    path.write_text(BAD_MODEL)
    cls = load_class(f"{path}:Broken")
    with pytest.raises(ShapeError) as caught:
        describe(cls, tiny())
    err = caught.value
    assert err.module == "blocks.0.attn.fc"  # the innermost module that was running
    assert err.file == str(path) and err.line == 21  # the `attn=Wrong(),` line
    assert err.reason.startswith("RuntimeError")
    assert f"{path}:21" in str(err)


def test_shape_error_falls_back_to_the_traceback_for_code_only_models(tmp_path):
    from nanoscope.inspect import ShapeError
    from nanoscope.modelref import load_class

    path = tmp_path / "broken2.py"
    path.write_text("import torch.nn as nn\n\n\nclass Tiny(nn.Module):\n"
                    "    def __init__(self, vocab_size: int):\n        super().__init__()\n"
                    "        self.emb = nn.Embedding(vocab_size, 8)\n"
                    "        self.fc = nn.Linear(9, 8)\n\n"
                    "    def forward(self, idx):\n        return self.fc(self.emb(idx))\n")
    with pytest.raises(ShapeError) as caught:
        describe(load_class(f"{path}:Tiny"), tiny())
    assert caught.value.module == "fc"
    assert caught.value.line is None or caught.value.file == str(path)


def test_cli_prints_a_table_and_json(capsys, tmp_path):
    import json

    from helpers import assert_valid

    from nanoscope.cli import main

    main(["describe", "nanoscope/models/modern.py:Modern", "--preset", "tinystories-5min",
          "--set", "n_layers=2", "d_model=32", "n_heads=4", "context_length=64"])
    text = capsys.readouterr().out
    lines = text.splitlines()
    assert lines[0] == "Modern  preset tinystories-5min  batch 8 x context 64"
    assert lines[1].startswith("params ") and "FLOPs/token" in lines[1] and "(analytic)" in lines[1]
    assert lines[2].startswith("memory ~")
    header = next(line for line in lines if line.startswith("module"))
    assert header.split() == ["module", "type", "in", "out", "params", "FLOPs/token"]
    attn = next(line for line in lines if line.lstrip().startswith("attn "))
    assert attn.index("attn") == 6 and "Attention" in attn and "8x64x32" in attn  # depth 3
    assert lines[-1].startswith("* FLOPs/token estimated")

    main(["describe", "nanoscope/models/gpt2.py:GPT2", "--json", "--set", "n_layers=1",
          "d_model=32", "n_heads=4"])
    report = json.loads(capsys.readouterr().out)
    assert_valid("describe", report)
    assert report["model"] == "GPT2" and report["kwargs"]["n_layers"] == 1

    broken = tmp_path / "broken.py"
    broken.write_text(BAD_MODEL)
    with pytest.raises(SystemExit, match="shape error in blocks.0.attn.fc: .*broken.py:21"):
        main(["describe", f"{broken}:Broken"])


def test_describe_json_validates_for_every_shipped_model():
    from helpers import assert_valid

    for cls, kw in [(Bigram, {}), (GPT2, SMALL), (Modern, SMALL)]:
        assert_valid("describe", describe(cls, tiny(), **kw))
