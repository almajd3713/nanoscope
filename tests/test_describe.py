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


@pytest.fixture
def trained(fake_data):
    """A 2-layer Modern trained 20 steps, with steps 3 and 10 archived."""
    from nanoscope import run
    from nanoscope.store import ref_of

    preset = tiny(checkpoint_interval=5, keep_checkpoints=1)
    result = run(Modern, preset, device="cpu", checkpoint_steps=[3, 10], progress=False,
                 n_layers=2, d_model=32, n_heads=4)
    return ref_of(result.run_dir)


def test_inspect_checkpoint_maps_and_lens_at_any_archived_step(trained):
    import torch
    from helpers import assert_valid

    from nanoscope.inspect import inspect_checkpoint
    from nanoscope.store import load_run
    from nanoscope.train_loop import _split_output

    report = inspect_checkpoint(trained, "Once upon a time", step=3)
    assert_valid("inspect", report)
    assert (report["step"], report["steps"], report["model"]) == (3, [3, 10, 20], "Modern")
    n = len(report["tokens"])
    assert [a["module"] for a in report["attention"]] == ["blocks.0.attn", "blocks.1.attn"]
    for entry in report["attention"]:
        assert entry["heads"] == 4 and len(entry["weights"]) == 4
        for head in entry["weights"]:
            assert [len(row) for row in head] == list(range(1, n + 1))  # causal rows
            assert all(abs(sum(row) - 1) < 1e-3 for row in head)
    names = [layer["name"] for layer in report["lens"]["layers"]]
    assert names == ["embeddings", "blocks.0", "blocks.1"]
    positions = report["lens"]["layers"][-1]["positions"]
    assert positions[-1]["next"] is None
    assert positions[0]["next"]["id"] == report["tokens"][1]["id"]
    assert len(positions[0]["top"]) == 5

    # The last lens layer is the model's own output at that checkpoint, which differs by step.
    loaded = load_run(trained, step=3)
    with torch.no_grad():
        logits, _ = _split_output(loaded.model(torch.tensor([[t["id"] for t in report["tokens"]]])))
    assert [p["top"][0]["id"] for p in positions] == logits[0].argmax(-1).tolist()
    latest = inspect_checkpoint(trained, "Once upon a time")
    assert latest["step"] == 20 and latest["attention"] != report["attention"]

    with pytest.raises(FileNotFoundError, match=r"no checkpoint at step 7; it has 3, 10, 20"):
        inspect_checkpoint(trained, step=7)
    with pytest.raises(ValueError, match=r"the prompt is \d+ tokens; inspect takes at most 32"):
        inspect_checkpoint(trained, "once upon a time " * 20)


def test_inspect_checkpoint_of_a_model_without_attention(fake_data):
    from helpers import assert_valid

    from nanoscope import run
    from nanoscope.inspect import inspect_checkpoint
    from nanoscope.store import ref_of

    ref = ref_of(run(Bigram, tiny(), device="cpu", progress=False).run_dir)
    report = inspect_checkpoint(ref, "Once")
    assert_valid("inspect", report)
    assert report["attention"] == [] and [x["name"] for x in report["lens"]["layers"]] == ["output"]
    assert report["notes"] == ["Bigram has no Attention blocks, so there are no maps",
                               "Bigram has no blocks/norm/head, so the lens shows only the "
                               "model's output"]


def test_inspect_cli_prints_the_lens_and_where_heads_look(trained, capsys):
    import json

    from helpers import assert_valid

    from nanoscope.cli import main

    main(["inspect", trained, "--step", "10", "--prompt", "Once upon a time"])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith(f"{trained} at step 10 (checkpoints: 3, 10, 20); prompt of ")
    assert lines[2] == "logit lens: the top next-token guess after each layer (p)"
    assert lines[3].split()[:2] == ["pos", "token"] and "blocks.1" in lines[3]
    assert lines[3].endswith("actual next (p, rank)")
    assert any(line.startswith("attention from the last token ") for line in lines)
    head_line = next(line for line in lines if line.startswith("blocks.1.attn  h0 "))
    assert head_line.count(" h") == 4

    main(["inspect", trained, "--json", "--top-k", "2"])
    report = json.loads(capsys.readouterr().out)
    assert_valid("inspect", report)
    assert report["step"] == 20 and len(report["lens"]["layers"][0]["positions"][0]["top"]) == 2

    with pytest.raises(SystemExit, match="no checkpoint at step 4"):
        main(["inspect", trained, "--step", "4"])
