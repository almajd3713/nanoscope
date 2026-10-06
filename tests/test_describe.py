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
