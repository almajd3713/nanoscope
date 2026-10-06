"""The model-file graph: parse, emit, and their round trip. Nothing here executes the files."""

from pathlib import Path

import pytest

from nanoscope.blocks.graph import parse

FIXTURES = Path(__file__).parent / "fixtures" / "graphs"


def graph_of(name):
    return parse(FIXTURES / f"{name}.py")


def one(name, cls=None):
    classes = graph_of(name)["classes"]
    return classes[0] if cls is None else next(c for c in classes if c["name"] == cls)


def test_parse_modern_like_decoder_into_nodes_and_spans():
    cls = one("modern_like")
    assert (cls["name"], cls["kind"], cls["representable"]) == ("MyModern", "decoder", True)
    assert [p["name"] for p in cls["params"]] == ["vocab_size", "context_length"]
    assert cls["params"][1]["default"] == 256
    args = cls["args"]
    # positional arguments are named by Decoder's signature
    assert args["vocab_size"] == {"kind": "param", "name": "vocab_size", "span": args[
        "vocab_size"]["span"]}
    assert args["context_length"]["kind"] == "param"
    assert args["d_model"]["value"] == 128 and args["tie_weights"]["value"] is True
    assert args["z_loss"]["value"] == 1e-4
    block = args["block"]
    assert (block["kind"], block["block"], block["family"]) == ("block", "Block", "structure")
    assert list(block["args"]) == ["norm", "mlp", "attn"]  # source order
    attn = block["args"]["attn"]
    assert attn["args"]["pos"]["block"] == "RoPE" and attn["args"]["qk_norm"]["value"] is True
    assert attn["args"]["n_kv_heads"]["value"] == 2
    assert block["args"]["mlp"]["args"]["hidden"]["value"] == 344
    # a span points at the source text
    lines = (FIXTURES / "modern_like.py").read_text().splitlines()
    span = block["args"]["mlp"]["args"]["hidden"]["span"]
    assert lines[span["line"] - 1][span["col"]:span["end_col"]] == "344"


def test_parse_aliases_and_layer_patterns():
    cls = one("layer_pattern")
    assert cls["name"] == "SlidingGlobal" and cls["representable"]
    pattern = cls["args"]["pattern"]
    assert pattern["kind"] == "list" and len(pattern["items"]) == 2
    sliding = pattern["items"][0]["args"]["attn"]
    assert sliding["block"] == "Attention"
    assert sliding["args"]["window"] == {"kind": "param", "name": "window",
                                         "span": sliding["args"]["window"]["span"]}
    assert "window" not in pattern["items"][1]["args"]["attn"]["args"]


def test_parse_gpt2_like_keeps_non_literal_free_params():
    cls = one("gpt2_like")
    assert cls["args"]["n_layers"]["kind"] == "param"
    assert cls["args"]["pos_emb"]["block"] == "LearnedPosition"
    assert cls["args"]["block"]["args"]["attn"]["args"]["bias"]["value"] is True


def test_unknown_calls_are_opaque_nodes_and_other_expressions_stay_text():
    cls = one("opaque_block")
    assert cls["representable"]
    block = cls["args"]["block"]["args"]
    assert block["attn"]["kind"] == "opaque" and block["attn"]["call"] == "Gated"
    assert block["attn"]["source"] == "Gated(n_heads=2)"
    assert block["mlp"]["kind"] == "opaque" and block["mlp"]["call"] == "mylib.FancyMLP"
    assert block["norm"]["kind"] == "block"  # the rest of the graph stays editable
    z = cls["args"]["z_loss"]
    assert (z["kind"], z["source"]) == ("expr", "2 * 1e-4 + 0")


def test_statements_outside_the_subset_make_a_class_code_only_with_reason_and_line():
    graph = graph_of("code_only")
    fine, dynamic, overrides = graph["classes"]
    assert fine["representable"] and fine["reason"] is None
    assert not dynamic["representable"]
    assert "assignment" in dynamic["reason"] and dynamic["reason_line"] == 12
    assert "args" not in dynamic
    assert not overrides["representable"]
    assert "forward" in overrides["reason"] and overrides["reason_line"] == 22


def test_composite_templates_list_their_slots_and_blocks_can_use_them():
    graph = graph_of("composite_template")
    template, user = graph["classes"]
    assert (template["kind"], template["slots"]) == ("composite", ["norm", "attn"])
    assert template["forward_span"]["line"] == 9
    node = user["args"]["block"]
    assert node["kind"] == "block" and node["block"] == "PreNormAttention"
    assert node["local"] is True and set(node["args"]) == {"norm", "attn"}


def test_comments_and_odd_formatting_do_not_change_the_graph():
    cls = one("odd_formatting")
    assert cls["representable"]
    assert cls["args"]["d_model"]["value"] == 96
    block = cls["args"]["block"]
    assert block["args"]["attn"]["args"] == {
        "n_heads": {"kind": "literal", "value": 6, "span": block["args"]["attn"]["args"][
            "n_heads"]["span"]},
        "n_kv_heads": {"kind": "literal", "value": 3, "span": block["args"]["attn"]["args"][
            "n_kv_heads"]["span"]}}


def test_parse_never_imports_the_file(tmp_path):
    path = tmp_path / "boom.py"
    path.write_text("raise RuntimeError('imported')\n"
                    "from nanoscope.blocks import Decoder\n"
                    "class M(Decoder):\n"
                    "    pass\n")
    (cls,) = parse(path)["classes"]
    assert cls["name"] == "M" and cls["args"] == {}
    path.write_text("def (:\n")
    with pytest.raises(ValueError, match=r"boom.py:1"):
        parse(path)


def test_classes_that_are_not_models_are_ignored(tmp_path):
    path = tmp_path / "plain.py"
    path.write_text("import torch.nn as nn\nclass A(nn.Module):\n    pass\n")
    assert parse(path)["classes"] == []


@pytest.mark.parametrize("name", sorted(p.stem for p in FIXTURES.glob("*.py")))
def test_schema_accepts_every_parsed_fixture(name):
    from helpers import assert_valid
    assert_valid("graph", graph_of(name))


def test_schema_rejects_a_malformed_graph():
    from helpers import assert_valid

    good = graph_of("modern_like")
    assert_valid("graph", good)
    bad = {**good, "classes": [{**good["classes"][0], "args": {"d_model": {"kind": "magic"}}}]}
    with pytest.raises(AssertionError, match="graph.v1 violated"):
        assert_valid("graph", bad)
    code_only = graph_of("code_only")["classes"][1]
    with pytest.raises(AssertionError, match="graph.v1 violated"):  # a code-only class has no args
        assert_valid("graph", {**good, "classes": [{**code_only, "args": {}}]})
