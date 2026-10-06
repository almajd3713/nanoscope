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


from nanoscope.blocks.graph import apply_edits, diff, emit  # noqa: E402


def edited(name, change):
    """(source, emitted source) after `change(first class args)` edits the parsed graph."""
    path = FIXTURES / f"{name}.py"
    source = path.read_text()
    graph = parse(path)
    change(graph["classes"][0]["args"])
    return source, emit(graph, source)


def changed_lines(before, after):
    old, new = before.splitlines(), after.splitlines()
    assert len(old) == len(new), "an edit should not add or remove lines here"
    return [(i + 1, a, b) for i, (a, b) in enumerate(zip(old, new, strict=True)) if a != b]


def lit(value):
    return {"kind": "literal", "value": value, "span": {"line": 1, "col": 0, "end_line": 1,
                                                        "end_col": 1}}


def test_minimal_diff_literal_edits_change_one_line():
    def bigger(args):
        args["d_model"]["value"] = 256
        args["block"]["args"]["attn"]["args"]["n_heads"]["value"] = 8
    before, after = edited("modern_like", bigger)
    lines = changed_lines(before, after)
    assert [n for n, _, _ in lines] == [7, 9]  # one line per edit
    assert "d_model=256" in after and "n_heads=8" in after
    assert parse_text(after)["classes"][0]["args"]["d_model"]["value"] == 256
    # the rest of both lines is untouched
    assert lines[0][1].replace("128", "256") == lines[0][2]


def parse_text(source):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "m.py"
        path.write_text(source)
        return parse(path)


def test_minimal_diff_add_and_remove_arguments():
    def add(args):
        args["block"]["args"]["attn"]["args"]["window"] = lit(16)
    before, after = edited("modern_like", add)
    ((line, old, new),) = changed_lines(before, after)
    assert line == 9 and "qk_norm=True, window=16)" in new

    def remove(args):
        del args["block"]["args"]["attn"]["args"]["n_kv_heads"]
    before, after = edited("modern_like", remove)
    ((line, old, new),) = changed_lines(before, after)
    assert "n_kv_heads" not in new and new.count("Attention(") == 1

    def remove_last(args):
        del args["z_loss"]
    before, after = edited("modern_like", remove_last)
    ((line, old, new),) = changed_lines(before, after)
    assert line == 11 and new == "            tie_weights=True,"


def test_minimal_diff_replace_block_and_pattern_items():
    def swap(args):
        args["block"]["args"]["mlp"] = {
            "kind": "block", "block": "SwiGLU", "args": {"hidden": lit(512)}, "span": None}
        args["final_norm"] = {"kind": "block", "block": "RMSNorm", "args": {"eps": lit(1e-5)},
                              "span": None}
    before, after = edited("modern_like", swap)
    assert [n for n, _, _ in changed_lines(before, after)] == [8, 10]
    assert "SwiGLU(hidden=512)" in after and "RMSNorm(eps=1e-05)" in after

    def pattern(args):
        first = args["pattern"]["items"][0]["args"]["attn"]["args"]
        first["n_heads"]["value"] = 2
        args["pattern"]["items"][1]["args"]["mlp"] = {
            "kind": "block", "block": "SwiGLU", "args": {"hidden": lit(64)}, "span": None}
    before, after = edited("layer_pattern", pattern)
    assert len(changed_lines(before, after)) == 2
    assert "nb.SwiGLU(hidden=64)" in after  # this file spells blocks through `nb.`


def test_minimal_diff_keeps_comments_and_odd_spacing():
    def change(args):
        args["d_model"]["value"] = 128
        args["block"]["args"]["attn"]["args"]["n_heads"]["value"] = 4
    before, after = edited("odd_formatting", change)
    lines = changed_lines(before, after)
    assert len(lines) == 2
    assert "d_model = 128 ,  # spaces around =" in after
    assert "# the attention" in after and "# from the tokenizer" in after


def test_emit_adds_missing_imports():
    def swap(args):
        args["block"]["args"]["attn"]["args"]["pos"] = {
            "kind": "block", "block": "NoPE", "args": {}, "span": None}
    source = (FIXTURES / "gpt2_like.py").read_text()
    graph = parse_text(source)
    swap(graph["classes"][0]["args"])
    after = emit(graph, source)
    assert "NoPE" in after.splitlines()[0]  # joined the existing import line
    assert "pos=NoPE()" in after
    again = parse_text(after)["classes"][0]["args"]["block"]["args"]["attn"]["args"]["pos"]
    assert again["kind"] == "block" and again["block"] == "NoPE"
    # a file with no blocks import gets one, after its other imports
    bare = ("import torch\nimport nanoscope.blocks.structure as s\n\n\n"
            "class M(s.Decoder):\n    def __init__(self):\n"
            "        super().__init__(1, 2, 3, 4, block=Block())\n")
    patched = apply_edits(bare, [{"op": "set_arg", "class": "M", "path": [], "arg": "pattern",
                                  "value": {"kind": "list", "span": None, "items": [
                                      {"kind": "block", "block": "RMSNorm", "args": {},
                                       "span": None}]}}])
    assert patched.splitlines()[:3] == [
        "import torch", "import nanoscope.blocks.structure as s",
        "from nanoscope.blocks import RMSNorm"]


def test_edits_that_cannot_apply_are_refused():
    source = (FIXTURES / "code_only.py").read_text()
    graph = parse_text(source)
    graph["classes"][1]["representable"] = True
    with pytest.raises(ValueError, match="code-only"):
        emit({**graph, "classes": [{**c, "args": {"d_model": lit(1)}} if c["name"] == "Dynamic"
                                   else c for c in graph["classes"]]}, source)
    modern = (FIXTURES / "modern_like.py").read_text()
    with pytest.raises(ValueError, match="no argument 'nope'"):
        apply_edits(modern, [{"op": "set_arg", "class": "MyModern", "path": ["nope"],
                              "arg": "x", "value": lit(1)}])
    with pytest.raises(ValueError, match="no Decoder or Composite class 'Ghost'"):
        apply_edits(modern, [{"op": "remove_arg", "class": "Ghost", "path": [], "arg": "x"}])
    with pytest.raises(ValueError, match="cannot add, remove or rename classes"):
        diff(parse_text(modern), {"classes": []})


@pytest.mark.parametrize("name", sorted(p.stem for p in FIXTURES.glob("*.py")))
def test_unedited_round_trip_is_byte_identical(name):
    import libcst as cst

    path = FIXTURES / f"{name}.py"
    source = path.read_text()
    assert cst.parse_module(source).code == source  # libcst itself loses nothing
    graph = parse(path)
    assert diff(graph, parse(path)) == []
    assert emit(graph, source) == source
    assert apply_edits(source, []) == source


def test_edits_leave_the_rest_of_the_file_byte_identical():
    path = FIXTURES / "odd_formatting.py"
    source = path.read_text()
    graph = parse(path)
    graph["classes"][0]["args"]["n_layers"]["value"] = 4
    after = emit(graph, source)
    assert after == source.replace("n_layers=3", "n_layers=4")


def _graph_strategy():
    from hypothesis import strategies as st

    span = {"line": 1, "col": 0, "end_line": 1, "end_col": 1}

    def literal(values):
        return values.map(lambda v: {"kind": "literal", "value": v, "span": span})

    def block(name, **args):
        return {"kind": "block", "block": name, "args": {k: v for k, v in args.items()
                                                         if v is not None}, "span": span}

    pos = st.sampled_from([None, "RoPE", "NoPE"])
    attn = st.builds(
        lambda heads, kv, p, qk, window: block(
            "Attention", n_heads=heads, n_kv_heads=kv, qk_norm=qk, window=window,
            pos=None if p is None else block(p, **({"base": None} if p == "NoPE" else {}))),
        literal(st.sampled_from([2, 4, 8])),
        st.none() | literal(st.sampled_from([1, 2])),
        pos,
        st.none() | literal(st.booleans()),
        st.none() | literal(st.integers(1, 64)))
    norm = st.sampled_from(["RMSNorm", "LayerNorm"]).map(lambda n: block(n))
    mlp = st.one_of(
        st.builds(lambda h: block("SwiGLU", hidden=h), st.none() | literal(st.integers(8, 512))),
        st.builds(lambda b: block("GELUMLP", bias=b), st.none() | literal(st.booleans())))
    layer = st.builds(lambda n, a, m: block("Block", norm=n, attn=a, mlp=m), norm, attn, mlp)

    def decoder(base_args, layers):
        return st.builds(
            lambda d, n, b, pattern, final, tie, z: {
                **{k: v for k, v in base_args.items() if k in ("vocab_size", "context_length")},
                **{k: v for k, v in {
                    "d_model": d, "n_layers": n, "block": None if pattern else b,
                    "pattern": {"kind": "list", "items": pattern, "span": span} if pattern
                    else None,
                    "final_norm": final, "tie_weights": tie, "z_loss": z}.items()
                   if v is not None}},
            literal(st.sampled_from([32, 64, 128])), literal(st.integers(1, 12)), layers,
            st.lists(layers, min_size=2, max_size=3) | st.just([]),
            st.none() | norm, st.none() | literal(st.booleans()),
            st.none() | literal(st.floats(0, 1, allow_nan=False)))
    return decoder, layer


def test_property_random_palette_graphs_round_trip():
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st

    from nanoscope.blocks.graph import strip_spans

    decoder, layer = _graph_strategy()
    bases = {name: (FIXTURES / f"{name}.py").read_text()
             for name in ("modern_like", "gpt2_like", "odd_formatting", "layer_pattern")}

    @given(st.sampled_from(sorted(bases)).flatmap(
        lambda name: decoder(parse_text(bases[name])["classes"][0]["args"], layer).map(
            lambda args: (name, args))))
    @settings(max_examples=60, deadline=None, suppress_health_check=list(HealthCheck))
    def check(case):
        name, args = case
        source = bases[name]
        graph = parse_text(source)
        graph["classes"][0]["args"] = args
        patched = emit(graph, source)
        again = parse_text(patched)
        assert strip_spans(again["classes"][0]["args"]) == strip_spans(args), patched
        assert emit(again, patched) == patched  # and a second emit changes nothing

    check()
