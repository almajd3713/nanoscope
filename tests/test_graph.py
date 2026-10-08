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


def test_cli_prints_the_graph_without_running_anything(capsys, tmp_path):
    import json

    from helpers import assert_valid

    from nanoscope.cli import main

    main(["graph", str(FIXTURES / "modern_like.py")])
    text = capsys.readouterr().out
    assert text.splitlines()[0].startswith("MyModern(Decoder)")
    assert "  vocab_size = <vocab_size>" in text
    assert "  d_model = 128" in text
    assert "    attn = Attention" in text and "      n_kv_heads = 2" in text

    main(["graph", f"{FIXTURES / 'code_only.py'}:Dynamic"])
    text = capsys.readouterr().out
    assert "Fine" not in text and "code-only, line 12: __init__ has an assignment" in text

    main(["graph", str(FIXTURES / "composite_template.py"), "--json"])
    graph = json.loads(capsys.readouterr().out)
    assert_valid("graph", graph)
    assert [c["name"] for c in graph["classes"]] == ["PreNormAttention", "UsesTemplate"]

    boom = tmp_path / "boom.py"
    boom.write_text("raise RuntimeError('imported')\n")
    main(["graph", str(boom)])  # prints nothing and does not import the file
    assert capsys.readouterr().out.strip() == ""
    with pytest.raises(SystemExit, match="no Decoder or Composite class 'Nope'"):
        main(["graph", f"{FIXTURES / 'modern_like.py'}:Nope"])


def test_template_slot():
    """Filling a template slot is an ordinary replace_block on that slot."""
    path = FIXTURES / "template_fill.py"
    source = path.read_text()
    graph = parse(path)
    cls = graph["classes"][0]
    assert cls["representable"]
    block = cls["args"]["block"]
    assert (block["kind"], block["block"], block["family"], block["tier"]) == (
        "block", "BlockTemplate", "template", "primitive")
    assert list(block["args"]) == ["norm1", "attn", "norm2", "mlp"]
    attn = block["args"]["attn"]
    assert attn["block"] == "AttentionTemplate" and list(attn["args"]) == [
        "q", "k", "v", "scores", "mask", "normalize", "mix", "out"]
    assert attn["args"]["mask"] == {"kind": "literal", "value": None,
                                    "span": attn["args"]["mask"]["span"]}  # the empty slot
    # drag a primitive into the slot: one replace_block, one changed line
    fill = {"kind": "block", "block": "CausalMask", "args": {}, "span": None}
    attn["args"]["mask"] = fill
    edits = diff(parse(path), graph)
    assert edits == [{"op": "replace_block", "class": "FromPrimitives",
                      "path": ["block", "attn", "mask"], "node": fill}]
    after = emit(graph, source)
    assert changed_lines(source, after) == [(
        13, "                    scores=ScaledDotScores(), mask=None, normalize=Softmax(),",
        "                    scores=ScaledDotScores(), mask=CausalMask(), normalize=Softmax(),")]
    again = parse_text(after)["classes"][0]["args"]["block"]["args"]["attn"]["args"]["mask"]
    assert (again["kind"], again["block"]) == ("block", "CausalMask")
    # swapping a filled slot works the same way
    graph = parse_text(after)
    graph["classes"][0]["args"]["block"]["args"]["attn"]["args"]["normalize"] = {
        "kind": "block", "block": "Softmax", "args": {}, "span": None}
    assert emit(graph, after) == after  # same block, nothing to change


def test_structural_edits_n_layers_and_pattern_items():
    source = (FIXTURES / "layer_pattern.py").read_text()
    # n_layers +/- 1: one changed line
    more = apply_edits(source, [{"op": "add_layer", "class": "SlidingGlobal"}])
    ((line, old, new),) = changed_lines(source, more)
    assert line == 11 and new == old.replace("n_layers=6", "n_layers=7")
    fewer = apply_edits(more, [{"op": "remove_layer", "class": "SlidingGlobal"}])
    assert fewer == source
    with pytest.raises(ValueError, match="at least one layer"):
        apply_edits(source.replace("n_layers=6", "n_layers=1"),
                    [{"op": "remove_layer", "class": "SlidingGlobal"}])

    # a pattern item: insert and remove keep the multi-line list and its trailing comma
    node = {"kind": "block", "block": "Block", "span": None, "args": {
        "norm": {"kind": "block", "block": "RMSNorm", "args": {}, "span": None},
        "attn": {"kind": "block", "block": "Attention", "args": {"n_heads": lit(2)},
                 "span": None},
        "mlp": {"kind": "block", "block": "SwiGLU", "args": {}, "span": None}}}
    grown = apply_edits(source, [{"op": "add_layer", "class": "SlidingGlobal",
                                  "path": ["pattern"], "node": node}])
    items = parse_text(grown)["classes"][0]["args"]["pattern"]["items"]
    assert len(items) == 3 and items[2]["args"]["attn"]["args"]["n_heads"]["value"] == 2
    assert "nb.Block(norm=nb.RMSNorm(), attn=nb.Attention(n_heads=2), mlp=nb.SwiGLU())" in grown
    assert grown.count("\n") == source.count("\n") + 1  # one new line
    front = apply_edits(source, [{"op": "add_layer", "class": "SlidingGlobal",
                                  "path": ["pattern"], "node": node, "index": 0}])
    assert parse_text(front)["classes"][0]["args"]["pattern"]["items"][0]["args"]["attn"][
        "args"]["n_heads"]["value"] == 2
    back = apply_edits(grown, [{"op": "remove_layer", "class": "SlidingGlobal",
                                "path": ["pattern"], "index": 2}])
    assert back == source
    with pytest.raises(ValueError, match="would be empty"):
        one = apply_edits(source, [{"op": "remove_layer", "class": "SlidingGlobal",
                                    "path": ["pattern"], "index": 0}])
        apply_edits(one, [{"op": "remove_layer", "class": "SlidingGlobal",
                           "path": ["pattern"], "index": 0}])
    with pytest.raises(ValueError, match="no item 5"):
        apply_edits(source, [{"op": "remove_layer", "class": "SlidingGlobal",
                              "path": ["pattern"], "index": 5}])


def test_structural_edits_come_out_of_emit_as_minimal_edits():
    path = FIXTURES / "layer_pattern.py"
    source = path.read_text()
    graph = parse(path)
    pattern = graph["classes"][0]["args"]["pattern"]
    pattern["items"].append(pattern["items"][0])
    assert diff(parse(path), graph) == [{
        "op": "add_layer", "class": "SlidingGlobal", "path": ["pattern"], "index": 2,
        "node": pattern["items"][0]}]
    after = emit(graph, source)
    assert len(parse_text(after)["classes"][0]["args"]["pattern"]["items"]) == 3
    graph = parse(path)
    del graph["classes"][0]["args"]["pattern"]["items"][0]
    assert diff(parse(path), graph) == [{
        "op": "remove_layer", "class": "SlidingGlobal", "path": ["pattern"], "index": 0}]
    assert len(parse_text(emit(graph, source))["classes"][0]["args"]["pattern"]["items"]) == 1


def test_structural_edits_set_pattern_replaces_block():
    source = (FIXTURES / "modern_like.py").read_text()
    layer = parse_text(source)["classes"][0]["args"]["block"]
    patched = apply_edits(source, [{"op": "set_pattern", "class": "MyModern",
                                    "items": [layer, layer]}])
    args = parse_text(patched)["classes"][0]["args"]
    assert "block" not in args and len(args["pattern"]["items"]) == 2
    with pytest.raises(ValueError, match="at least one block"):
        apply_edits(source, [{"op": "set_pattern", "class": "MyModern", "items": []}])
    # a pattern that already exists is replaced in place
    pat = (FIXTURES / "layer_pattern.py").read_text()
    new = apply_edits(pat, [{"op": "set_pattern", "class": "SlidingGlobal", "items": [
        {"kind": "block", "block": "Block", "span": None, "args": {
            k: {"kind": "block", "block": b, "args": {}, "span": None}
            for k, b in (("norm", "RMSNorm"), ("attn", "Attention"), ("mlp", "SwiGLU"))}}]}])
    assert len(parse_text(new)["classes"][0]["args"]["pattern"]["items"]) == 1


def test_structural_edits_fill_slot():
    path = FIXTURES / "template_fill.py"
    source = path.read_text()
    fill = {"kind": "block", "block": "CausalMask", "args": {}, "span": None}
    edit = {"op": "fill_slot", "class": "FromPrimitives", "path": ["block", "attn"],
            "slot": "mask", "node": fill}
    after = apply_edits(source, [edit])
    assert changed_lines(source, after) == [(
        13, "                    scores=ScaledDotScores(), mask=None, normalize=Softmax(),",
        "                    scores=ScaledDotScores(), mask=CausalMask(), normalize=Softmax(),")]
    assert apply_edits(after, [{**edit, "node": None}]) == source  # emptying the slot
    with pytest.raises(ValueError, match="has the slots .*not 'nope'"):
        apply_edits(source, [{**edit, "slot": "nope"}])
    with pytest.raises(ValueError, match="not a template"):
        apply_edits(source, [{**edit, "path": ["block", "attn", "q"]}])
    local = (FIXTURES / "composite_template.py").read_text()
    out = apply_edits(local, [{"op": "fill_slot", "class": "UsesTemplate", "path": ["block"],
                               "slot": "norm", "node": {"kind": "block", "block": "LayerNorm",
                                                        "args": {}, "span": None}}])
    assert "norm=LayerNorm()" in out


def test_property_all_ops():
    """Random sequences of every edit keep the file parseable, and the graph and file agree."""
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st

    from nanoscope.blocks.graph import strip_spans

    def block(name, **args):
        return {"kind": "block", "block": name, "args": args, "span": None}

    layer = block("Block", norm=block("RMSNorm"), attn=block("Attention", n_heads=lit(2)),
                  mlp=block("SwiGLU"))
    bases = {name: (FIXTURES / f"{name}.py").read_text()
             for name in ("modern_like", "layer_pattern", "odd_formatting")}
    op = st.one_of(
        st.tuples(st.just("layers"), st.sampled_from(["add_layer", "remove_layer"])),
        st.tuples(st.just("item"), st.sampled_from(["add_layer", "remove_layer"]),
                  st.integers(0, 3)),
        st.tuples(st.just("pattern"), st.integers(1, 3)),
        st.tuples(st.just("set"), st.sampled_from(["d_model", "z_loss"]), st.integers(1, 9)))

    @given(st.sampled_from(sorted(bases)), st.lists(op, max_size=8))
    @settings(max_examples=60, deadline=None, suppress_health_check=list(HealthCheck))
    def check(name, ops):
        source = bases[name]
        cls = parse_text(source)["classes"][0]["name"]
        for step in ops:
            args = parse_text(source)["classes"][0]["args"]
            if step[0] == "layers":
                n = args["n_layers"]["value"]
                if step[1] == "remove_layer" and n == 1:
                    continue
                edit = {"op": step[1], "class": cls}
            elif step[0] == "item":
                if "pattern" not in args:
                    continue
                n = len(args["pattern"]["items"])
                if step[1] == "remove_layer":
                    if n == 1:
                        continue
                    edit = {"op": step[1], "class": cls, "path": ["pattern"],
                            "index": step[2] % n}
                else:
                    edit = {"op": step[1], "class": cls, "path": ["pattern"],
                            "node": layer, "index": step[2] % (n + 1)}
            elif step[0] == "pattern":
                edit = {"op": "set_pattern", "class": cls, "items": [layer] * step[1]}
            else:
                edit = {"op": "set_arg", "class": cls, "path": [], "arg": step[1],
                        "value": lit(step[2])}
            source = apply_edits(source, [edit])
            again = parse_text(source)
            assert again["classes"][0]["representable"], source
            graph = parse_text(source)
            assert strip_spans(parse_text(emit(graph, source))) == strip_spans(graph)
            assert emit(graph, source) == source  # and emitting what was parsed changes nothing

    check()
