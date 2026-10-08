import json
from statistics import mean

import pytest
from fakes import tiny

from nanoscope import RunGroup, compare, run
from nanoscope.compare import export_baseline, load_runs
from nanoscope.models import GPT2, Bigram, Modern

pytestmark = pytest.mark.usefixtures("fake_data")
SMALL = dict(d_model=16, n_layers=1, n_heads=2)


def test_seeds_trains_one_run_per_seed():
    group = run(Bigram, tiny(), device="cpu", seeds=3)

    assert isinstance(group, RunGroup)
    assert group.seeds == [0, 1, 2]
    assert [r.run_dir.name for r in group] == ["seed-0", "seed-1", "seed-2"]
    assert len({r.run_dir.parent for r in group}) == 1
    assert len({r.metrics[-1]["loss"] for r in group}) == 3
    assert group.summary()["val_bpb"]["ci95_low"] is not None


def test_compare_reports_paired_difference_with_ci():
    modern = run(Modern, tiny(), device="cpu", seeds=3, n_kv_heads=1, **SMALL)
    gpt2 = run(GPT2, tiny(), device="cpu", seeds=3, **SMALL)

    result = compare(modern, gpt2)
    text = str(result)

    assert result.baseline == "GPT2"
    modern_row, gpt2_row = result.rows
    assert gpt2_row["delta"] is None
    d = modern_row["delta"]
    assert d["paired"] and d["n"] == 3
    expected = mean(m.summary()["final_val_bpb"] - g.summary()["final_val_bpb"]
                    for m, g in zip(modern, gpt2, strict=True))
    assert d["mean"] == pytest.approx(expected)
    assert d["ci95_low"] < d["mean"] < d["ci95_high"]
    assert "bits per byte" in text and "Δ vs GPT2" in text
    assert any(v in text for v in ("better", "worse", "within noise"))


def test_single_seeds_compare_without_a_ci():
    a = run(Modern, tiny(), device="cpu", **SMALL)
    b = run(GPT2, tiny(), device="cpu", **SMALL)
    assert "need 3+ seeds" in str(compare(a, b))


def test_labels_show_what_differs():
    with_rope = run(Modern, tiny(max_steps=2), device="cpu", **SMALL)
    without = run(Modern, tiny(max_steps=2), device="cpu", rope=False, **SMALL)
    labels = [row["label"] for row in compare(without, with_rope).rows]
    assert labels == ["Modern(rope=False)", "Modern(rope=True)"]


def test_runs_can_be_named_by_folder_name():
    a = run(Bigram, tiny(max_steps=2), device="cpu")
    b = run(Bigram, tiny(max_steps=2), device="cpu", d_model=16)
    by_name = compare(b.run_dir.parent.name, "bigram", preset=a.run_dir.parent.parent.name)
    assert [row["source"] for row in by_name.rows] == [
        str(b.run_dir.parent), str(a.run_dir.parent)]
    with pytest.raises(FileNotFoundError, match="available: bigram"):
        compare(b, "nope")


def test_different_eval_text_is_refused():
    a = run(Bigram, tiny(max_steps=2), device="cpu")
    b = run(Bigram, tiny(max_steps=2, eval_docs=5), device="cpu")
    with pytest.raises(ValueError, match="different text"):
        compare(a, b)


def test_loss_across_tokenizers_is_refused_but_bpb_works():
    a = run(Bigram, tiny(max_steps=2), device="cpu")
    b = run(Bigram, tiny(max_steps=2, tokenizer="bytes", vocab_size=None), device="cpu")
    with pytest.raises(ValueError, match="use val_bpb"):
        compare(a, b, metric="val_loss")
    assert compare(a, b).metric == "val_bpb"


def test_different_budgets_get_a_note():
    a = run(Bigram, tiny(max_steps=4), device="cpu")
    b = run(Bigram, tiny(max_steps=2), device="cpu")
    assert any("different numbers of tokens" in n for n in compare(a, b).notes)


def test_exported_baseline_compares_like_the_original(tmp_path):
    group = run(Bigram, tiny(), device="cpu", seeds=2)
    out = export_baseline(group, tmp_path / "baselines", every=5)

    assert out == tmp_path / "baselines" / group[0].run_dir.parent.parent.name / "bigram"
    lines = (out / "seed-0" / "metrics.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    assert [r["step"] for r in rows] == [5, 10, 15, 20]
    assert not any("sample" in r for r in rows)
    assert [r.final("val_bpb") for r in load_runs(out)] == \
        [r.summary()["final_val_bpb"] for r in group]


def test_compare_accepts_refs_paths_and_names(capsys):
    a = run(Bigram, tiny(), device="cpu", seeds=2, progress=False)
    b = run(Bigram, tiny(), device="cpu", seeds=2, progress=False, d_model=8)
    by_ref = compare(a.ref, b.ref)
    by_path = compare(str(a[0].run_dir.parent), str(b[0].run_dir.parent))
    by_object = compare(a, b)
    def rows(c):  # the source column is the ref, the path, or the run folder
        return [{k: v for k, v in r.items() if k != "source"} for r in c.rows]

    assert rows(by_ref) == rows(by_path) == rows(by_object)
    assert [s.source for s in by_ref.sets] == [a.ref, b.ref]
    with pytest.raises(ValueError, match="can't find runs named 'no/such/set'"):
        compare(a.ref, "no/such/set")


def test_every_row_has_a_verdict_and_it_is_printed(capsys):
    few = run(Bigram, tiny(), device="cpu", seeds=2, progress=False)
    small = run(Bigram, tiny(), device="cpu", seeds=2, progress=False, d_model=8)
    rows = compare(few, small).rows  # the last one is the baseline
    assert [r["verdict"] for r in rows] == ["no CI", "baseline"]
    assert "no CI: need 3+ seeds each" in str(compare(few, small))

    many_a = run(Bigram, tiny(), device="cpu", seeds=3, progress=False, d_model=4)
    many_b = run(Bigram, tiny(), device="cpu", seeds=3, progress=False, d_model=48)
    many_c = run(Bigram, tiny(), device="cpu", seeds=3, progress=False, d_model=32)
    rows = compare(many_a, many_b, many_c)
    text = str(rows)
    for row in rows.rows:
        assert row["verdict"] in {"better", "worse", "within noise", "baseline"}
        if row["verdict"] != "baseline":
            assert row["verdict"] in text
    assert rows.rows[-1]["verdict"] == "baseline"


def test_verdict_of_follows_the_confidence_interval():
    from nanoscope.compare import verdict_of

    ci = {"ci95_low": -0.2, "ci95_high": -0.1}
    assert verdict_of(None) == "baseline"
    assert verdict_of({"ci95_low": None, "ci95_high": None}) == "no CI"
    assert verdict_of(ci) == "better"
    assert verdict_of({"ci95_low": 0.1, "ci95_high": 0.3}) == "worse"
    assert verdict_of({"ci95_low": -0.1, "ci95_high": 0.3}) == "within noise"


def test_to_dict_curves(fake_data):
    """The comparison as data carries every seed's curve and the precision plan, so a client
    can draw the figure and say how precise it is without calling anything else."""
    import json

    from fakes import tiny
    from helpers import assert_valid

    from nanoscope import compare, run
    from nanoscope.models import Bigram

    a = run(Bigram, tiny(), device="cpu", seeds=3, progress=False)
    b = run(Bigram, tiny(), device="cpu", seeds=3, progress=False, d_model=8)
    doc = compare(a, b).to_dict()
    assert_valid("comparison", doc)
    json.dumps(doc)
    assert [c["label"] for c in doc["curves"]] == [r["label"] for r in doc["rows"]]
    first = doc["curves"][0]
    assert [s["seed"] for s in first["seeds"]] == [0, 1, 2]
    points = first["seeds"][0]["points"]
    assert points[0][0] == 4 * 32 * 10  # tokens at the first eval step (batch 4 x context 32)
    assert [x for x, _ in points] == sorted(x for x, _ in points) and len(points) == 2
    plan = doc["precision_plan"]
    assert (plan["metric"], plan["preset"], plan["n_seeds"]) == ("val_bpb", "test-tiny", 3)
    assert plan["half_width"] is None and "no shipped baselines" in plan["note"]


def test_params_kind_says_which_count_the_column_holds(tmp_path):
    import shutil

    own = run(Bigram, tiny(), device="cpu", seeds=3, progress=False)
    # the shipped v0 baselines only record the total: make a copy of ours that does the same
    old = tmp_path / "old"
    shutil.copytree(own[0].run_dir.parent, old)
    for config in old.glob("seed-*/config.json"):
        doc = json.loads(config.read_text())
        del doc["stats"]["n_non_embedding_params"]
        config.write_text(json.dumps(doc))

    mixed = compare(own, old)
    assert [r["params_kind"] for r in mixed.rows] == ["non-embedding", "total"]
    header = str(mixed).splitlines()[2]
    assert "params" in header and "non-emb params" not in header and "total params" not in header
    assert "(total)" in str(mixed) and "(non-emb)" in str(mixed)

    plain = compare(own, run(Bigram, tiny(), device="cpu", seeds=3, d_model=48, progress=False))
    assert {r["params_kind"] for r in plain.rows} == {"non-embedding"}
    assert "non-emb params" in str(plain).splitlines()[2]

    # two shipped baselines are both totals, and the header says so
    both = compare("bigram", "gpt2", preset="tinystories-5min")
    assert {r["params_kind"] for r in both.rows} == {"total"}
    assert "total params" in str(both).splitlines()[2]


def test_rows_carry_the_text_the_table_prints(capsys):
    result = compare("bigram", "modern", "gpt2", preset="tinystories-5min")
    doc = result.to_dict()
    assert doc["title"].startswith("bits per byte on the first 200 validation documents of ")
    assert doc["params_header"] == "total params"
    bigram, modern, gpt2 = doc["rows"]
    assert modern["text"]["delta"].startswith("−0.16") and modern["text"]["verdict"] == "better"
    assert modern["text"]["statement"].startswith("Modern vs GPT2: −0.16")
    assert modern["text"]["statement"].endswith("bpb, 3 seeds each, paired")
    assert gpt2["text"]["delta"] == "(baseline)" and gpt2["text"]["statement"] is None
    # the printed table is made of exactly these cells
    printed = str(result)
    for row in doc["rows"]:
        assert row["text"]["value"] in printed and row["text"]["delta"] in printed
    assert "With 3 seeds per model, a difference on this preset is known to about ±0.0" in (
        doc["precision_plan"]["text"])
