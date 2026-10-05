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
