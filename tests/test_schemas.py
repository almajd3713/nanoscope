import json

import pytest
from helpers import assert_valid

from nanoscope.schemas import CURRENT, get
from nanoscope.schemas.upgrade import read_json


def test_every_schema_loads_and_is_a_valid_json_schema():
    import jsonschema

    for name in CURRENT:
        jsonschema.Draft202012Validator.check_schema(get(name))


def test_upgrade_reads_v0_stamps_v1_and_refuses_newer(tmp_path):
    old = tmp_path / "config.json"
    old.write_text(json.dumps({"model": {"class": "Bigram", "kwargs": {}},
                               "preset": {"name": "p", "max_steps": 3},
                               "tokenizer": "byte", "seed": 0}))
    data = read_json(old, "config")
    assert data["schema"] == 1
    assert_valid("config", data)

    new = tmp_path / "newer.json"
    new.write_text(json.dumps({"schema": 2, "nanoscope": "9.9.9"}))
    with pytest.raises(ValueError, match=r"written by a newer nanoscope \(9\.9\.9\)"):
        read_json(new, "config")


def test_upgrade_leaves_current_files_alone(tmp_path):
    path = tmp_path / "plan.json"
    doc = {"schema": 1, "nanoscope": "0.2.0", "study": "s", "runs": []}
    path.write_text(json.dumps(doc))
    assert read_json(path, "plan") == doc


def test_assert_valid_names_the_broken_field():
    with pytest.raises(AssertionError, match="state"):
        assert_valid("status", {"schema": 1, "state": "paused", "updated_at": "now"})


def test_every_file_a_toy_study_writes_matches_its_schema(fake_data):
    from fakes import tiny

    from nanoscope import Study, Tokens, paths
    from nanoscope.models import Bigram

    study = Study("toy", preset=tiny(), seeds=2, budget=Tokens(4 * 32 * 6), baseline="small")
    study.add("small", Bigram, d_model=8)
    study.add("wide", Bigram, d_model=32)
    study.run(devices=["cpu"])
    study.report()

    configs = sorted(study.dir.glob("*/seed-*/config.json"))
    assert len(configs) == 4
    for path in configs:
        assert_valid("config", json.loads(path.read_text()))
    assert_valid("plan", json.loads((study.dir / "plan.json").read_text()))
    assert_valid("results", json.loads((paths.reports_dir() / "toy" / "results.json").read_text()))


def test_shipped_baselines_load_as_v0_and_compare_as_before():
    from nanoscope import compare
    from nanoscope.compare import BASELINES_DIR, load_runs

    seed_dirs = sorted(BASELINES_DIR.glob("*/*/seed-*"))
    assert len(seed_dirs) >= 9
    for seed_dir in seed_dirs:
        raw = json.loads((seed_dir / "config.json").read_text())
        assert "schema" not in raw  # shipped before schemas existed
        loaded = load_runs(seed_dir)[0]
        assert loaded.config == {**raw, "schema": 1}
        assert_valid("config", loaded.config)
    out = str(compare("baselines/tinystories-5min/gpt2", "baselines/tinystories-5min/modern"))
    assert "Modern" in out and "GPT2" in out


def test_results_expose_plain_dicts_that_match_their_schemas(fake_data):
    from fakes import tiny

    from nanoscope import Study, Tokens, compare, paths, run
    from nanoscope.models import Bigram

    group = run(Bigram, tiny(), device="cpu", seeds=3, progress=False)
    other = run(Bigram, tiny(), device="cpu", seeds=3, progress=False, d_model=8)
    single = group[0].to_dict(metrics=True)
    assert single["ref"] == group[0].ref and len(single["metrics"]) == 20
    assert group.to_dict()["ref"] == group.ref and len(group.to_dict()["runs"]) == 3
    json.dumps(group.to_dict(metrics=True))  # all plain data

    comparison = compare(group, other).to_dict()
    assert_valid("comparison", comparison)
    json.dumps(comparison)

    study = Study("toy", preset=tiny(), seeds=2, budget=Tokens(4 * 32 * 6), baseline="small")
    study.add("small", Bigram, d_model=8)
    study.add("wide", Bigram, d_model=32)
    study.run(devices=["cpu"])
    report = study.report()
    assert_valid("results", json.loads(json.dumps(report.to_dict(), default=str)))
    written = json.loads((paths.reports_dir() / "toy" / "results.json").read_text())
    assert written == json.loads(json.dumps(report.to_dict(), default=str))
