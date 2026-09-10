from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nanoscope.config import load_config
from nanoscope.eval.compare import build_comparison
from nanoscope.eval.config import load_corpus_config, load_eval_config, relative_path

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("marker", ["pyproject.toml", ".git"])
def test_alias_uses_config_project_from_another_cwd(tmp_path, monkeypatch, marker):
    root = tmp_path / "workspace"
    nested = root / "configs" / "study"
    nested.mkdir(parents=True)
    (root / marker).write_text("")  # .git files also support worktrees.
    monkeypatch.chdir(tmp_path)
    source = nested / "eval.yaml"
    assert relative_path("@/runs/future", source) == root / "runs/future"
    assert relative_path("@", source) == root
    assert relative_path("@/runs/seed-*/evaluations/*.json", source) == (
        root / "runs/seed-*/evaluations/*.json"
    )
    assert relative_path("../data", source) == root / "configs/data"
    assert relative_path(str(tmp_path / "absolute"), source) == tmp_path / "absolute"


def test_alias_requires_a_project_and_valid_prefix(tmp_path, monkeypatch):
    # The machine's /tmp or filesystem root may itself contain a project marker.
    with monkeypatch.context() as isolated:
        isolated.setattr(Path, "is_file", lambda self: False)
        isolated.setattr(Path, "exists", lambda self: False)
        with pytest.raises(ValueError, match="no parent contains"):
            relative_path("@/runs", tmp_path / "config.yaml")
    for value in ("@runs", "@//runs"):
        with pytest.raises(ValueError, match="must use"):
            relative_path(value, tmp_path / "config.yaml")


def test_nearest_project_marker_wins(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / ".git").mkdir()
    assert relative_path("@/runs", nested / "configs/eval.yaml") == nested / "runs"


def test_training_alias_is_relocatable_without_changing_digest(tmp_path, monkeypatch):
    original = load_config(ROOT / "configs/test/m0/local-smoke.yaml")
    raw = original.to_dict()
    raw["run"]["output_dir"] = "@/runs"
    configs = []
    for name in ("first", "relocated"):
        root = tmp_path / name
        root.mkdir()
        (root / "pyproject.toml").write_text("")
        path = root / "training.yaml"
        path.write_text(yaml.safe_dump(raw))
        monkeypatch.chdir(tmp_path)
        config = load_config(path)
        assert config.run.output_dir == str(root / "runs")
        assert config.digest == original.digest
        configs.append(config)
    assert configs[0].digest == configs[1].digest
    raw["run"]["output_dir"] = "relative-runs"
    path.write_text(yaml.safe_dump(raw))
    assert load_config(path).run.output_dir == "relative-runs"


def test_templates_and_moved_eval_config_resolve_to_same_paths(tmp_path, monkeypatch):
    corpus = load_corpus_config(ROOT / "configs/eval-corpus-template.yaml")
    evaluation = load_eval_config(ROOT / "configs/eval-template.yaml")
    assert corpus.output == evaluation.corpus == ROOT / "runs/eval-corpora/my-validation"
    exported = tmp_path / "exported"
    source = exported / "configs/deeper/eval.yaml"
    source.parent.mkdir(parents=True)
    (exported / "pyproject.toml").write_text("")
    source.write_text((ROOT / "configs/eval-template.yaml").read_text())
    monkeypatch.chdir(tmp_path)
    moved = load_eval_config(source)
    assert moved.corpus == exported / "runs/eval-corpora/my-validation"
    assert moved.output == exported / "runs/evaluations"


def test_study_alias_globs_reach_result_files(tmp_path):
    # A corrupt result proves the study expanded the glob and reached the artifact loader.
    (tmp_path / "pyproject.toml").write_text("")
    results = tmp_path / "runs/base-seed-1337/evaluations"
    results.mkdir(parents=True)
    (results / "result.json").write_text("{}")
    study = tmp_path / "configs/deep/study.yaml"
    study.parent.mkdir(parents=True)
    study.write_text(
        yaml.safe_dump(
            {
                "name": "paths",
                "baseline": "base",
                "axis": "tokens",
                "budget": 32,
                "variants": [
                    {"name": "base", "results": ["@/runs/base-seed-*/evaluations/*.json"]},
                    {
                        "name": "increment",
                        "results": ["@/runs/increment-seed-*/evaluations/*.json"],
                    },
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="incomplete evaluation result"):
        build_comparison(study)
