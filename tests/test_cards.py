import json

import pytest
from helpers import assert_valid
from test_study import commit_all, git, write_study

from nanoscope.cards import (
    compare_cards,
    export_card,
    format_comparison,
    push,
    push_plan,
    read_card,
    write_card,
)
from nanoscope.study import load_study

pytestmark = pytest.mark.usefixtures("fake_data")


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "test")
    (tmp_path / ".gitignore").write_text("runs/\ncache/\nexperiments/\nhome/\n")
    git(tmp_path, "add", ".gitignore")
    git(tmp_path, "commit", "-qm", "init")
    return tmp_path


@pytest.fixture
def finished(repo):
    path = repo / "study.py"
    write_study(path, mode="record")
    commit_all(repo)
    study = load_study(path)
    study.run(devices=["cpu"])
    return study


def test_export_card_holds_spec_finals_and_provenance(finished):
    card = export_card(finished)

    assert_valid("card", card)
    assert card["mode"] == "record" and card["study"] == "toy"
    assert sorted(card["finals"]) == ["small", "wide"]
    assert sorted(card["finals"]["wide"]) == ["0", "1", "2"]
    assert card["spec"]["variants"][0]["name"] == "small"
    assert card["provenance"]["commit"] and card["eval"]["dataset"] == "fake/stories"
    path = write_card(card, finished.dir.parent.parent / "card.json")
    assert read_card(path)["finals"] == card["finals"]


def test_export_refuses_explore_mode_and_unfinished_studies(repo):
    path = repo / "study.py"
    write_study(path, mode="explore")
    with pytest.raises(ValueError, match="record-mode studies only"):
        export_card(load_study(path))

    write_study(path, mode="record")
    commit_all(repo)
    with pytest.raises(ValueError, match="has not started"):
        export_card(load_study(path))


def test_export_refuses_a_study_with_missing_seeds(finished):
    import shutil

    shutil.rmtree(finished.dir / "wide" / "seed-2")
    with pytest.raises(ValueError, match="not finished.*wide seed 2"):
        export_card(finished)


class FakeApi:
    uploads: list[dict] = []

    def upload_file(self, **kw):
        FakeApi.uploads.append(kw)


def test_push_uploads_one_file_and_the_plan_shows_it_first(finished, monkeypatch):
    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    FakeApi.uploads = []
    card = export_card(finished)

    plan = push_plan(card, "me/ablation-cards")
    assert FakeApi.uploads == []  # planning uploads nothing
    assert plan["path_in_repo"] == "cards/toy.json" and json.loads(plan["content"]) == card

    result = push(card, "me/ablation-cards")
    (upload,) = FakeApi.uploads
    assert upload["repo_type"] == "dataset" and upload["repo_id"] == "me/ablation-cards"
    assert json.loads(upload["path_or_fileobj"]) == card
    assert "content" not in result and result["path_in_repo"] == "cards/toy.json"


def card_with(base, **seeds):
    card = json.loads(json.dumps(base))
    card["study"] = seeds.pop("study", "other")
    card["finals"] = {v: {str(i): {"val_bpb": x} for i, x in enumerate(xs)}
                      for v, xs in seeds.items()}
    card["baseline"] = next(iter(seeds))
    return card


def test_compare_cards_uses_welch_intervals(finished):
    base = export_card(finished)
    mine = card_with(base, study="mine", small=[1.50, 1.51, 1.49], wide=[1.40, 1.41, 1.39])
    theirs = card_with(base, study="theirs", small=[1.52, 1.50, 1.51, 1.53])

    result = compare_cards([mine, theirs])
    rows = {r["label"]: r for r in result["rows"]}
    assert result["baseline"] == "theirs/small"
    assert rows["theirs/small"]["verdict"] == "baseline"
    wide = rows["mine/wide"]
    assert wide["delta"]["paired"] is False and wide["verdict"] == "better"
    assert wide["delta"]["ci95_high"] < 0
    assert "mine/wide" in format_comparison(result) and "unpaired" in format_comparison(result)


def test_compare_cards_needs_the_same_evaluation_text(finished):
    base = export_card(finished)
    other = card_with(base, small=[1.0, 1.0, 1.0])
    other["eval"]["eval_docs"] = 99
    with pytest.raises(ValueError, match="different text"):
        compare_cards([base, other])
    with pytest.raises(ValueError, match="at least two"):
        compare_cards([base])


def test_card_cli_export_push_and_compare(finished, tmp_path, monkeypatch, capsys):
    from nanoscope.cli import main

    out = tmp_path / "cards" / "toy.json"
    main(["card", "export", str(finished.source), "--out", str(out)])
    assert out.exists() and "wrote" in capsys.readouterr().out

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    FakeApi.uploads = []
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    main(["card", "push", str(out), "--repo", "me/cards"])
    shown = capsys.readouterr().out
    assert "cards/toy.json" in shown and '"finals"' in shown and "not uploaded" in shown
    assert FakeApi.uploads == []
    main(["card", "push", str(out), "--repo", "me/cards", "--yes"])
    assert len(FakeApi.uploads) == 1

    other = tmp_path / "cards" / "other.json"
    card = card_with(read_card(out), study="other", small=[1.6, 1.5, 1.55])
    write_card(card, other)
    main(["card", "compare", str(out), str(other)])
    assert "Δ vs other/small" in capsys.readouterr().out


def test_card_push_is_a_job_that_needs_the_token_and_is_refused_offline(
        finished, tmp_path, monkeypatch):
    from nanoscope import queue
    from nanoscope.cli import main
    from nanoscope.jobs.runner import job_env

    out = write_card(export_card(finished), tmp_path / "toy.json")
    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    FakeApi.uploads = []
    payload = {"card": str(out), "repo": "me/cards"}
    assert job_env("card-push", payload, {"HF_TOKEN": "h", "WANDB_API_KEY": "w"}) == {
        "HF_TOKEN": "h"}

    job_id = queue.enqueue("card-push", payload, lane="interactive")
    queue.claim("w", "cpu")
    with pytest.raises(SystemExit) as stopped:
        main(["run-job", str(job_id)])
    assert stopped.value.code == 0 and len(FakeApi.uploads) == 1

    monkeypatch.setenv("NANOSCOPE_JOBS_OFFLINE", "1")
    job_id = queue.enqueue("card-push", {**payload, "repo": "me/other"}, lane="interactive")
    queue.claim("w", "cpu")
    with pytest.raises(SystemExit):
        main(["run-job", str(job_id)])
    assert "uploads a card" in queue.get(job_id)["error"] and len(FakeApi.uploads) == 1
