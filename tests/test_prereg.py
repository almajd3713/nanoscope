import subprocess
import textwrap

import pytest

from nanoscope.prereg import PreregError, commit, preview
from nanoscope.study import load_study

SPEC = textwrap.dedent("""
    from nanoscope import Study, Tokens
    from nanoscope.models import Bigram
    from nanoscope.presets import get_preset

    study = Study("pre", preset=get_preset("tinystories-5min"), seeds=3, budget=Tokens(1e5),
                  baseline="small", mode="record")
    study.add("small", Bigram, d_model=8)
    study.add("wide", Bigram, d_model=32)
    study.predict("wide", val_bpb={prediction})
""")


def git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True,
                          text=True).stdout


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "test")
    (tmp_path / "README.md").write_text("hello")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "init")
    return tmp_path


@pytest.fixture
def home(tmp_path_factory, monkeypatch):
    """A NANOSCOPE_HOME outside the repository, as in a real workspace (the default home would
    put queue.db in the repo as an unrelated file)."""
    folder = tmp_path_factory.mktemp("nanoscope-home")
    monkeypatch.setenv("NANOSCOPE_HOME", str(folder))
    monkeypatch.delenv("NANOSCOPE_WORKSPACE", raising=False)
    return folder


def write_spec(repo, prediction=1.5):
    path = repo / "study.py"
    path.write_text(SPEC.format(prediction=prediction))
    return path


def test_preview_shows_the_new_file_and_changes_nothing(repo):
    path = write_spec(repo)
    before = git(repo, "status", "--porcelain")
    p = preview(load_study(path))

    assert p.files == ["study.py"] and p.unrelated == [] and p.identity
    assert p.diff.startswith("diff --git") and "+study.predict" in p.diff
    assert "Preregister study pre" in p.message and "wide val_bpb=1.5" in p.message
    assert p.nothing_to_commit is False and len(p.hash) == 16
    assert git(repo, "status", "--porcelain") == before  # not even staged


def test_preview_diffs_a_tracked_edit_against_head(repo):
    path = write_spec(repo)
    git(repo, "add", "."), git(repo, "commit", "-qm", "spec")
    assert preview(load_study(path)).nothing_to_commit

    write_spec(repo, prediction=1.2)
    p = preview(load_study(path))
    assert "-study.predict" in p.diff and "+study.predict" in p.diff and not p.nothing_to_commit


def test_preview_lists_unrelated_files_and_the_hash_follows_the_content(repo):
    path = write_spec(repo)
    first = preview(load_study(path))
    (repo / "scratch.txt").write_text("x")
    (repo / "README.md").write_text("changed")
    p = preview(load_study(path))
    assert p.unrelated == ["README.md", "scratch.txt"] and p.hash != first.hash
    assert preview(load_study(path)).hash == p.hash  # stable while nothing changes


def test_preview_outside_a_repository_says_so(tmp_path):
    path = tmp_path / "study.py"
    path.write_text(SPEC.format(prediction=1.5))
    with pytest.raises(PreregError, match="not inside a git repository"):
        preview(load_study(path))


def run_job(kind, payload):
    from nanoscope import queue
    from nanoscope.cli import main

    job_id = queue.enqueue(kind, payload, lane="interactive")
    queue.claim("w", "cpu")
    with pytest.raises(SystemExit) as stopped:
        main(["run-job", str(job_id)])
    row = queue.get(job_id)
    return stopped.value.code, row


def test_commit_makes_one_commit_with_only_the_spec(repo, home):
    import json

    path = write_spec(repo)
    p = preview(load_study(path))
    code, row = run_job("commit", {"spec": str(path), "preview_hash": p.hash})

    assert code == 0, row["error"]
    sha = json.loads(row["result"])["commit"]
    assert git(repo, "rev-parse", "HEAD").strip() == sha
    assert git(repo, "show", "--name-only", "--format=%s", "HEAD").split() == [
        "Preregister", "study", "pre", "study.py"]
    assert git(repo, "status", "--porcelain") == ""


def test_commit_refuses_a_changed_preview(repo, home):
    path = write_spec(repo)
    p = preview(load_study(path))
    write_spec(repo, prediction=1.1)
    code, row = run_job("commit", {"spec": str(path), "preview_hash": p.hash})
    assert code == 1 and "preview changed" in row["error"]
    assert git(repo, "log", "--format=%s").strip() == "init"


def test_prereg_preview_is_a_job_too(repo, home):
    import json

    path = write_spec(repo)
    code, row = run_job("prereg-preview", {"spec": str(path)})
    assert code == 0 and json.loads(row["result"])["hash"] == preview(load_study(path)).hash


@pytest.mark.usefixtures("fake_data")
def test_preregistration_is_recorded_in_report(repo):
    import json

    from test_study import PRESET_SRC

    path = repo / "study.py"
    path.write_text(PRESET_SRC + textwrap.dedent("""
        study = Study("toy", preset=preset, seeds=3, budget=Tokens(4 * 32 * 6),
                      baseline="small", mode="record")
        study.add("small", Bigram, d_model=8)
        study.add("wide", Bigram, d_model=32)
        study.predict("wide", val_bpb=1.5)
    """))
    study = load_study(path)
    sha = commit(study, preview(study).hash)
    study.run(devices=["cpu"])

    manifest = json.loads((study.dir / "study.json").read_text())
    assert manifest["preregistration_commit"] == sha and manifest["committed_via"] == "nanoscope"
    config = json.loads((study.dir / "wide" / "seed-0" / "config.json").read_text())
    assert config["study"]["preregistration_commit"] == sha
    assert config["study"]["committed_via"] == "nanoscope"
    assert "preregistration committed via nanoscope" in str(study.report(write=False))


def test_commit_refuses_unrelated_dirty_files_and_lists_them(repo, home):
    path = write_spec(repo)
    (repo / "notes.txt").write_text("scratch")
    study = load_study(path)
    p = preview(study)
    with pytest.raises(PreregError, match="notes.txt"):
        commit(study, p.hash)
    assert git(repo, "log", "--format=%s").strip() == "init"
    assert not git(repo, "diff", "--cached", "--name-only")  # nothing was staged either


def test_commit_refuses_without_a_git_identity(repo, home, monkeypatch, tmp_path):
    git(repo, "config", "--unset", "user.email")
    git(repo, "config", "--unset", "user.name")
    monkeypatch.setenv("HOME", str(tmp_path / "nohome"))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
    study = load_study(write_spec(repo))
    p = preview(study)
    assert p.identity is False
    with pytest.raises(PreregError, match="no identity"):
        commit(study, p.hash)


def test_commit_refuses_when_nothing_changed(repo, home):
    path = write_spec(repo)
    git(repo, "add", "."), git(repo, "commit", "-qm", "spec")
    study = load_study(path)
    with pytest.raises(PreregError, match="nothing to commit"):
        commit(study, preview(study).hash)


def test_git_hooks_run_in_the_worker(repo, home, monkeypatch):
    hook = repo / ".git" / "hooks" / "pre-commit"
    marker = repo.parent / f"{repo.name}-hook-ran"
    hook.write_text(f'#!/bin/sh\necho "job=$NANOSCOPE_JOB_ID" > {marker}\n')
    hook.chmod(0o755)
    monkeypatch.setenv("NANOSCOPE_JOB_ID", "")  # execute() sets it; teardown undoes that
    monkeypatch.delenv("NANOSCOPE_JOB_ID")
    path = write_spec(repo)
    shown = preview(load_study(path))
    code, row = run_job("commit", {"spec": str(path), "preview_hash": shown.hash})
    assert code == 0, row["error"]
    # NANOSCOPE_JOB_ID is only set inside a job's execute(): the hook ran for a worker's job
    assert marker.read_text().strip() == f"job={row['id']}"


def test_the_api_never_imports_prereg():
    from pathlib import Path

    server = Path(__file__).parent.parent / "nanoscope" / "server"
    for file in server.rglob("*.py"):
        assert "nanoscope.prereg" not in file.read_text(encoding="utf-8"), file
        assert "import prereg" not in file.read_text(encoding="utf-8"), file


def test_cli_shows_diff_before_committing(repo, home, capsys, monkeypatch):
    from nanoscope.cli import main

    path = write_spec(repo)
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    main(["study", "preregister", str(path)])
    out = capsys.readouterr().out
    assert "+study.predict" in out and "Preregister study pre" in out and "not committed" in out
    assert git(repo, "log", "--format=%s").strip() == "init"

    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    main(["study", "preregister", str(path)])
    assert "committed " in capsys.readouterr().out
    assert git(repo, "log", "-1", "--format=%s").strip() == "Preregister study pre"


def test_cli_yes_commits_and_refusals_exit_with_the_reason(repo, home, capsys):
    from nanoscope.cli import main

    path = write_spec(repo)
    (repo / "notes.txt").write_text("x")
    with pytest.raises(SystemExit, match="notes.txt"):
        main(["study", "preregister", str(path), "--yes"])
    assert "other files have uncommitted changes" in capsys.readouterr().out
