"""Lessons and paths as data: the loader, lesson.md, progress."""

import textwrap

import pytest
from helpers import assert_valid

from nanoscope.learn.loader import (
    CurriculumError,
    load_lesson,
    load_path,
    parse_lesson_md,
)

LESSON = """\
title = "Bigram"
level = 0
summary = "The smallest language model."
unlocks = ["block:Attention"]
forbid = ["torch.nn.MultiheadAttention"]

[experiment]
kind = "run"
model = "nanoscope.models:Bigram"

[[checks]]
id = "built"
kind = "defines"
class = "Bigram"

[[checks]]
id = "learns"
kind = "trains"
metric = "val_bpb"
threshold = 2.5

[depth.deep]
seeds = 3

[compute.cpu]
preset = "tinystories-5min"
estimate_minutes = 2
"""

MD = """\
Intro text.

## Surface
Train it and read the curve.

## Deep
Three seeds.

## Reading
- Karpathy, makemore
"""


def make(root, path="foundations", slug="01-bigram", toml=LESSON, md=MD, path_toml=None):
    folder = root / path
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "path.toml").write_text(path_toml or 'title = "Foundations"\nlevel = 0\n')
    lesson = folder / slug
    lesson.mkdir(exist_ok=True)
    if toml is not None:
        (lesson / "lesson.toml").write_text(toml)
    if md is not None:
        (lesson / "lesson.md").write_text(md)
    return lesson


def problems_of(call):
    with pytest.raises(CurriculumError) as caught:
        call()
    return {p.field: p.message for p in caught.value.problems}


def test_loader_reads_a_lesson_and_a_path(tmp_path):
    make(tmp_path)
    lesson = load_lesson("foundations/01-bigram", tmp_path)
    assert (lesson.id, lesson.slug, lesson.level, lesson.title) == (
        "foundations/01-bigram", "01-bigram", 0, "Bigram")
    assert lesson.unlocks == ["block:Attention"] and lesson.forbid == [
        "torch.nn.MultiheadAttention"]
    assert [(c.id, c.kind) for c in lesson.checks] == [("built", "defines"), ("learns", "trains")]
    assert lesson.checks[1].args == {"metric": "val_bpb", "threshold": 2.5}
    assert lesson.depth == {"deep": {"seeds": 3}} and lesson.experiment["kind"] == "run"
    assert lesson.compute["cpu"].preset == "tinystories-5min"
    assert lesson.estimate_minutes == 2 and not lesson.has_starter
    assert lesson.text.surface == "Train it and read the curve."
    path = load_path("foundations", tmp_path)
    assert path.title == "Foundations" and [x.id for x in path.lessons] == ["foundations/01-bigram"]


def test_loader_reports_every_problem_at_once(tmp_path):
    bad = textwrap.dedent("""\
        level = 7
        unlocks = ["Attention"]
        surprise = true
        [experiment]
        kind = "dance"
        [[checks]]
        id = "a"
        kind = "trains"
        [[checks]]
        id = "a"
        kind = "mystery"
        [compute.cpu]
        estimate_minutes = -1
        """)
    make(tmp_path, toml=bad, md="no sections here")
    found = problems_of(lambda: load_lesson("foundations/01-bigram", tmp_path))
    prefix = "foundations/01-bigram/lesson.toml: "
    assert found[prefix + "title"] == "is required"
    assert "must be one of 0, 1, 2, 3" in found[prefix + "level"]
    assert "must look like block:Attention" in found[prefix + "unlocks[0]"]
    assert "not a known field" in found[prefix + "surprise"]
    assert "must be one of run, study, check" in found[prefix + "experiment.kind"]
    assert "required for a trains check" in found[prefix + "checks[0].metric"]
    assert "required for a trains check" in found[prefix + "checks[0].threshold"]
    assert "must be one of defines" in found[prefix + "checks[1].kind"]
    assert "needs exactly one of preset or budget" in found[prefix + "compute.cpu"]
    assert "above 0" in found[prefix + "compute.cpu.estimate_minutes"]
    assert found["foundations/01-bigram/lesson.md"].startswith("needs a '## Surface'")


def test_loader_reports_missing_and_broken_files(tmp_path):
    make(tmp_path, toml=None, md=None)
    found = problems_of(lambda: load_lesson("foundations/01-bigram", tmp_path))
    assert found["foundations/01-bigram/lesson.toml"] == "file is missing"
    assert found["foundations/01-bigram/lesson.md"] == "file is missing"
    make(tmp_path, toml="title = [", md=MD)
    found = problems_of(lambda: load_lesson("foundations/01-bigram", tmp_path))
    assert "not valid TOML" in found["foundations/01-bigram/lesson.toml"]
    assert "should be written like" in problems_of(lambda: load_lesson("nope", tmp_path))[None]


def test_path_collects_problems_from_every_lesson(tmp_path):
    make(tmp_path, slug="02-mlp", toml=LESSON.replace("level = 0", "level = 5"))
    make(tmp_path, slug="03-attn", toml=LESSON.replace(
        "[experiment]", 'prerequisites = ["foundations/99-ghost"]\n[experiment]'))
    make(tmp_path, path_toml="level = 9\n")
    found = problems_of(lambda: load_path("foundations", tmp_path))
    assert found["foundations/path.toml: title"] == "is required"
    assert "level" in " ".join(found)
    assert any("02-mlp" in k and k.endswith("level") for k in found)
    assert any("03-attn" in k and "99-ghost" in v for k, v in found.items())


def test_compute_variants(tmp_path):
    cpu_gpu = LESSON.replace("estimate_minutes = 2", "estimate_minutes = 2") + (
        '\n[compute.gpu]\nbudget = "1e17 FLOPs"\nestimate_minutes = 240\n')
    make(tmp_path, toml=cpu_gpu)
    lesson = load_lesson("foundations/01-bigram", tmp_path)
    assert lesson.compute["gpu"].budget == "1e17 FLOPs" and lesson.compute["gpu"].preset is None
    # a gpu variant alone is refused
    only_gpu = LESSON.replace('[compute.cpu]\npreset = "tinystories-5min"\nestimate_minutes = 2',
                              '[compute.gpu]\npreset = "fineweb-edu"\nestimate_minutes = 90')
    make(tmp_path, slug="02-x", toml=only_gpu)
    found = problems_of(lambda: load_lesson("foundations/02-x", tmp_path))
    assert "needs a cpu variant too" in found["foundations/02-x/lesson.toml: compute.gpu"]
    # a long cpu estimate needs a gpu variant
    long_cpu = LESSON.replace("estimate_minutes = 2", "estimate_minutes = 16")
    make(tmp_path, slug="03-y", toml=long_cpu)
    found = problems_of(lambda: load_lesson("foundations/03-y", tmp_path))
    assert "over 15" in found["foundations/03-y/lesson.toml: compute.cpu.estimate_minutes"]
    make(tmp_path, slug="04-z", toml=long_cpu + '\n[compute.gpu]\npreset = "p"\n'
         "estimate_minutes = 60\n")
    assert load_lesson("foundations/04-z", tmp_path).compute["gpu"].estimate_minutes == 60
    # exactly 15 minutes is fine without a gpu variant
    make(tmp_path, slug="05-w",
         toml=LESSON.replace("estimate_minutes = 2", "estimate_minutes = 15"))
    assert load_lesson("foundations/05-w", tmp_path).estimate_minutes == 15


def test_lesson_md_sections():
    text = parse_lesson_md(MD)
    assert (text.intro, text.surface, text.deep) == (
        "Intro text.", "Train it and read the curve.", "Three seeds.")
    assert text.reading == "- Karpathy, makemore"
    only_surface = parse_lesson_md("## Surface\nHi\n### A subsection\nstays")
    assert only_surface.surface == "Hi\n### A subsection\nstays" and only_surface.deep == ""
    assert parse_lesson_md("## SURFACE\nupper case heading works").surface.startswith("upper")
    with pytest.raises(CurriculumError, match="'## Surface' section"):
        from nanoscope.learn import loader
        problems = loader._Problems("lesson.md")
        parse_lesson_md("## Deep\nonly deep", problems)
        raise CurriculumError(problems.items)


def test_schemas_describe_the_files():
    from nanoscope.learn.loader import tomllib  # no stdlib tomllib on Python 3.10

    assert_valid("lesson", tomllib.loads(LESSON))
    assert_valid("path", tomllib.loads('title = "F"\nlevel = 0\n[compute.cpu]\npreset = "p"\n'
                                       "estimate_minutes = 5\n"))
    with pytest.raises(AssertionError):
        assert_valid("lesson", tomllib.loads(LESSON.replace("level = 0", "level = 9")))
    with pytest.raises(AssertionError):  # a variant needs a preset or a budget
        assert_valid("lesson", tomllib.loads(LESSON.replace('preset = "tinystories-5min"\n', "")))


def test_progress_store(home):
    import json

    from nanoscope import paths
    from nanoscope.learn import progress

    lid = "foundations/01-bigram"
    assert progress.state(lid) == "not-started" and progress.read()["lessons"] == {}
    progress.mark(lid, "started")
    doc = json.loads((paths.learn_dir() / "progress.json").read_text())
    assert_valid("progress", doc)
    assert doc["lessons"][lid]["state"] == "started" and doc["lessons"][lid]["attempts"] == 0
    assert doc["lessons"][lid]["started_at"] is not None
    progress.mark(lid, "checking", check_id="c1")
    progress.mark(lid, "failed", check_id="c1")
    again = progress.mark(lid, "checking", check_id="c2")
    assert (again["attempts"], again["last_check"], again["state"]) == (2, "c2", "checking")
    done = progress.mark(lid, "passed", check_id="c2")
    assert done["passed_at"] is not None and progress.state(lid) == "passed"
    # what was earned stays earned
    assert progress.mark(lid, "failed")["state"] == "passed"
    assert progress.mark(lid, "checking")["state"] == "passed"
    assert progress.state("foundations/02-mlp") == "not-started"
    assert not list(paths.learn_dir().glob("*.tmp"))  # atomic: no leftovers
    with pytest.raises(ValueError, match="state must be one of"):
        progress.mark(lid, "done")


def test_progress_is_kept_per_owner(home):
    from nanoscope import paths
    from nanoscope.learn import progress

    progress.mark("a/01-x", "started")
    progress.mark("a/01-x", "passed", owner="ada")
    assert progress.state("a/01-x") == "started"
    assert progress.state("a/01-x", owner="ada") == "passed"
    assert paths.learn_dir("ada") == paths.learn_dir().parent / "users" / "ada" / "learn"
    with pytest.raises(ValueError, match="owner"):
        paths.learn_dir("../etc")


@pytest.fixture
def curricula(tmp_path, monkeypatch):
    """Two paths of fake lessons, installed as the package's curricula."""
    root = tmp_path / "curricula"
    make(root, slug="01-bigram", toml=LESSON.replace(
        "[compute.cpu]", '[compute.gpu]\nbudget = "1e16 FLOPs"\nestimate_minutes = 240\n'
        "[compute.cpu]"), path_toml='title = "Foundations"\nlevel = 0\n')
    (root / "foundations" / "01-bigram" / "starter.py").write_text("# your bigram\n")
    make(root, slug="02-mlp", toml=LESSON.replace("[experiment]", 'prerequisites = '
         '["foundations/01-bigram"]\n[experiment]').replace("Bigram", "MLP"),
         path_toml='title = "Foundations"\nlevel = 0\n')
    make(root, path="modern", slug="01-rope", toml=LESSON.replace("level = 0", "level = 1"),
         path_toml='title = "The modern block"\nlevel = 1\n')
    monkeypatch.setattr("nanoscope.learn.loader.curricula_dir", lambda: root)
    return root


def test_learn_list(curricula, home, capsys):
    from nanoscope.cli import main
    from nanoscope.learn import progress

    main(["learn", "list"])
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "foundations  Foundations (level 0)"
    assert out[1].startswith("  01-bigram  not-started") and "cpu 2 min / gpu 4 h" in out[1]
    assert "locked (needs foundations/01-bigram)" in out[2] and out[2].endswith("MLP")
    assert "modern  The modern block (level 1)" in out
    progress.mark("foundations/01-bigram", "passed")
    main(["learn", "list", "--path", "foundations"])
    out = capsys.readouterr().out
    assert "passed" in out.splitlines()[1] and "locked" not in out and "modern" not in out


MARIMO_NOTEBOOK = (
    'import marimo\n\napp = marimo.App()\n\n\n@app.cell\ndef _():\n    return\n')


def test_lesson_marimo_notebook_must_define_an_app(tmp_path):
    lesson = make(tmp_path)
    (lesson / "notebook.py").write_text(MARIMO_NOTEBOOK)
    assert load_lesson("foundations/01-bigram", root=tmp_path).has_notebook
    (lesson / "notebook.py").write_text("print('not a notebook')\n")
    found = problems_of(lambda: load_lesson("foundations/01-bigram", root=tmp_path))
    assert "does not define a marimo app" in found["foundations/01-bigram/notebook.py"]
    (lesson / "notebook.py").write_text("def broken(:\n")
    found = problems_of(lambda: load_lesson("foundations/01-bigram", root=tmp_path))
    assert "not valid Python" in found["foundations/01-bigram/notebook.py"]


def test_learn_start_copies_the_marimo_notebook(curricula, home, capsys, monkeypatch, tmp_path):
    from nanoscope import paths
    from nanoscope.cli import main

    (curricula / "foundations" / "01-bigram" / "notebook.py").write_text(MARIMO_NOTEBOOK)
    monkeypatch.setenv("NANOSCOPE_WORKSPACE", str(tmp_path / "ws"))
    main(["learn", "start", "foundations/01-bigram"])
    target = paths.workspace_dir() / "lessons" / "foundations" / "01-bigram" / "notebook.py"
    assert target.read_text() == MARIMO_NOTEBOOK and f"copied {target}" in capsys.readouterr().out


def test_learn_start(curricula, home, capsys, monkeypatch, tmp_path):
    from nanoscope import paths
    from nanoscope.cli import main
    from nanoscope.learn import progress

    monkeypatch.setenv("NANOSCOPE_WORKSPACE", str(tmp_path / "ws"))
    main(["learn", "start", "foundations/01-bigram"])
    out = capsys.readouterr().out
    target = paths.workspace_dir() / "lessons" / "foundations" / "01-bigram" / "starter.py"
    assert target.read_text() == "# your bigram\n"
    assert "started foundations/01-bigram: Bigram" in out and f"copied {target}" in out
    assert "compute: cpu 2 min / gpu 4 h" in out
    assert "next: edit the file, then run: nanoscope learn check foundations/01-bigram" in out
    assert progress.state("foundations/01-bigram") == "started"
    # never overwrite the learner's edits
    target.write_text("# my work\n")
    main(["learn", "start", "foundations/01-bigram"])
    out = capsys.readouterr().out
    assert target.read_text() == "# my work\n" and f"kept your {target} (not overwritten)" in out
    assert progress.state("foundations/01-bigram") == "started"
    # a lesson with no starter files, built on one not passed yet
    main(["learn", "start", "foundations/02-mlp"])
    out = capsys.readouterr().out
    assert "no starter files" in out and "builds on foundations/01-bigram, not passed yet" in out
    with pytest.raises(CurriculumError):
        load_lesson("foundations/99-none")
    # a passed lesson restarted keeps its state
    progress.mark("foundations/01-bigram", "passed")
    main(["learn", "start", "foundations/01-bigram"])
    assert progress.state("foundations/01-bigram") == "passed"


def test_learn_check_output(curricula, home, capsys, monkeypatch):
    import json

    from nanoscope import paths
    from nanoscope.cli import main
    from nanoscope.learn import checks, progress
    from nanoscope.learn.checks import Result

    verdicts = {"built": True, "learns": False}

    def fake(ctx, check):
        ok = verdicts[check.id]
        return Result(check.id, check.kind, ok, "all good" if ok else "val_bpb 3.1 is above 2.5",
                      {"value": 3.1})
    monkeypatch.setitem(checks.CHECKERS, "defines", fake)
    monkeypatch.setitem(checks.CHECKERS, "trains", fake)

    with pytest.raises(SystemExit) as caught:
        main(["learn", "check", "foundations/01-bigram"])
    assert caught.value.code == 1
    out = capsys.readouterr().out
    assert "checking foundations/01-bigram (2 checks)" in out
    assert "  [pass] built (defines): all good" in out
    assert "  [FAIL] learns (trains): val_bpb 3.1 is above 2.5" in out
    assert "result: 1 of 2 checks passed: read the reasons above" in out
    assert progress.state("foundations/01-bigram") == "failed"
    (file,) = list((paths.learn_dir() / "checks").glob("*.json"))
    doc = json.loads(file.read_text())
    assert_valid("check", doc)
    assert doc["passed"] is False and [c["passed"] for c in doc["checks"]] == [True, False]
    assert progress.entry("foundations/01-bigram")["last_check"] == doc["id"]

    verdicts["learns"] = True
    main(["learn", "check", "foundations/01-bigram"])  # exit code 0: no SystemExit
    assert "result: 2 of 2 checks passed" in capsys.readouterr().out
    assert progress.state("foundations/01-bigram") == "passed"
    assert len(list((paths.learn_dir() / "checks").glob("*.json"))) == 2


def test_failed_defines_skips_the_rest_and_crashes_are_reported(curricula, home, capsys,
                                                                monkeypatch):
    from nanoscope.cli import main
    from nanoscope.learn import checks
    from nanoscope.learn.checks import Result

    monkeypatch.setitem(checks.CHECKERS, "defines",
                        lambda ctx, c: Result(c.id, c.kind, False, "class Bigram not found"))
    with pytest.raises(SystemExit):
        main(["learn", "check", "foundations/01-bigram"])
    out = capsys.readouterr().out
    assert "[FAIL] built (defines): class Bigram not found" in out
    assert "[skip] learns (trains): skipped: fix the failing 'defines' check first" in out

    def boom(ctx, check):
        raise ZeroDivisionError("division by zero")
    monkeypatch.setitem(checks.CHECKERS, "defines", lambda ctx, c: Result(c.id, c.kind, True, "ok"))
    monkeypatch.setitem(checks.CHECKERS, "trains", boom)
    with pytest.raises(SystemExit):
        main(["learn", "check", "foundations/01-bigram"])
    assert "crashed: ZeroDivisionError: division by zero" in capsys.readouterr().out
    capsys.readouterr()
    with pytest.raises(SystemExit) as caught:  # 02-mlp has no gpu variant
        main(["learn", "check", "foundations/02-mlp", "--variant", "gpu"])
    assert caught.value.code == 2
    assert "has no gpu variant; it has: cpu" in capsys.readouterr().out


def test_check_job(curricula, home, capsys, monkeypatch):
    import json

    from nanoscope import queue
    from nanoscope.cli import main
    from nanoscope.learn import checks, progress
    from nanoscope.learn.checks import Result

    monkeypatch.setitem(checks.CHECKERS, "defines",
                        lambda ctx, c: Result(c.id, c.kind, True, "builds"))
    monkeypatch.setitem(checks.CHECKERS, "trains",
                        lambda ctx, c: Result(c.id, c.kind, True, "reaches 2.1"))
    main(["learn", "check", "foundations/01-bigram", "--queue"])
    out = capsys.readouterr().out
    assert "queued check job #1 for foundations/01-bigram" in out
    row = queue.get(1)
    assert (row["kind"], row["lane"]) == ("check", "interactive")
    assert json.loads(row["payload"]) == {"lesson": "foundations/01-bigram", "variant": "cpu"}
    queue.claim("w", "cpu")
    with pytest.raises(SystemExit) as stopped:
        main(["run-job", "1"])
    assert stopped.value.code == 0
    result = json.loads(queue.get(1)["result"])
    assert result["passed"] is True and result["lesson"] == "foundations/01-bigram"
    assert [c["reason"] for c in result["checks"]] == ["builds", "reaches 2.1"]
    assert progress.state("foundations/01-bigram") == "passed"
    with pytest.raises(queue.InvalidJob, match="variant"):
        queue.enqueue("check", {"lesson": "x", "variant": "tpu"})


AUTHOR_TOML = """\
title = "My layer"
level = 1
summary = "Build a thing."

[experiment]
kind = "check"

[[checks]]
id = "built"
kind = "defines"
class = "MyThing"

[compute.cpu]
preset = "tinystories-5min"
estimate_minutes = 0.2
"""
STARTER = "import torch.nn as nn\n\n\nclass Other(nn.Module):\n    pass\n"
SOLUTION = (
    "import torch\nimport torch.nn as nn\n\n\nclass MyThing(nn.Module):\n"
    "    def __init__(self):\n        super().__init__()\n"
    "        self.w = nn.Parameter(torch.ones(2))\n")


def author_folder(tmp_path, toml=AUTHOR_TOML, starter=STARTER, solution=SOLUTION):
    folder = tmp_path / "my-course" / "01-thing"
    folder.mkdir(parents=True)
    (folder / "lesson.toml").write_text(toml)
    (folder / "lesson.md").write_text("## Surface\nBuild it.\n")
    (folder / "starter.py").write_text(starter)
    if solution is not None:
        (folder / "solution.py").write_text(solution)
    return folder


def test_author_check_ready_when_the_starter_fails_and_the_solution_passes(tmp_path, home):
    from nanoscope.learn.authoring import author_check, format_author_check
    from nanoscope.learn.authorstate import last_check

    folder = author_folder(tmp_path)
    doc = author_check(folder)
    assert_valid("author-check", doc)
    assert doc["state"] == "ready" and doc["lesson"] == "my-course/01-thing"
    assert doc["summary"] == ("The folder loads, starter.py fails its check "
                              "and solution.py passes it.")
    [row] = doc["checks"]
    assert not row["starter"]["passed"] and row["solution"]["passed"]
    assert "MyThing" in row["starter"]["reason"]
    assert "[FAIL] built (defines) on starter.py" in format_author_check(doc)

    assert last_check(folder)["fresh"] is True
    (folder / "starter.py").write_text(STARTER + "\n# edited\n")
    assert last_check(folder)["fresh"] is False


def test_author_check_lists_every_problem_when_the_folder_does_not_load(tmp_path, home):
    from nanoscope.learn.authoring import author_check

    bad = AUTHOR_TOML.replace('level = 1', 'level = 1\nunlocks = ["LayerNorm"]') \
        .replace('kind = "defines"', 'kind = "definez"').replace("[compute.cpu]", "[compute.gpu]")
    doc = author_check(author_folder(tmp_path, toml=bad))
    assert_valid("author-check", doc)
    assert doc["state"] == "does not load" and doc["title"] is None and doc["checks"] == []
    where = [p["where"] for p in doc["problems"]]
    assert "lesson.toml: unlocks[0]" in where and "lesson.toml: checks[0].kind" in where
    assert doc["summary"].startswith(f"The curriculum loader found {len(where)} problems.")


def test_author_check_says_when_the_solution_fails_or_the_starter_passes(tmp_path, home):
    from nanoscope.learn.authoring import author_check

    both_fail = author_check(author_folder(tmp_path, solution=STARTER))
    assert both_fail["state"] == "solution fails"
    assert both_fail["summary"].startswith("The folder loads, but solution.py fails its check")
    tmp2 = tmp_path / "again"
    tmp2.mkdir()
    both_pass = author_check(author_folder(tmp2, starter=SOLUTION))
    assert both_pass["state"] == "starter passes"
    tmp3 = tmp_path / "third"
    tmp3.mkdir()
    gone = author_check(author_folder(tmp3, solution=None))
    assert gone["state"] == "files missing" and "solution.py is missing" in gone["summary"]


def test_author_check_cli(tmp_path, home, capsys):
    from nanoscope.cli import main

    folder = author_folder(tmp_path)
    main(["learn", "author-check", str(folder)])  # exit code 0: it returns
    out = capsys.readouterr().out
    assert "my-course/01-thing: ready" in out and "on solution.py" in out
    (folder / "solution.py").write_text(STARTER)
    with pytest.raises(SystemExit) as failed:
        main(["learn", "author-check", str(folder)])
    assert failed.value.code == 1
