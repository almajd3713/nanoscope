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
    import tomllib

    assert_valid("lesson", tomllib.loads(LESSON))
    assert_valid("path", tomllib.loads('title = "F"\nlevel = 0\n[compute.cpu]\npreset = "p"\n'
                                       "estimate_minutes = 5\n"))
    with pytest.raises(AssertionError):
        assert_valid("lesson", tomllib.loads(LESSON.replace("level = 0", "level = 9")))
    with pytest.raises(AssertionError):  # a variant needs a preset or a budget
        assert_valid("lesson", tomllib.loads(LESSON.replace('preset = "tinystories-5min"\n', "")))
