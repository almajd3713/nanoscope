"""Fake curricula for the learn tests: one lesson per call, a learner's file in the workspace."""

import uuid
from pathlib import Path

from nanoscope.learn.loader import load_lesson

BASE = """\
title = "Lesson"
level = 0
{extra}
[compute.cpu]
preset = "tinystories-5min"
estimate_minutes = 1
"""


def lesson_with(root: Path, workspace: Path, monkeypatch, checks_toml: str, user_code: str,
                extra: str = "", slug: str = "01-x"):
    """Write a lesson with these [[checks]] and the learner's starter.py; returns a Context."""
    from nanoscope.learn.checks import Context

    folder = root / "p" / slug
    folder.mkdir(parents=True, exist_ok=True)
    (root / "p" / "path.toml").write_text('title = "P"\nlevel = 0\n')
    (folder / "lesson.toml").write_text(BASE.format(extra=extra) + checks_toml)
    (folder / "lesson.md").write_text("## Surface\nHi\n")
    monkeypatch.setenv("NANOSCOPE_WORKSPACE", str(workspace))
    monkeypatch.setattr("nanoscope.learn.loader.curricula_dir", lambda: root)
    lesson = load_lesson(f"p/{slug}")
    ctx = Context(lesson, check_id=uuid.uuid4().hex)  # a fresh runs folder per context
    ctx.user_file.parent.mkdir(parents=True, exist_ok=True)
    ctx.user_file.write_text(user_code)
    return ctx
