"""Writes the structure job results the web tests use (web/test/fixtures/structure-*.json): lesson
04's starter as shipped (unfinished: it builds nothing and its forward raises) and lesson 06's
solution (every module traced). Not collected by default; regenerate from the repository root with

    uv run pytest tests/server/gen_structure_fixtures.py
"""
import json
from pathlib import Path

from nanoscope.inspect import structure
from nanoscope.learn.loader import load_lesson
from nanoscope.modelref import load_class

OUT = Path("web/test/fixtures")


def dump(name, obj):
    obj["file"] = Path(obj["file"]).relative_to(Path.cwd()).as_posix()  # not this machine's path
    (OUT / name).write_text(json.dumps(obj, indent=2) + "\n")


def test_dump():
    lesson = load_lesson("foundations/04-multi-head")
    check = next(c for c in lesson.checks if c.kind == "equivalent")
    starter = "nanoscope/curricula/foundations/04-multi-head/starter.py"
    dump("structure-04-starter.json",
         structure(load_class(f"{starter}:MultiHead"), check.args.get("args"),
                   check.args["inputs"]))
    dump("structure-06-solution.json",
         structure(load_class("tests/solutions/foundations/06-gpt2.py:MyGPT2")))
