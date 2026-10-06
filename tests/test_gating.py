"""Lesson gating: the unlock store, the policy, and the three places it is enforced."""

import json

import pytest
from helpers import assert_valid

from nanoscope import paths
from nanoscope.learn import unlocks


def test_store(home):
    assert not unlocks.exists() and unlocks.policy() == "open"
    assert unlocks.is_unlocked("block:Attention")  # no file: nothing is locked
    unlocks.set_policy("guided")
    assert not unlocks.is_unlocked("block:Attention")
    unlocks.earn("foundations/04-attention", ["block:Attention", "feature:gqa"],
                 "learn/checks/c1.json")
    doc = json.loads((paths.learn_dir() / "unlocks.json").read_text())
    assert_valid("unlocks", doc)
    entry = doc["unlocks"]["block:Attention"]
    assert (entry["how"], entry["lesson"], entry["evidence"]) == (
        "earned", "foundations/04-attention", "learn/checks/c1.json")
    assert doc["policy"] == "guided" and set(doc["unlocks"]) == {"block:Attention", "feature:gqa"}
    assert unlocks.is_unlocked("block:Attention") and not unlocks.is_unlocked("block:RoPE")
    # what was earned is never downgraded
    unlocks.grant("block:Attention", "skipped", "x")
    unlocks.grant("block:Attention", "open")
    assert unlocks.read()["unlocks"]["block:Attention"]["how"] == "earned"
    unlocks.grant("block:RoPE", "open")
    unlocks.grant("block:RoPE", "skipped", "foundations/x")
    assert unlocks.read()["unlocks"]["block:RoPE"]["how"] == "skipped"  # skipped beats open
    assert not list(paths.learn_dir().glob("*.tmp"))  # atomic writes leave nothing behind
    with pytest.raises(ValueError, match="policy must be"):
        unlocks.set_policy("lenient")
    with pytest.raises(ValueError, match="how must be"):
        unlocks.grant("block:X", "stolen")
    # one owner's unlocks are not another's
    assert unlocks.policy(owner="ada") == "open" and not unlocks.exists("ada")


from learn_helpers import BASE  # noqa: E402


@pytest.fixture
def lock_curricula(tmp_path, monkeypatch):
    """foundations/04-attention unlocks block:Attention; modern/01-rope unlocks two more."""
    from nanoscope.learn import gating

    root = tmp_path / "curricula"
    for path, slug, ids in (("foundations", "04-attention", ["block:Attention"]),
                            ("modern", "01-rope", ["block:RoPE", "feature:gqa"])):
        folder = root / path / slug
        folder.mkdir(parents=True)
        (root / path / "path.toml").write_text(f'title = "{path}"\nlevel = 0\n')
        (folder / "lesson.toml").write_text(BASE.format(extra=f"unlocks = {json.dumps(ids)}"))
        (folder / "lesson.md").write_text("## Surface\nHi\n")
    monkeypatch.setattr("nanoscope.learn.loader.curricula_dir", lambda: root)
    gating.reload()
    yield root
    gating.reload()


def test_policy_default_open(home, lock_curricula):
    from nanoscope.learn import gating

    assert gating.lock_table() == {
        "block:Attention": "foundations/04-attention", "block:RoPE": "modern/01-rope",
        "feature:gqa": "modern/01-rope"}
    assert gating.lockable("block:RoPE") and not gating.lockable("block:Linear")
    # no unlocks.json: policy open, nothing is locked
    assert gating.check(["block:Attention", "feature:gqa"]) == []
    unlocks.set_policy("guided")
    locked = gating.check(["block:Attention", "block:Linear", "feature:gqa"])
    assert [(x.id, x.lesson) for x in locked] == [
        ("block:Attention", "foundations/04-attention"), ("feature:gqa", "modern/01-rope")]
    assert "nanoscope learn start foundations/04-attention" in locked[0].message
    assert "nanoscope learn unlock --all" in locked[0].message
    unlocks.earn("foundations/04-attention", ["block:Attention"], "learn/checks/c.json")
    assert [x.id for x in gating.check(["block:Attention", "feature:gqa"])] == ["feature:gqa"]
    unlocks.set_policy("open")
    assert gating.check(["feature:gqa"]) == []
    assert gating.check(["feature:gqa"], owner="ada") == []  # another owner: no file, open


def test_broken_lessons_never_break_the_lock_table(home, lock_curricula):
    from nanoscope.learn import gating

    (lock_curricula / "modern" / "01-rope" / "lesson.toml").write_text("title = [")
    gating.reload()
    assert gating.lock_table() == {"block:Attention": "foundations/04-attention"}


def test_import_gate(home, lock_curricula):
    import importlib

    from nanoscope.learn.gating import LockedBlockError

    blocks = importlib.import_module("nanoscope.blocks")
    assert blocks.RMSNorm.__name__ == "RMSNorm"  # no unlocks.json: nothing is gated
    unlocks.set_policy("guided")
    with pytest.raises(LockedBlockError) as caught:
        from nanoscope.blocks import Attention  # noqa: F401
    message = str(caught.value)
    assert isinstance(caught.value, ImportError)
    assert "block:Attention is locked until you build it yourself in the lesson " \
           "foundations/04-attention" in message
    assert "nanoscope learn start foundations/04-attention" in message
    assert "nanoscope learn unlock --all" in message
    assert caught.value.name == "Attention"
    assert blocks.RMSNorm and blocks.Linear  # not lockable: never locked
    assert not hasattr(blocks, "NoSuchBlock")
    # library code imports submodules, so shipped models and tools are not gated
    from nanoscope.blocks.attention import Attention as Shipped
    assert Shipped.__name__ == "Attention"
    from nanoscope.blocks.catalog import catalog
    assert any(b["name"] == "Attention" for b in catalog()["blocks"])
    unlocks.earn("foundations/04-attention", ["block:Attention"], "learn/checks/c.json")
    assert blocks.Attention is Shipped
    unlocks.set_policy("open")
    assert blocks.RoPE.__name__ == "RoPE"
