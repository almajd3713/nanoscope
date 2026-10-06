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
