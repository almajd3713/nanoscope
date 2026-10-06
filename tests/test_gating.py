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


def test_build_gate(home, lock_curricula):
    from nanoscope.blocks.attention import Attention
    from nanoscope.blocks.composite import Composite
    from nanoscope.blocks.mlp import SwiGLU
    from nanoscope.blocks.norm import RMSNorm
    from nanoscope.blocks.positional import RoPE
    from nanoscope.blocks.structure import Block, Decoder
    from nanoscope.learn.gating import LockedBlockError
    from nanoscope.models import Modern

    def decoder(**attn):
        return Decoder(vocab_size=20, context_length=8, d_model=16, n_layers=1,
                       block=Block(norm=RMSNorm(), attn=Attention(n_heads=4, **attn),
                                   mlp=SwiGLU()))

    decoder(n_kv_heads=2)  # no unlocks.json: nothing is checked
    unlocks.set_policy("guided")
    with pytest.raises(LockedBlockError, match="block:Attention is locked") as caught:
        decoder()
    assert [x.id for x in caught.value.all] == ["block:Attention"]
    unlocks.earn("foundations/04-attention", ["block:Attention"], "learn/checks/c.json")
    assert decoder().blocks[0].attn.n_kv_heads == 4  # plain multi-head is fine now
    with pytest.raises(LockedBlockError, match="feature:gqa is locked") as caught:
        decoder(n_kv_heads=2)  # a feature lock an import check can't see
    assert caught.value.locked.id == "feature:gqa"
    with pytest.raises(LockedBlockError) as caught:
        decoder(n_kv_heads=2, pos=RoPE())
    assert [x.id for x in caught.value.all] == ["block:RoPE", "feature:gqa"]
    assert "this model uses 2 locked parts: block:RoPE, feature:gqa" in str(caught.value)
    # a subclass is gated too, a shipped model never is
    class Mine(Decoder):
        def __init__(self, vocab_size):
            super().__init__(vocab_size, 8, d_model=16, n_layers=1, block=Block(
                norm=RMSNorm(), attn=Attention(n_heads=2, n_kv_heads=1), mlp=SwiGLU()))
    with pytest.raises(LockedBlockError):
        Mine(20)
    model = Modern(vocab_size=20, context_length=8, d_model=16, n_layers=1, n_heads=4,
                   n_kv_heads=1)  # GQA, RoPE, QK-norm, z-loss: all fine, it is shipped
    assert model.blocks[0].attn.n_kv_heads == 1
    # composite templates check their slots
    class Template(Composite):
        SLOTS = ("attn",)
    with pytest.raises(LockedBlockError, match="block:RoPE"):
        Template(attn=RoPE()).build(16, 8)
    unlocks.set_policy("open")
    assert decoder(n_kv_heads=2, pos=RoPE()).blocks[0].attn.n_kv_heads == 2


GUIDED_FILE = '''\
from nanoscope.blocks import Block, Decoder, RMSNorm, SwiGLU
from nanoscope.blocks import Attention, RoPE


class MyLM(Decoder):
    def __init__(self, vocab_size: int):
        super().__init__(
            vocab_size, 8, d_model=16, n_layers=1, z_loss=1e-4,
            block=Block(norm=RMSNorm(), mlp=SwiGLU(),
                        attn=Attention(n_heads=4, n_kv_heads=2, pos=RoPE(), qk_norm=True)),
        )
'''


def test_scan(home, lock_curricula, tmp_path):
    from nanoscope.learn import gating

    file = tmp_path / "mine.py"
    file.write_text(GUIDED_FILE)
    assert gating.scan(file) == []  # no unlocks.json: open
    unlocks.set_policy("guided")
    uses = gating.scan(file)
    assert [(u.id, u.line) for u in uses] == [
        ("block:Attention", 2), ("block:RoPE", 2), ("block:Attention", 10), ("block:RoPE", 10),
        ("feature:gqa", 10)]
    assert uses[0].lesson == "foundations/04-attention"
    assert "nanoscope learn start foundations/04-attention" in uses[0].message
    unlocks.earn("foundations/04-attention", ["block:Attention"], "e")
    assert {u.id for u in gating.scan(file)} == {"block:RoPE", "feature:gqa"}
    unlocks.set_policy("open")
    assert gating.scan(file) == []
    # reading the file runs nothing
    boom = tmp_path / "boom.py"
    boom.write_text("raise RuntimeError('never')\nfrom nanoscope.blocks import RoPE\n")
    unlocks.set_policy("guided")
    assert [u.id for u in gating.scan(boom)] == ["block:RoPE"]


def test_scan_feeds_run_request_validation(home, lock_curricula, tmp_path):
    from nanoscope.cli import _load_model_class
    from nanoscope.models import Modern
    from nanoscope.specs import validate_run_request

    file = tmp_path / "mine.py"
    file.write_text(GUIDED_FILE.replace("from nanoscope.blocks import Attention, RoPE",
                                        "from nanoscope.blocks.attention import Attention\n"
                                        "from nanoscope.blocks.positional import RoPE"))
    MyLM = _load_model_class(f"{file}:MyLM")
    assert validate_run_request(MyLM, "tinystories-5min", {}) == []  # open
    unlocks.set_policy("guided")
    problems = validate_run_request(MyLM, "tinystories-5min", {})
    assert {p.code for p in problems} == {"locked"}
    assert any("line 11" in p.message and "feature:gqa is locked" in p.message
               for p in problems)
    assert all("unlocks in the lesson" in p.hint for p in problems)
    assert validate_run_request(Modern, "tinystories-5min", {"n_kv_heads": 1}) == []  # shipped


def test_forbid_check_refuses_locked_blocks(home, lock_curricula, tmp_path, monkeypatch):
    from learn_helpers import lesson_with

    from nanoscope.learn.checks import run_one

    toml = '\n[[checks]]\nid = "own"\nkind = "forbid"\n'
    ctx = lesson_with(tmp_path / "cur", tmp_path / "ws", monkeypatch, toml, GUIDED_FILE)
    monkeypatch.setattr("nanoscope.learn.loader.curricula_dir", lambda: lock_curricula)
    gating_reload()
    assert run_one(ctx, ctx.lesson.checks[0]).passed  # no unlocks.json
    unlocks.set_policy("guided")
    result = run_one(ctx, ctx.lesson.checks[0])
    assert not result.passed
    assert result.reason.startswith("starter.py uses blocks you have not unlocked yet: line 2: "
                                    "block:Attention (locked until foundations/04-attention)")
    assert result.evidence["locked"][0] == {
        "line": 2, "id": "block:Attention", "lesson": "foundations/04-attention"}


def gating_reload():
    from nanoscope.learn import gating

    gating.reload()


def test_earn(home, lock_curricula, capsys, monkeypatch):
    import json

    from nanoscope.learn import checks, gating, progress
    from nanoscope.learn.checks import Result, run_lesson_checks
    from nanoscope.learn.loader import load_lesson

    lesson = load_lesson("modern/01-rope")  # unlocks block:RoPE and feature:gqa
    verdict = {"ok": False}
    monkeypatch.setattr(checks, "CHECKERS", dict(checks.CHECKERS))
    checks.CHECKERS["defines"] = lambda ctx, c: Result(c.id, c.kind, True, "builds")
    checks.CHECKERS["trains"] = lambda ctx, c: Result(
        c.id, c.kind, verdict["ok"], "learns" if verdict["ok"] else "does not learn yet")
    lesson_toml = lock_curricula / "modern" / "01-rope" / "lesson.toml"
    lesson_toml.write_text(lesson_toml.read_text() + (
        '\n[[checks]]\nid = "a"\nkind = "defines"\nclass = "X"\n'
        '[[checks]]\nid = "b"\nkind = "trains"\nmetric = "val_bpb"\nthreshold = 1\n'))
    lesson = load_lesson("modern/01-rope")
    unlocks.set_policy("guided")

    run_lesson_checks(lesson)  # a failing check earns nothing
    assert unlocks.read()["unlocks"] == {} and progress.state(lesson.id) == "failed"
    assert [x.id for x in gating.check(["block:RoPE", "feature:gqa"])] == [
        "block:RoPE", "feature:gqa"]

    verdict["ok"] = True
    doc = run_lesson_checks(lesson)
    assert doc["passed"] and progress.state(lesson.id) == "passed"
    stored = unlocks.read()["unlocks"]
    assert set(stored) == {"block:RoPE", "feature:gqa"}
    for entry in stored.values():
        assert entry["how"] == "earned" and entry["lesson"] == "modern/01-rope"
        assert entry["evidence"] == f"learn/checks/{doc['id']}.json"
    evidence = json.loads((paths.learn_dir() / "checks" / f"{doc['id']}.json").read_text())
    assert evidence["passed"] is True  # the evidence file exists and says so
    assert gating.check(["block:RoPE", "feature:gqa"]) == []
    assert "unlocked: block:RoPE, feature:gqa" in capsys.readouterr().out
    # a later failure takes nothing back
    verdict["ok"] = False
    run_lesson_checks(lesson)
    assert set(unlocks.read()["unlocks"]) == {"block:RoPE", "feature:gqa"}
    assert progress.state(lesson.id) == "passed"


def test_passing_without_a_guided_start_records_no_unlocks(home, lock_curricula, monkeypatch):
    from nanoscope.learn import checks
    from nanoscope.learn.checks import Result, run_lesson_checks
    from nanoscope.learn.loader import load_lesson

    lesson_toml = lock_curricula / "modern" / "01-rope" / "lesson.toml"
    lesson_toml.write_text(lesson_toml.read_text()
                           + '\n[[checks]]\nid = "a"\nkind = "forbid"\n')
    ok = {"forbid": lambda c, k: Result(k.id, k.kind, True, "ok")}
    monkeypatch.setattr(checks, "CHECKERS", ok)
    run_lesson_checks(load_lesson("modern/01-rope"))
    assert not unlocks.exists()  # still open: nothing to earn, nothing was locked
