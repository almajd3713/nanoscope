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


def run_cli(*argv):
    from nanoscope.cli import main

    try:
        main(["learn", *argv])
    except SystemExit as exc:
        return exc.code
    return 0


def test_unlock_cli(home, lock_curricula, capsys):
    from nanoscope.learn import gating

    assert run_cli("unlock") == 2 and "name what to unlock" in capsys.readouterr().out
    unlocks.set_policy("guided")
    assert run_cli("unlock", "Attention") == 2
    assert "skipping a lesson needs a reason" in capsys.readouterr().out
    assert run_cli("unlock", "Nope", "--reason", "x") == 2
    assert "'Nope' is not locked by any lesson" in capsys.readouterr().out
    assert run_cli("unlock", "Attention", "--reason", "I know this") == 0  # a bare name works
    out = capsys.readouterr().out
    assert "block:Attention unlocked (skipped; recorded with your reason)" in out
    entry = unlocks.read()["unlocks"]["block:Attention"]
    assert (entry["how"], entry["reason"], entry["lesson"]) == (
        "skipped", "I know this", "foundations/04-attention")
    assert gating.check(["block:Attention", "block:RoPE"])[0].id == "block:RoPE"
    # unlock --all: policy open, every lockable id recorded as open, earned/skipped kept
    unlocks.earn("modern/01-rope", ["block:RoPE"], "learn/checks/c.json")
    assert run_cli("unlock", "--all") == 0
    doc = unlocks.read()
    assert doc["policy"] == "open"
    assert {k: v["how"] for k, v in doc["unlocks"].items()} == {
        "block:Attention": "skipped", "block:RoPE": "earned", "feature:gqa": "open"}
    assert gating.check(["feature:gqa"]) == []
    # lock --reset: guided again, keeps earned and skipped, drops what was only opened
    assert run_cli("lock") == 2
    assert run_cli("lock", "--reset") == 0
    doc = unlocks.read()
    assert doc["policy"] == "guided" and set(doc["unlocks"]) == {"block:Attention", "block:RoPE"}
    assert [x.id for x in gating.check(["feature:gqa", "block:RoPE"])] == ["feature:gqa"]
    assert_valid("unlocks", doc)


def test_first_start_guided(home, lock_curricula, capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("NANOSCOPE_WORKSPACE", str(tmp_path / "ws"))
    assert not unlocks.exists()
    assert run_cli("start", "foundations/04-attention") == 0
    out = capsys.readouterr().out
    assert unlocks.policy() == "guided"
    assert ("bigger blocks stay locked until you build them in a lesson; to unlock "
            "everything now: nanoscope learn unlock --all") in out
    assert run_cli("start", "modern/01-rope") == 0  # the second start changes nothing
    assert "to unlock everything now" not in capsys.readouterr().out
    unlocks.set_policy("open")
    assert run_cli("start", "modern/01-rope") == 0
    assert unlocks.policy() == "open"  # an existing choice is respected


def test_start_open_skips_gating(home, lock_curricula, capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("NANOSCOPE_WORKSPACE", str(tmp_path / "ws"))
    assert run_cli("start", "foundations/04-attention", "--open") == 0
    assert unlocks.policy() == "open"
    assert "gating is off (--open); turn it on any time: nanoscope learn lock --reset" in (
        capsys.readouterr().out)


def test_learn_status(home, lock_curricula, capsys):
    assert run_cli("status") == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "policy: open (no unlocks.json yet: nothing is locked)"
    assert "block:Attention  open" in out
    unlocks.set_policy("guided")
    unlocks.earn("foundations/04-attention", ["block:Attention"], "learn/checks/c.json")
    unlocks.grant("feature:gqa", "skipped", "modern/01-rope", reason="used it at work")
    run_cli("status")
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "policy: guided"
    assert "foundations  foundations (level 0)" in out  # the lessons, as in `learn list`
    table = out.split("unlocks:\n")[1].splitlines()
    assert [row.split()[:2] for row in table] == [
        ["block:Attention", "earned"], ["block:RoPE", "locked"], ["feature:gqa", "skipped"]]
    assert "foundations/04-attention" in table[0] and "modern/01-rope" in table[1]
    assert "skipped (used it at work)" in table[2]


def test_cli_and_study(home, lock_curricula, tmp_path, capsys):
    from nanoscope import Study
    from nanoscope.cli import _load_model_class, main
    from nanoscope.learn.gating import LockedBlockError
    from nanoscope.models import Modern

    file = tmp_path / "mine.py"
    file.write_text(GUIDED_FILE.replace("from nanoscope.blocks import Attention, RoPE",
                                        "from nanoscope.blocks.attention import Attention\n"
                                        "from nanoscope.blocks.positional import RoPE"))
    unlocks.set_policy("guided")
    with pytest.raises(SystemExit) as caught:
        main(["run", f"{file}:MyLM", "--preset", "tinystories-5min"])
    assert caught.value.code == 2
    out = capsys.readouterr().out
    assert "mine.py line 11" in out and "feature:gqa is locked" in out
    assert "nanoscope learn unlock --all" in out
    assert not (paths.runs_dir() / "tinystories-5min").exists()  # nothing was started

    MyLM = _load_model_class(f"{file}:MyLM")
    study = Study("gated", preset="tinystories-5min", seeds=1)
    study.add("mine", MyLM)
    with pytest.raises(LockedBlockError):
        study.run()
    with pytest.raises(LockedBlockError):
        study.enqueue()
    shipped = Study("fine", preset="tinystories-5min", seeds=1)
    shipped.add("modern", Modern, n_kv_heads=1)  # GQA in a shipped model: always allowed
    shipped._refuse_locked()
    # an open policy lets the same study through the gate
    unlocks.set_policy("open")
    study._refuse_locked()


def test_owner_path(home, lock_curricula, monkeypatch):
    """All learner state goes through learn_dir(owner), so a hosted deployment can give each
    user their own folder."""
    import ast
    from pathlib import Path

    import nanoscope.learn as learn_package
    from nanoscope.learn import checks, gating, progress
    from nanoscope.learn.checks import Result, run_lesson_checks
    from nanoscope.learn.loader import load_lesson

    ada = paths.learn_dir("ada")
    assert ada == paths.learn_dir().parent / "users" / "ada" / "learn" != paths.learn_dir()
    unlocks.set_policy("guided", owner="ada")
    progress.mark("foundations/04-attention", "started", owner="ada")
    assert not unlocks.exists() and (ada / "unlocks.json").exists()
    assert progress.state("foundations/04-attention") == "not-started"
    assert [x.id for x in gating.check(["block:Attention"], owner="ada")] == ["block:Attention"]
    assert gating.check(["block:Attention"]) == []  # the local user is still open

    monkeypatch.setattr(checks, "CHECKERS", {"forbid": lambda c, k: Result(k.id, k.kind, True,
                                                                          "all fine here")})
    toml = lock_curricula / "foundations" / "04-attention" / "lesson.toml"
    toml.write_text(toml.read_text() + '\n[[checks]]\nid = "a"\nkind = "forbid"\n')
    run_lesson_checks(load_lesson("foundations/04-attention"), owner="ada")
    assert list((ada / "checks").glob("*.json")) and not (paths.learn_dir() / "checks").exists()
    assert unlocks.read("ada")["unlocks"]["block:Attention"]["how"] == "earned"
    assert progress.state("foundations/04-attention", owner="ada") == "passed"
    assert gating.check(["block:Attention"], owner="ada") == []

    # no module in the learn package builds a learner path without an owner
    for file in Path(learn_package.__file__).parent.glob("*.py"):
        for node in ast.walk(ast.parse(file.read_text())):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "learn_dir":
                assert node.args or node.keywords, f"{file.name}:{node.lineno} learn_dir() "


def test_lock_table_consistent():
    """The lock list comes from the curricula, so it must agree with the block registry."""
    import importlib

    from nanoscope.blocks import registry
    from nanoscope.learn import gating, loader

    gating.reload()
    importlib.import_module("nanoscope.blocks").load_all()
    unlockers: dict[str, list[str]] = {}
    for path_id in loader.list_path_ids():
        for lesson in loader.load_path(path_id).lessons:
            for unlock_id in lesson.unlocks:
                unlockers.setdefault(unlock_id, []).append(lesson.id)
    assert unlockers, "no lesson unlocks anything"
    # every lockable id has exactly one unlocking lesson
    assert {i: ls for i, ls in unlockers.items() if len(ls) != 1} == {}
    assert gating.lock_table() == {i: ls[0] for i, ls in unlockers.items()}
    # every unlock id names a registered block or feature
    blocks = {info.name: info for info in registry.all_blocks()}
    features = {f for info in blocks.values() for f in info.features}
    for unlock_id in unlockers:
        kind, _, name = unlock_id.partition(":")
        assert kind in ("block", "feature"), unlock_id
        assert name in (blocks if kind == "block" else features), f"{unlock_id} is not registered"
    # every composite-tier block is lockable; primitives never are
    for info in blocks.values():
        if not info.module.startswith("nanoscope."):
            continue  # blocks other tests define for themselves
        locked = f"block:{info.name}" in unlockers
        assert locked == (info.tier == "composite" and not info.user), (
            f"{info.name} is tier {info.tier} but "
            f"{'is' if locked else 'is not'} unlocked by a lesson")


@pytest.mark.usefixtures("fake_data")
def test_level0_untouched(home):
    """A learner who never starts a lesson sees no gating at all."""
    import importlib

    from fakes import tiny

    from nanoscope import run
    from nanoscope.learn import gating
    from nanoscope.models import GPT2, Bigram

    blocks = importlib.import_module("nanoscope.blocks")
    for name in ("Attention", "Block", "Decoder", "RMSNorm", "RoPE", "SwiGLU"):
        assert getattr(blocks, name).__name__ == name  # every locked name imports freely
    assert gating.check(["block:Attention", "feature:gqa"]) == []
    assert gating.scan(__file__) == []
    result = run(GPT2, tiny(), device="cpu", progress=False, n_layers=1, d_model=16, n_heads=2)
    assert result.final_step == 20
    assert run(Bigram, tiny(), device="cpu", progress=False).final_step == 20
    # and nothing about it was recorded
    assert not paths.learn_dir().exists() and not unlocks.exists()
    assert unlocks.policy() == "open"


@pytest.fixture
def window_curricula(tmp_path, monkeypatch):
    """modern/02-window unlocks feature:sliding_window."""
    from nanoscope.learn import gating

    root = tmp_path / "curricula"
    folder = root / "modern" / "02-window"
    folder.mkdir(parents=True)
    (root / "modern" / "path.toml").write_text('title = "modern"\nlevel = 0\n')
    (folder / "lesson.toml").write_text(BASE.format(extra='unlocks = ["feature:sliding_window"]'))
    (folder / "lesson.md").write_text("## Surface\nHi\n")
    monkeypatch.setattr("nanoscope.learn.loader.curricula_dir", lambda: root)
    gating.reload()
    yield root
    gating.reload()


WINDOW_FILE = '''\
from nanoscope.blocks import Attention, Block, Decoder, RMSNorm, SwiGLU


class MyLM(Decoder):
    def __init__(self, vocab_size: int):
        local = Block(norm=RMSNorm(), mlp=SwiGLU(), attn=Attention(n_heads=4, window=4))
        glob = Block(norm=RMSNorm(), mlp=SwiGLU(), attn=Attention(n_heads=4))
        super().__init__(vocab_size, 8, d_model=16, n_layers=4, pattern=[local, local, glob])
'''


def test_sliding_window_is_a_feature_lock_in_a_layer_pattern(home, window_curricula, tmp_path):
    from nanoscope.blocks.attention import Attention
    from nanoscope.blocks.mlp import SwiGLU
    from nanoscope.blocks.norm import RMSNorm
    from nanoscope.blocks.structure import Block, Decoder
    from nanoscope.learn import gating
    from nanoscope.learn.gating import LockedBlockError

    assert gating.lock_table() == {"feature:sliding_window": "modern/02-window"}

    def block(**attn):
        return Block(norm=RMSNorm(), mlp=SwiGLU(), attn=Attention(n_heads=4, **attn))

    def decoder(pattern):
        return Decoder(vocab_size=20, context_length=8, d_model=16, n_layers=4, pattern=pattern)

    unlocks.set_policy("guided")
    decoder([block(), block()])  # global attention only: nothing locked
    with pytest.raises(LockedBlockError, match="feature:sliding_window is locked") as caught:
        decoder([block(window=4), block(window=4), block()])  # local, local, global
    assert caught.value.locked.lesson == "modern/02-window"
    # the static scan finds the line of the window argument, inside the pattern's blocks
    file = tmp_path / "mine.py"
    file.write_text(WINDOW_FILE)
    assert [(u.id, u.line) for u in gating.scan(file)] == [("feature:sliding_window", 6)]
    unlocks.earn("modern/02-window", ["feature:sliding_window"], "learn/checks/c.json")
    layers = decoder([block(window=4), block(window=4), block()])
    assert [m.attn.window for m in layers.blocks] == [4, 4, None, 4]  # the pattern repeats
    assert gating.scan(file) == []
