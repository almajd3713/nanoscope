"""The kinds of check: each says pass or fail in words, with the numbers behind it."""

import textwrap

import pytest
from learn_helpers import lesson_with

from nanoscope.learn.checks import CHECKERS, run_one

RMS_OK = '''\
import torch
import torch.nn as nn


class MyNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x / x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt() * self.weight
'''

DEFINES = '\n[[checks]]\nid = "built"\nkind = "defines"\nclass = "MyNorm"\n'
EQUIV = textwrap.dedent('''
    [[checks]]
    id = "same"
    kind = "equivalent"
    class = "MyNorm"
    reference = "rms_norm"
    inputs = [[2, 5, 16]]
    call = ["input:0", "param:weight"]
    ''')


@pytest.fixture
def make(tmp_path, monkeypatch, home):
    def build(checks, code, **kw):
        return lesson_with(tmp_path / "cur", tmp_path / "ws", monkeypatch, checks, code, **kw)
    return build


def run(ctx, index=0):
    return run_one(ctx, ctx.lesson.checks[index])


def test_defines(make):
    ctx = make(DEFINES, RMS_OK)
    result = run(ctx)
    assert result.passed and result.reason == "MyNorm(d_model=16) builds, 16 parameters"
    assert result.evidence == {"class": "MyNorm", "parameters": 16}

    wrong_name = make(DEFINES, RMS_OK.replace("MyNorm", "Other"))
    result = run(wrong_name)
    assert not result.passed
    assert result.reason == "class MyNorm is not defined in starter.py (it defines: Other)"

    broken = make(DEFINES, "class MyNorm(\n")
    assert run(broken).reason.startswith("starter.py line 1: ")

    raises = make(DEFINES, RMS_OK.replace("super().__init__()", "raise ValueError('nope')"))
    assert run(raises).reason.startswith("MyNorm(d_model=16) failed to build: ValueError: nope")

    needs = make(DEFINES, RMS_OK.replace("d_model, eps", "d_model, width, eps"))
    assert "needs a value for width" in run(needs).reason

    ctx.user_file.unlink()
    assert "does not exist: run `nanoscope learn start p/01-x` first" in run(ctx).reason


def test_defines_uses_lesson_args(make):
    ctx = make(DEFINES + "args = { d_model = 8 }\n", RMS_OK)
    assert run(ctx).reason == "MyNorm(d_model=8) builds, 8 parameters"


def test_equivalent(make):
    ctx = make(DEFINES + EQUIV, RMS_OK)
    result = run(ctx, 1)
    assert result.passed, result.reason
    assert result.reason.startswith("matches the reference rms_norm: max abs diff ")
    assert result.evidence["max_abs_diff"] <= 1e-5 and result.evidence["trials"] == 3

    # a subtle bug: forgets the epsilon-less mean (uses sum instead)
    sloppy = make(DEFINES + EQUIV, RMS_OK.replace(".mean(-1", ".sum(-1"))
    result = run(sloppy, 1)
    assert not result.passed
    assert "output differs from the reference: max abs diff" in result.reason
    assert "on inputs of shape [[2, 5, 16]]" in result.reason and "tolerance 1e-05" in result.reason
    assert result.evidence["max_abs_diff"] > 1e-5

    # several input shapes are all tried
    both = make(DEFINES + EQUIV.replace("[[2, 5, 16]]", "[[[2, 5, 16]], [[1, 3, 16]]]"), RMS_OK)
    assert run(both, 1).evidence["trials"] == 6
    shape_bug = make(DEFINES + EQUIV,
                     RMS_OK.replace("return x /", "return x[..., :8].repeat(1, 1, 2) /"))
    assert not run(shape_bug, 1).passed

    wrong_param = make(DEFINES + EQUIV, RMS_OK.replace("self.weight", "self.scale"))
    assert "has no parameter named 'weight' (it has: scale)" in run(wrong_param, 1).reason

    bad_reference = make(DEFINES + EQUIV.replace("rms_norm", "no_such_ref"), RMS_OK)
    assert "bug in the lesson" in run(bad_reference, 1).reason

    raises = make(DEFINES + EQUIV, RMS_OK.replace("return x /", "return 1 // 0 + x /"))
    assert "raised ZeroDivisionError" in run(raises, 1).reason


def test_equivalent_with_token_inputs_and_constants(make):
    code = textwrap.dedent('''
        import torch
        import torch.nn as nn

        class Emb(nn.Module):
            def __init__(self, d_model, vocab_size):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(vocab_size, d_model))

            def forward(self, ids):
                return self.weight[ids]
        ''')
    toml = textwrap.dedent('''
        [[checks]]
        id = "same"
        kind = "equivalent"
        class = "Emb"
        reference = "embed_one_hot"
        inputs = [{ shape = [2, 4], ints = 50 }]
        call = ["input:0", "param:weight"]
        ''')
    result = run(make(toml, code))
    assert result.passed, result.reason


def test_forbid(make):
    forbid = 'forbid = ["torch.nn.MultiheadAttention", "F.scaled_dot_product_attention"]'
    code = textwrap.dedent('''
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.nn import MultiheadAttention as MHA

        class A(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = nn.MultiheadAttention(8, 2)

            def forward(self, q, k, v):
                return F.scaled_dot_product_attention(q, k, v)
        ''')
    toml = '\n[[checks]]\nid = "own"\nkind = "forbid"\n'
    result = run(make(toml, code, extra=forbid))
    assert not result.passed
    assert result.reason == (
        "starter.py uses what this lesson asks you to build yourself: line 4: "
        "torch.nn.MultiheadAttention; line 9: torch.nn.MultiheadAttention; line 12: "
        "torch.nn.functional.scaled_dot_product_attention")
    assert [u["line"] for u in result.evidence["uses"]] == [4, 9, 12]
    clean = run(make(toml, "import torch\nx = torch.zeros(1)\n", extra=forbid))
    assert clean.passed
    assert "avoids the shortcuts this lesson forbids (torch.nn.Multihead" in clean.reason
    bare = run(make(toml, "import torch.nn.functional as F\nF.scaled_dot_product_attention\n",
                    extra='forbid = ["scaled_dot_product_attention"]'))
    assert not bare.passed and "line 2" in bare.reason
    assert "line 1" in run(make(toml, "def (:\n", extra=forbid)).reason


def test_registered_kinds():
    assert {"defines", "equivalent", "forbid"} <= set(CHECKERS)


BIGRAM = '''\
import torch.nn as nn


class MyBigram(nn.Module):
    def __init__(self, vocab_size, d_model=16):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        return self.head(self.emb(idx))
'''


@pytest.mark.usefixtures("fake_data")
def test_trains(make):
    from fakes import tiny

    from nanoscope.presets import register_preset

    register_preset(tiny(name="tinystories-5min"))  # the lesson's cpu preset, kept small here
    toml = ('\n[[checks]]\nid = "learns"\nkind = "trains"\nclass = "MyBigram"\n'
            'metric = "val_bpb"\nthreshold = {threshold}\n')
    ctx = make(toml.format(threshold=10), BIGRAM)
    result = run(ctx)
    assert result.passed, result.reason
    assert result.reason.startswith("val_bpb ")
    assert "after 20 steps on tinystories-5min" in result.reason
    assert "at or below the target 10" in result.reason
    assert result.evidence["threshold"] == 10.0 and len(result.evidence["runs"]) == 1

    hard = make(toml.format(threshold=0.01), BIGRAM)
    result = run(hard)
    assert not result.passed and "is above the target 0.01" in result.reason

    two = make(toml.format(threshold=10) + "seeds = 2\n", BIGRAM)
    assert "(mean of 2 seeds)" in run(two).reason

    bad_metric = make(toml.replace("val_bpb", "accuracy").format(threshold=1), BIGRAM)
    assert "metric 'accuracy'" in run(bad_metric).reason
    crash = make(toml.format(threshold=10), BIGRAM.replace("self.head(self.emb(idx))", "idx + 1"))
    assert run(crash).passed is False


DUMB = '''\
import torch
import torch.nn as nn


class Dumb(nn.Module):
    """Predicts every token with the same probability: it learns nothing."""

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.p = nn.Parameter(torch.zeros(1))

    def forward(self, idx):
        return torch.zeros(*idx.shape, self.vocab_size) + self.p
'''


@pytest.mark.usefixtures("fake_data")
def test_verdict(make):
    from fakes import tiny

    from nanoscope.presets import register_preset

    register_preset(tiny(name="tinystories-5min", max_steps=30))
    toml = ('\n[[checks]]\nid = "gap"\nkind = "verdict"\na = "Bigram"\nb = "Dumb"\n'
            'expect = "{expect}"\nseeds = 3\n')
    ctx = make(toml.format(expect="better"), DUMB)
    result = run(ctx)
    assert result.passed, result.reason
    assert result.reason.startswith("Bigram vs Dumb: better (val_bpb difference −") or \
        result.reason.startswith("Bigram vs Dumb: better (val_bpb difference -")
    assert result.reason.endswith("3 seeds), as the lesson expects")
    assert result.evidence["verdict"] == "better" and result.evidence["ci95"][1] < 0

    wrong = run(make(toml.format(expect="within noise"), DUMB))
    assert not wrong.passed and "the lesson expects 'within noise'" in wrong.reason
    assert "is zero inside it?" in wrong.reason

    near = make(toml.replace('b = "Dumb"', 'b = "Bigram"').format(expect="within noise")
                + "a_kwargs = { d_model = 16 }\nb_kwargs = { d_model = 17 }\n", DUMB)
    assert run(near).passed  # two nearly identical models: the interval includes zero
    twin = make(toml.replace('b = "Dumb"', 'b = "Bigram"').format(expect="within noise"), DUMB)
    result = run(twin)  # identical on every seed: exactly zero, no spread, still within noise
    assert result.passed and "identical on every seed" in result.reason
    assert result.evidence["degenerate"] is True
    twin_better = make(toml.replace('b = "Dumb"', 'b = "Bigram"').format(expect="better"), DUMB)
    assert "Are the two models really different?" in run(twin_better).reason

    two = make(toml.format(expect="better").replace("seeds = 3", "seeds = 2"), DUMB)
    assert "fewer than 3 seeds" in run(two).reason
    unknown = make(toml.replace("Dumb", "Nope").format(expect="better"), DUMB)
    assert "class Nope is not defined in starter.py (it defines: Dumb)" in run(unknown).reason
    bad = make(toml.format(expect="much better"), DUMB)
    assert "bug in the lesson" in run(bad).reason


def test_predicted(make, capsys):
    import hashlib

    from nanoscope.cli import main
    from nanoscope.learn import progress
    from nanoscope.learn.checks import prediction_path, run_one

    toml = ('\n[[checks]]\nid = "gap"\nkind = "verdict"\na = "A"\nb = "B"\nexpect = "better"\n'
            '[[checks]]\nid = "foresaw"\nkind = "predicted"\nquantity = "gap"\n')
    ctx = make(toml, "")
    lid = ctx.lesson.id
    gap = {"verdict": "better", "delta": -0.10, "ci95": [-0.14, -0.06]}
    ctx.shared["evidence"] = {"gap": gap}

    def predict(*flags):
        try:
            main(["learn", "predict", lid, *flags])
        except SystemExit as exc:
            return exc.code
        return 0

    assert "no prediction was recorded" in run_one(ctx, ctx.lesson.checks[1]).reason
    assert predict() == 2 and "say what you expect" in capsys.readouterr().out
    assert predict("--low", "0.2", "--high", "0.1") == 2
    assert "low must not be above high" in capsys.readouterr().out
    assert predict("--verdict", "better", "--low", "-0.2", "--high", "-0.05") == 0
    assert "recorded your prediction" in capsys.readouterr().out
    # before the experiment runs: scored against the CI
    progress.mark(lid, "checking")
    ok = run_one(ctx, ctx.lesson.checks[1])
    assert ok.passed, ok.reason
    assert ok.reason.startswith("your prediction was recorded first and holds: ")
    assert "the run found -0.100" in ok.reason
    assert ok.evidence["hit"] is True and ok.evidence["verdict_right"] is True

    # a wrong guess says how
    prediction_path(ctx.lesson).write_text('verdict = "worse"\nlow = 0.0\nhigh = 0.1\n')
    progress.record_prediction(lid, prediction_path(ctx.lesson), hashlib.sha256(
        prediction_path(ctx.lesson).read_bytes()).hexdigest())
    entry = progress.entry(lid)
    assert entry["first_checked_at"] is not None
    late = run_one(ctx, ctx.lesson.checks[1])  # recorded after the first check began
    assert not late.passed and "after the experiment first ran" in late.reason
    assert predict("--verdict", "better") == 1  # and the CLI refuses outright
    assert "too late for this lesson" in capsys.readouterr().out


def test_predicted_scores_and_tamper_checks(make):
    import hashlib

    from nanoscope.learn import progress
    from nanoscope.learn.checks import prediction_path, run_one

    toml = ('\n[[checks]]\nid = "foresaw"\nkind = "predicted"\nquantity = "gap"\n')
    ctx = make(toml, "")
    lid, file = ctx.lesson.id, prediction_path(ctx.lesson)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text('verdict = "better"\nlow = -0.5\nhigh = 0.5\n')
    progress.record_prediction(lid, file, hashlib.sha256(file.read_bytes()).hexdigest())
    ctx.shared["evidence"] = {"gap": {"verdict": "better", "delta": -0.10,
                                      "ci95": [-0.14, -0.06]}}
    progress.mark(lid, "checking")
    vague = run_one(ctx, ctx.lesson.checks[0])
    assert not vague.passed and "far wider than the run's own uncertainty" in vague.reason
    file.write_text('verdict = "better"\nlow = -0.5\nhigh = -0.4\n')  # edited after recording
    assert "was edited after it was recorded" in run_one(ctx, ctx.lesson.checks[0]).reason
    ctx.shared["evidence"] = {}
    file.write_text('verdict = "better"\nlow = -0.5\nhigh = 0.5\n')
    assert "nothing to score against" in run_one(ctx, ctx.lesson.checks[0]).reason
    # the scoring function itself
    miss = score_prediction(0.3, (0.2, 0.4), low=-0.1, high=0.1)
    assert miss["passed"] is False and miss["hit"] is False
    assert score_prediction(0.3, (0.2, 0.4), verdict="worse", actual_verdict="worse")["passed"]
    assert score_prediction(0.3, None, low=0.0, high=1.0)["sharp"] is None
    assert score_prediction(0.3, None)["passed"] is False


from nanoscope.statistics import score_prediction  # noqa: E402


@pytest.mark.usefixtures("fake_data")
def test_reproduces(make, tmp_path, monkeypatch):
    from fakes import tiny

    from nanoscope.models import Bigram
    from nanoscope.presets import register_preset
    from nanoscope.run import run as train
    from nanoscope.statistics import reproduction_interval

    preset = tiny(name="tinystories-5min")
    register_preset(preset)
    import sys

    compare = sys.modules["nanoscope.compare"]  # nanoscope.compare the name is the function
    baselines = tmp_path / "baselines"
    monkeypatch.setattr(compare, "BASELINES_DIR", baselines)
    group = train(Bigram, preset, seeds=3, device="cpu", progress=False)
    compare.export_baseline(group, baselines / "tinystories-5min" / "bigram")
    values = [r.val_bpb[-1][1] for r in group.results]
    low, high = reproduction_interval(values)
    assert low < sum(values) / 3 < high and reproduction_interval(values[:2]) is None

    toml = ('\n[[checks]]\nid = "same"\nkind = "reproduces"\nbaseline = "bigram"\n'
            'class = "MyBigram"\nkwargs = {kwargs}\n')
    # the learner's bigram has the shipped Bigram's layers (d_model 32 by default) in other names
    mine = BIGRAM.replace("d_model=16", "d_model=32")
    result = run(make(toml.format(kwargs="{}"), mine))
    assert result.passed, result.reason
    assert "is inside the range" in result.reason and "you reproduced it" in result.reason

    broken = run(make(toml.format(kwargs="{}"), mine.replace("self.head(self.emb(idx))",
                                                             "self.head(self.emb(idx)) * 0")))
    assert not broken.passed and "worse than the bigram baseline's seeds" in broken.reason
    assert broken.evidence["interval"] == pytest.approx([low, high])

    missing = run(make(toml.replace('"bigram"', '"nope"').format(kwargs="{}"), mine))
    assert "no runs named 'nope'" in missing.reason


@pytest.mark.usefixtures("fake_data")
def test_visible_reasons(make, tmp_path, monkeypatch):
    """Every kind of check says, in words, why it passed and why it failed."""
    import hashlib
    import sys

    from fakes import tiny

    from nanoscope.learn import progress
    from nanoscope.learn.checks import prediction_path
    from nanoscope.models import Bigram
    from nanoscope.presets import register_preset
    from nanoscope.run import run as train

    preset = tiny(name="tinystories-5min", max_steps=30)
    register_preset(preset)
    compare = sys.modules["nanoscope.compare"]
    monkeypatch.setattr(compare, "BASELINES_DIR", tmp_path / "baselines")
    compare.export_baseline(train(Bigram, preset, seeds=3, device="cpu", progress=False),
                            tmp_path / "baselines" / "tinystories-5min" / "bigram")
    mine = BIGRAM.replace("d_model=16", "d_model=32")
    cases = {  # kind -> (check toml, good code, bad code)
        "defines": (DEFINES, RMS_OK, "class Other: pass\n"),
        "equivalent": (DEFINES + EQUIV, RMS_OK, RMS_OK.replace(".mean(-1", ".sum(-1")),
        "forbid": ('\n[[checks]]\nid = "own"\nkind = "forbid"\n', "x = 1\n",
                   "import torch.nn.functional as F\nF.softmax\n"),
        "trains": ('\n[[checks]]\nid = "t"\nkind = "trains"\nclass = "MyBigram"\n'
                   'metric = "val_bpb"\nthreshold = {t}\n', mine, mine),
        "verdict": ('\n[[checks]]\nid = "v"\nkind = "verdict"\na = "Bigram"\nb = "Dumb"\n'
                    'expect = "{e}"\n', DUMB, DUMB),
        "reproduces": ('\n[[checks]]\nid = "r"\nkind = "reproduces"\nbaseline = "bigram"\n'
                       'class = "MyBigram"\n', mine, mine.replace("self.head(self.emb(idx))",
                                                                  "self.head(self.emb(idx)) * 0")),
    }
    forbid_extra = 'forbid = ["F.softmax"]'
    reasons = {}
    for kind, (toml, good, bad) in cases.items():
        extra = forbid_extra if kind == "forbid" else ""
        thresholds = {"t": ("10", "0.01"), "e": ("better", "worse")}
        for outcome, code in (("pass", good), ("fail", bad)):
            body = toml
            if "{t}" in body:
                body = body.format(t=thresholds["t"][outcome == "fail"])
            if "{e}" in body:
                body = body.format(e=thresholds["e"][outcome == "fail"])
            ctx = make(body, code, extra=extra)
            result = run(ctx, len(ctx.lesson.checks) - 1)
            assert result.passed == (outcome == "pass"), (kind, outcome, result.reason)
            reasons[(kind, outcome)] = result.reason

    toml = ('\n[[checks]]\nid = "gap"\nkind = "verdict"\na = "A"\nb = "B"\nexpect = "better"\n'
            '[[checks]]\nid = "foresaw"\nkind = "predicted"\nquantity = "gap"\n')
    ctx = make(toml, "")
    ctx.shared["evidence"] = {"gap": {"verdict": "better", "delta": -0.1, "ci95": [-0.14, -0.06]}}
    reasons[("predicted", "fail")] = run(ctx, 1).reason  # nothing recorded yet
    file = prediction_path(ctx.lesson)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text('verdict = "better"\nlow = -0.2\nhigh = -0.05\n')
    progress.record_prediction(ctx.lesson.id, file, hashlib.sha256(file.read_bytes()).hexdigest())
    progress.mark(ctx.lesson.id, "checking")
    reasons[("predicted", "pass")] = run(ctx, 1).reason

    assert {k for k, _ in reasons} == set(cases) | {"predicted"}
    for (kind, outcome), reason in reasons.items():
        assert len(reason.split()) >= 4, (kind, outcome, reason)  # words, not a code
        assert "Traceback" not in reason and "<" not in reason.split()[0], (kind, reason)
        if outcome == "fail":  # a failure says what is wrong or what to do about it
            assert any(word in reason for word in (
                "not", "differs", "above", "outside", "no ", "uses", "expects", "worse",
                "failed")), (kind, reason)
