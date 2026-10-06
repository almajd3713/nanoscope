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
    assert clean.passed and "avoids torch.nn.MultiheadAttention" in clean.reason
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
