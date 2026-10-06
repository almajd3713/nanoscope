"""The frozen reference models: what the rebuilt library models are checked against."""

import math

import pytest
import torch
import torch.nn.functional as F

from nanoscope.reference.gpt2_ref import GPT2Ref
from nanoscope.reference.modern_ref import ModernRef

ATOL = 1e-5
SMALL = dict(vocab_size=50, context_length=16, d_model=32, n_layers=2, n_heads=4)


def logits_of(out):
    return out if isinstance(out, torch.Tensor) else out[0]


@pytest.mark.parametrize("ref", [GPT2Ref, ModernRef])
def test_frozen_models_pass_existing_checks(ref):
    model = ref(**SMALL)
    idx = torch.randint(50, (2, 16))
    changed = idx.clone()
    changed[:, 10:] = torch.randint(50, (2, 6))
    a, b = logits_of(model(idx)), logits_of(model(changed))
    torch.testing.assert_close(a[:, :10], b[:, :10], atol=ATOL, rtol=0)  # causal
    assert not torch.allclose(a[:, 10:], b[:, 10:])
    assert model.head.weight is model.tok_emb.weight  # tied

    big = ref(vocab_size=512)
    logits = logits_of(big(torch.randint(512, (4, 32))))
    loss = F.cross_entropy(logits.reshape(-1, 512), torch.randint(512, (4 * 32,)))
    assert abs(loss.item() - math.log(512)) < 0.5  # near uniform before training
    assert big.flops_per_token(32) > 0


def test_reference_modules_are_independent():
    """They import only torch, math, __future__ and each other, so nothing in the library can
    change what they mean."""
    import ast
    from pathlib import Path

    import nanoscope.reference as package

    allowed = {"torch", "math", "__future__"}
    files = sorted(Path(package.__file__).parent.glob("*.py"))
    assert len(files) >= 4
    for file in files:
        for node in ast.walk(ast.parse(file.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                ok = name.split(".")[0] in allowed or name.startswith("nanoscope.reference")
                assert ok, f"{file.name} imports {name}"


def test_hand_computed_embedding_position_and_head():
    from nanoscope.reference import functional as R

    table = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert R.embed_one_hot(torch.tensor([2, 0]), table).tolist() == [[5.0, 6.0], [1.0, 2.0]]
    positions = torch.tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
    assert R.add_learned_position(torch.zeros(1, 2, 2), positions).tolist() == [
        [[10.0, 20.0], [30.0, 40.0]]]
    head = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    assert R.tied_head(torch.tensor([[1.0, 1.0]]), head).tolist() == [[1.0, 1.0, 4.0]]


def test_hand_computed_attention_parts():
    from nanoscope.reference import functional as R

    assert R.causal_mask(3).tolist() == [
        [True, False, False], [True, True, False], [True, True, True]]
    assert R.softmax(torch.tensor([0.0, 0.0])).tolist() == [0.5, 0.5]
    torch.testing.assert_close(R.softmax(torch.tensor([0.0, math.log(3)])),
                               torch.tensor([0.25, 0.75]))
    assert R.softmax(torch.tensor([1000.0, 1000.0])).tolist() == [0.5, 0.5]  # no overflow
    weights = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    assert R.weighted_sum(weights, torch.tensor([[2.0], [4.0]])).tolist() == [[2.0], [3.0]]
    kv = torch.tensor([[1.0], [2.0]])
    assert R.repeat_kv_heads(kv, 4).flatten().tolist() == [1.0, 1.0, 2.0, 2.0]
    with pytest.raises(AssertionError, match="multiple"):
        R.repeat_kv_heads(torch.zeros(3, 1), 4)


def test_hand_computed_qk_norm_and_z_loss():
    from nanoscope.reference import functional as R

    # rms of (3, 4) is sqrt((9 + 16) / 2) = sqrt(12.5)
    out = R.qk_norm(torch.tensor([3.0, 4.0]), torch.ones(2), eps=0.0)
    torch.testing.assert_close(out, torch.tensor([3.0, 4.0]) / math.sqrt(12.5))
    # equal logits over 2 tokens: log Z = ln 2
    z = R.z_loss(torch.zeros(1, 2), 0.5)
    torch.testing.assert_close(z, torch.tensor(0.5 * math.log(2) ** 2))
    assert R.z_loss(torch.zeros(3, 2), 0.0).item() == 0.0
