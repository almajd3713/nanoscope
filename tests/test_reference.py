"""The frozen reference models: what the rebuilt library models are checked against."""

import math

import pytest
import torch
import torch.nn.functional as F

from nanoscope.models import GPT2, Modern
from nanoscope.reference.gpt2_ref import GPT2Ref
from nanoscope.reference.modern_ref import ModernRef

ATOL = 1e-5
SMALL = dict(vocab_size=50, context_length=16, d_model=32, n_layers=2, n_heads=4)
PAIRS = [(GPT2Ref, GPT2), (ModernRef, Modern)]


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


@pytest.mark.parametrize(("ref", "current"), PAIRS)
def test_frozen_models_equal_the_current_models_today(ref, current):
    """Until the rebuild changes the models, a reference and its model are the same function."""
    reference, model = ref(**SMALL), current(**SMALL)
    model.load_state_dict(reference.state_dict())
    idx = torch.randint(50, (2, 16))
    torch.testing.assert_close(logits_of(model(idx)), logits_of(reference(idx)), atol=ATOL, rtol=0)
    assert model.flops_per_token(16) == reference.flops_per_token(16)
