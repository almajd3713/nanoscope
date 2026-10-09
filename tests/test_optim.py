import pytest
import torch
from fakes import tiny

from nanoscope import run
from nanoscope.models import Bigram
from nanoscope.optim import muon
from nanoscope.optim.muon import orthogonalize

pytestmark = pytest.mark.usefixtures("fake_data")


def test_orthogonalize_flattens_singular_values():
    torch.manual_seed(0)
    g = torch.randn(16, 32) * torch.linspace(0.01, 5, 16).unsqueeze(1)
    before = torch.linalg.svdvals(g)
    after = torch.linalg.svdvals(orthogonalize(g))
    assert before.max() / before.min() > 50
    assert after.max() < 1.5 and after.max() / after.min() < 6


def test_muon_trains_a_tiny_model():
    result = run(Bigram, tiny(max_steps=30, learning_rate=2e-3), device="cpu", optimizer=muon)
    losses = result.train_losses
    assert losses[-1] < losses[0]
    assert "muon" in (result.run_dir / "config.json").read_text()
