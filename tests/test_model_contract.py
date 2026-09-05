import torch

from nanoscope.model import LMOutput, build_model


def test_toy_model_implements_contract() -> None:
    model, _ = build_model("toy_lm", {"vocab_size": 17, "hidden_size": 8})
    output = model(torch.zeros((2, 4), dtype=torch.long))
    assert isinstance(output, LMOutput)
    assert output.logits.shape == (2, 4, 17)

