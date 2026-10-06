"""BlockStats: per-block numbers at eval steps, through a hook only."""

import json
import math
from pathlib import Path

import pytest
import torch
from fakes import tiny
from helpers import assert_valid

from nanoscope import run
from nanoscope.blockstats import FILE, read_blockstats
from nanoscope.models import GPT2, Bigram, Modern

pytestmark = pytest.mark.usefixtures("fake_data")
SMALL = dict(n_layers=2, d_model=32, n_heads=4)


def test_block_stats_are_written_at_eval_steps():
    result = run(Modern, tiny(), device="cpu", block_stats=True, progress=False, **SMALL)
    lines = [json.loads(x) for x in (result.run_dir / FILE).read_text().splitlines()]
    assert [line["step"] for line in lines] == [10, 20]  # eval_interval=10, max_steps=20
    for line in lines:
        assert_valid("blockstats", line)
        assert [b["name"] for b in line["blocks"]] == ["blocks.0", "blocks.1"]
        for block in line["blocks"]:
            assert block["activation_rms"] > 0 and block["grad_norm"] > 0
            # an untrained-ish model attends over at most T positions: entropy below log(T)
            assert 0 < block["attention_entropy"] <= math.log(32)
    assert lines[0]["val_loss"] == pytest.approx(result.val_losses[0][1])
    first, second = lines[0]["blocks"], lines[1]["blocks"]
    assert all(b["update_to_weight"] is None for b in first)  # nothing to compare with yet
    assert all(0 < b["update_to_weight"] < 1 for b in second)
    assert [x["step"] for x in read_blockstats(result.run_dir)] == [10, 20]


def test_block_stats_are_off_by_default_and_leave_training_unchanged():
    plain = run(Modern, tiny(), device="cpu", progress=False, output_dir=Path("plain"), **SMALL)
    assert not (plain.run_dir / FILE).exists()
    assert read_blockstats(plain.run_dir) == []
    watched = run(Modern, tiny(), device="cpu", progress=False, block_stats=True,
                  output_dir=Path("watched"), **SMALL)
    assert [loss for _, loss in watched.val_losses] == [loss for _, loss in plain.val_losses]


def test_models_without_attention_or_blocks_still_get_stats():
    result = run(Bigram, tiny(), device="cpu", block_stats=True, progress=False)
    (first, _) = read_blockstats(result.run_dir)
    (only,) = first["blocks"]
    assert only["name"] == "model" and only["attention_entropy"] is None


def test_hook_does_not_touch_the_training_gradients():
    from nanoscope.blockstats import BlockStats
    from nanoscope.dataset import load_data

    preset = tiny()
    data = load_data(preset)
    model = GPT2(vocab_size=data.tokenizer.vocab_size, context_length=32, **SMALL)
    out = model(torch.zeros(2, 32, dtype=torch.long))
    out.sum().backward()
    before = [p.grad.clone() for p in model.parameters()]
    stats = BlockStats(model, data, preset, Path("."), torch.device("cpu"))
    stats(10, 1.0)
    assert model.training
    for p, g in zip(model.parameters(), before, strict=True):
        assert torch.equal(p.grad, g)


def test_train_loop_knows_nothing_about_block_stats():
    import nanoscope.train_loop as loop

    assert "blockstats" not in Path(loop.__file__).read_text().lower()
    assert "block_stats" not in Path(loop.__file__).read_text().lower()
