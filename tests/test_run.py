import json
import math
import os
import signal

import pytest
import torch
from fakes import tiny

from nanoscope import run
from nanoscope.dataset import load_data, load_tokenizer
from nanoscope.models import Bigram
from nanoscope.tokenizer import BPETokenizer, ByteTokenizer

pytestmark = pytest.mark.usefixtures("fake_data")


def test_run_trains_evaluates_and_samples():
    result = run(Bigram, tiny(), device="cpu")

    assert result.final_step == 20
    assert [s for s, _ in result.val_losses] == [10, 20]
    assert result.val_losses[-1][1] < math.log(result.data.tokenizer.vocab_size)
    assert all(bpb > 0 for _, bpb in result.val_bpb)
    assert (result.run_dir / "samples.txt").read_text().count("--- step") == 2
    config = json.loads((result.run_dir / "config.json").read_text())
    assert config["tokenizer"] == "bpe-300-100"
    assert config["stats"]["vocab_size"] == result.data.tokenizer.vocab_size
    assert isinstance(result.generate("Once upon", max_new_tokens=5), str)


def test_rerunning_a_finished_run_loads_it():
    first = run(Bigram, tiny(), device="cpu")
    again = run(Bigram, tiny(), device="cpu")

    assert again.final_step == 20
    assert again.metrics == first.metrics
    assert again.model.state_dict()["head.weight"].equal(first.model.state_dict()["head.weight"])


def test_interrupted_run_resumes_exactly():
    straight = run(Bigram, tiny(), device="cpu", output_dir="straight")

    def interrupt(step, row):
        if step == 7:
            os.kill(os.getpid(), signal.SIGINT)

    stopped = run(Bigram, tiny(), device="cpu", output_dir="resumed", on_step=interrupt)
    assert stopped.train_result.stopped_early and stopped.final_step == 7
    resumed = run(Bigram, tiny(), device="cpu", output_dir="resumed")

    assert [r["loss"] for r in resumed.metrics] == [r["loss"] for r in straight.metrics]
    lines = (resumed.run_dir / "metrics.jsonl").read_text().splitlines()
    assert [json.loads(line)["step"] for line in lines] == list(range(1, 21))


def test_overrides_get_their_own_run_dir():
    default = run(Bigram, tiny(max_steps=2), device="cpu")
    explicit_default = run(Bigram, tiny(max_steps=2), device="cpu", d_model=32)
    smaller = run(Bigram, tiny(max_steps=2), device="cpu", d_model=16)
    other_lr = run(Bigram, tiny(max_steps=2), device="cpu", learning_rate=1e-3)

    assert default.run_dir == explicit_default.run_dir
    assert len({default.run_dir, smaller.run_dir, other_lr.run_dir}) == 3
    assert smaller.model.token_embedding.embedding_dim == 16
    assert other_lr.preset.learning_rate == 1e-3


def test_changed_config_in_same_dir_refuses_to_resume():
    run(Bigram, tiny(max_steps=2), device="cpu", output_dir="out")
    with pytest.raises(ValueError, match="learning_rate|preset"):
        run(Bigram, tiny(max_steps=2), device="cpu", output_dir="out", learning_rate=1e-3)
    restarted = run(Bigram, tiny(max_steps=2), device="cpu", output_dir="out",
                    learning_rate=1e-3, resume=False)
    assert restarted.final_step == 2


def test_unknown_keyword_is_an_error():
    with pytest.raises(TypeError, match="neither a parameter"):
        run(Bigram, tiny(), device="cpu", d_modle=64)


def test_bpe_tokenizer_trains_once_and_round_trips(fake_data):
    tok = load_tokenizer(tiny())
    assert isinstance(tok, BPETokenizer)
    assert tok.vocab_size <= 300
    text = "Once upon a time, the fox saw the frog."
    assert tok.decode(tok.encode(text)) == text
    assert len(tok.encode(text)) < len(text.encode())

    n_calls = len(fake_data)
    load_tokenizer(tiny())
    assert len(fake_data) == n_calls


def test_vocab_size_override_trains_a_new_tokenizer():
    small = load_data(tiny(vocab_size=280)).tokenizer
    large = load_data(tiny(vocab_size=400)).tokenizer
    assert small.vocab_size < large.vocab_size


def test_bits_per_byte_matches_byte_tokenizer():
    data = load_data(tiny(tokenizer="bytes", vocab_size=None))
    assert isinstance(data.tokenizer, ByteTokenizer)
    # one token per byte, plus one EOS per document
    assert len(data.val) == data.val_bytes + 10
    assert data.bits_per_byte(math.log(2)) == pytest.approx(len(data.val) / data.val_bytes)


def test_token_files_hold_documents_separated_by_eos():
    data = load_data(tiny())
    tokens = torch.from_numpy(data.val.astype("int64"))
    eos = data.tokenizer.eos_token_id
    assert int((tokens == eos).sum()) == 10
    first_doc = tokens[: int((tokens == eos).nonzero()[0])].tolist()
    assert data.tokenizer.decode(first_doc).startswith("Once upon a time")


def test_a_seed_gives_the_same_initial_weights_whatever_ran_before(fake_data):
    first = run(Bigram, tiny(), output_dir="a", seed=3, progress=False)
    run(Bigram, tiny(), output_dir="other", seed=9, progress=False)  # disturbs the global RNG
    again = run(Bigram, tiny(), output_dir="b", seed=3, progress=False)
    assert again.metrics[0]["loss"] == first.metrics[0]["loss"]
