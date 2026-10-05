import json
import math
import os
import signal

import pytest
import torch
from fakes import tiny

from nanoscope import paths, run, store
from nanoscope.dataset import load_data, load_tokenizer
from nanoscope.models import Bigram
from nanoscope.tokenizer import BPETokenizer, ByteTokenizer

pytestmark = pytest.mark.usefixtures("fake_data")


def out(name):
    return paths.runs_dir() / name


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
    straight = run(Bigram, tiny(), device="cpu", output_dir=out("straight"))

    def interrupt(step, row):
        if step == 7:
            os.kill(os.getpid(), signal.SIGINT)

    stopped = run(Bigram, tiny(), device="cpu", output_dir=out("resumed"), on_step=interrupt)
    assert stopped.train_result.stopped_early and stopped.final_step == 7
    resumed = run(Bigram, tiny(), device="cpu", output_dir=out("resumed"))

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
    run(Bigram, tiny(max_steps=2), device="cpu", output_dir=out("out"))
    with pytest.raises(ValueError, match="learning_rate|preset"):
        run(Bigram, tiny(max_steps=2), device="cpu", output_dir=out("out"), learning_rate=1e-3)
    restarted = run(Bigram, tiny(max_steps=2), device="cpu", output_dir=out("out"),
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
    first = run(Bigram, tiny(), output_dir=out("a"), seed=3, progress=False)
    run(Bigram, tiny(), output_dir=out("other"), seed=9, progress=False)  # disturbs the global RNG
    again = run(Bigram, tiny(), output_dir=out("b"), seed=3, progress=False)
    assert again.metrics[0]["loss"] == first.metrics[0]["loss"]


def test_results_know_their_ref():
    single = run(Bigram, tiny(), device="cpu", progress=False)
    group = run(Bigram, tiny(), device="cpu", seeds=2, progress=False, d_model=8)
    assert single.ref == single.run_dir.relative_to(paths.runs_dir()).as_posix()
    assert single.summary()["ref"] == single.ref
    assert group.ref == group[0].ref.removesuffix("/seed-0")
    assert group.summary()["ref"] == group.ref
    assert store.resolve(group.ref) == group[0].run_dir.parent


def test_config_records_its_schema_and_version():
    from helpers import assert_valid

    import nanoscope

    result = run(Bigram, tiny(), device="cpu", progress=False)
    config = json.loads((result.run_dir / "config.json").read_text())
    assert config["schema"] == 1 and config["nanoscope"] == nanoscope.__version__
    assert_valid("config", config)


def test_v0_config_without_schema_keys_still_resumes():
    first = run(Bigram, tiny(), device="cpu", output_dir=out("old"), progress=False)
    path = first.run_dir / "config.json"
    config = json.loads(path.read_text())
    del config["schema"], config["nanoscope"]  # what runs written before schemas look like
    path.write_text(json.dumps(config))
    again = run(Bigram, tiny(), device="cpu", output_dir=out("old"), progress=False)
    assert again.final_step == first.final_step


def test_train_stops_early_with_a_checkpoint_when_should_stop_says_so():
    from nanoscope.train_loop import train

    done = run(Bigram, tiny(), device="cpu", output_dir=out("source"), progress=False)
    result = train(done.model, done.data, done.preset, out("stopper"), torch.device("cpu"),
                   should_stop=lambda: True)
    assert result.stopped_early and result.final_step == 1
    assert (out("stopper") / "latest.json").exists()


STOP_SCRIPT = """
import sys, time
from pathlib import Path
sys.path.insert(0, {tests!r})
from fakes import tiny
from nanoscope import run
from nanoscope.models import Bigram

def slow(step, row):
    time.sleep(0.02)
    if step == 3:
        Path({marker!r}).write_text("started")

run(Bigram, tiny(max_steps=200, eval_interval=50, checkpoint_interval=100), device="cpu",
    output_dir={out!r}, on_step=slow, progress=False)
"""


def test_stop_then_resume_matches_an_uninterrupted_run(tmp_path):
    import subprocess
    import sys
    import time
    from pathlib import Path

    from nanoscope.cli import main

    tuned = dict(max_steps=200, eval_interval=50, checkpoint_interval=100)
    straight = run(Bigram, tiny(**tuned), device="cpu", output_dir=out("straight"),
                   progress=False)  # also caches the data the subprocess will read

    marker, victim = tmp_path / "started", out("victim")
    script = tmp_path / "victim.py"
    script.write_text(STOP_SCRIPT.format(
        tests=str(Path(__file__).parent), marker=str(marker), out=str(victim)))
    proc = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 120
        while not marker.exists() and time.time() < deadline and proc.poll() is None:
            time.sleep(0.05)
        assert marker.exists(), "the run never reached step 3"
        main(["stop", store.ref_of(victim)])
        assert proc.wait(timeout=60) == 0
    finally:
        if proc.poll() is None:
            proc.kill()

    doc = json.loads((victim / "status.json").read_text())
    assert doc["state"] == "cancelled" and 3 <= doc["step"] < 200
    assert (victim / "latest.json").exists()

    resumed = run(Bigram, tiny(**tuned), device="cpu", output_dir=victim, progress=False)
    assert json.loads((victim / "status.json").read_text())["state"] == "done"
    assert not (victim / "STOP").exists()
    assert [r["loss"] for r in resumed.metrics] == [r["loss"] for r in straight.metrics]


MODEL_FILE = """
import torch.nn as nn
from nanoscope.models import Bigram


class FromFile(Bigram):
    pass  # {marker}
"""


def test_model_ref_is_recorded_in_config(tmp_path):
    from helpers import assert_valid

    from nanoscope.cli import _load_model_class

    result = run(Bigram, tiny(), device="cpu", progress=False)
    model = json.loads((result.run_dir / "config.json").read_text())["model"]
    assert model["ref"] == "nanoscope.models.bigram:Bigram" and model["rebuildable"] is True
    assert len(model["source_sha256"]) == 64

    path = tmp_path / "mine.py"
    path.write_text(MODEL_FILE.format(marker="v1"))
    custom = run(_load_model_class(f"{path}:FromFile"), tiny(), device="cpu", progress=False)
    config = json.loads((custom.run_dir / "config.json").read_text())
    assert config["model"]["ref"] == f"{path.resolve()}:FromFile"
    assert config["model"]["rebuildable"] is True
    assert_valid("config", config)


def test_a_changed_model_source_is_logged_as_drift_and_still_resumes(tmp_path, capsys):
    from nanoscope.cli import _load_model_class

    path = tmp_path / "mine.py"
    path.write_text(MODEL_FILE.format(marker="v1"))
    run(_load_model_class(f"{path}:FromFile"), tiny(), device="cpu", output_dir=out("drift"),
        progress=False)
    path.write_text(MODEL_FILE.format(marker="v2"))  # edited after the run
    capsys.readouterr()
    again = run(_load_model_class(f"{path}:FromFile"), tiny(), device="cpu",
                output_dir=out("drift"), progress=False)
    assert again.final_step == 20  # it resumed (nothing left to train), it did not refuse
    assert "the source of FromFile changed since this run started" in capsys.readouterr().out


def test_main_class_source_is_saved_and_not_rebuildable(monkeypatch):
    class Local(Bigram):
        pass

    Local.__module__ = "__main__"
    import sys

    monkeypatch.setattr(sys.modules["nanoscope.run"], "class_source",
                        lambda cls: "class Local: ...\n")
    result = run(Local, tiny(), device="cpu", progress=False)
    model = json.loads((result.run_dir / "config.json").read_text())["model"]
    assert model["rebuildable"] is False and model["ref"].startswith("__main__:")
    assert model["ref"].endswith("Local")
    assert (result.run_dir / "model_source.py").read_text() == "class Local: ...\n"
