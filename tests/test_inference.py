from __future__ import annotations

import copy
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from nanoscope.config import dump_config, load_config
from nanoscope.data.tokenizer import ByteTokenizer, GPT2Tokenizer, tokenizer_identity
from nanoscope.inference import GenerationSession, InferenceConfig, load_inference_config
from nanoscope.inference.runner import read_prompts, resolve_checkpoint, sample_token
from nanoscope.model import LMOutput, build_model
from nanoscope.train.checkpoint import CheckpointManager

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def saved(tmp_path):
    training = load_config(ROOT / "configs/test/m0/local-smoke.yaml")
    training.run.id = "generation-test"
    training.data.sequence_length = 4
    training.model.params.update(vocab_size=257, hidden_size=257, dropout=0.9)
    model, _ = build_model(training.model.name, training.model.params)
    weights = model.state_dict()
    weights["token_embedding.weight"] = torch.eye(257)
    weights["projection.weight"] = torch.full((257, 257), -20.0)
    for token in range(257):
        weights["projection.weight"][(token + 1) % 257, token] = 20.0
    state = {"step": 4, "model": weights, "config": training.to_dict()}
    metadata = {
        "step": 4,
        "run_id": training.run.id,
        "config_digest": training.digest,
        "tokenizer": tokenizer_identity(training.tokenizer),
    }
    run = tmp_path / "run"
    checkpoint = CheckpointManager(run, 3).save(4, state, metadata)
    return training, state, metadata, run, checkpoint


def test_decode_roundtrip_and_eos():
    byte = ByteTokenizer()
    text = "Hello, café 🌍"
    assert byte.decode(byte.encode(text) + [256]) == text
    assert byte.decode([0xC3]) == "\ufffd"
    assert byte.decode([256]) == ""


def test_gpt2_decode_uses_tiktoken_without_downloading_assets(monkeypatch):
    import tiktoken

    encoding = tiktoken.Encoding(
        name="local-test",
        pat_str=r"(?s).",
        mergeable_ranks={bytes([value]): value for value in range(256)},
        special_tokens={"<|endoftext|>": 50256},
    )
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: encoding)
    tokenizer = GPT2Tokenizer()
    text = "Hello, café 🌍"
    assert tokenizer.decode(tokenizer.encode(text) + [tokenizer.eos_token_id]) == text


@pytest.mark.network
def test_gpt2_decode_real_vocabulary():
    tokenizer = GPT2Tokenizer()
    text = "Hello, café 🌍"
    assert tokenizer.decode(tokenizer.encode(text) + [tokenizer.eos_token_id]) == text


def test_config_aliases_relative_paths_and_templates(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").touch()
    directory = tmp_path / "configs/inference"
    directory.mkdir(parents=True)
    source = directory / "sample.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "run_dir": "@/runs/base",
                "checkpoint": "../saved/step_00000004",
                "training_config": "@/resolved.yaml",
                "output": "outputs/samples.jsonl",
            }
        )
    )
    monkeypatch.chdir("/")
    config = load_inference_config(source)
    assert config.run_dir == tmp_path / "runs/base"
    assert config.checkpoint == tmp_path / "configs/saved/step_00000004"
    assert config.training_config == tmp_path / "resolved.yaml"
    assert config.output == directory / "outputs/samples.jsonl"
    for path in [
        ROOT / "configs/inference-template.yaml",
        *sorted((ROOT / "configs/test/inference").glob("*.yaml")),
    ]:
        load_inference_config(path)
    source.write_text("run_dir: ./run\ntemprature: 1\n")
    with pytest.raises(ValueError, match="unknown"):
        load_inference_config(source)


@pytest.mark.parametrize(
    "values",
    [
        {"max_new_tokens": 0},
        {"max_new_tokens": True},
        {"seed": -1},
        {"seed": 2**63},
        {"seed": False},
        {"temperature": -0.1},
        {"temperature": float("nan")},
        {"temperature": float("inf")},
        {"temperature": "warm"},
        {"top_p": 0},
        {"top_p": 1.1},
        {"top_p": True},
        {"device": "metal"},
        {"checkpoint": ""},
    ],
)
def test_invalid_generation_settings(values):
    with pytest.raises(ValueError):
        InferenceConfig(run_dir=Path("/unused"), **values)


def test_latest_skips_corrupt_and_does_not_write_input(saved, tmp_path):
    _, _, _, run, checkpoint = saved
    incomplete = run / "checkpoints/step_00000008"
    incomplete.mkdir()
    (incomplete / "manifest.json").write_text('{"schema_version": 1, "files": {}}')
    before = {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    for path in run.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    session = GenerationSession(InferenceConfig(run_dir=run, device="cpu"))
    assert session.identity["checkpoint"]["path"] == str(checkpoint)
    session.generate("abc", max_new_tokens=2)
    assert before == {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="corrupt"):
        resolve_checkpoint(InferenceConfig(checkpoint=incomplete))
    absent = tmp_path / "missing"
    with pytest.raises(ValueError, match="no complete"):
        resolve_checkpoint(InferenceConfig(run_dir=absent))
    assert not absent.exists()
    with pytest.raises(ValueError, match="run_dir"):
        InferenceConfig()


def test_greedy_generation_slides_context_and_appends_results(saved, tmp_path):
    _, _, _, run, _ = saved
    destination = tmp_path / "results/samples.jsonl"
    rng_before = torch.random.get_rng_state().clone()
    session = GenerationSession(
        InferenceConfig(
            run_dir=run, device="cpu", temperature=0, max_new_tokens=6, output=destination
        )
    )
    inputs = []

    def capture(model, args):
        assert not model.training and not torch.is_grad_enabled()
        inputs.append(args[0].tolist()[0])

    hook = session.model.register_forward_pre_hook(capture)
    result = session.generate("abcdef")
    hook.remove()
    assert result["completion"] == "ghijkl"
    assert result["text"] == "abcdefghijkl"
    assert inputs == [
        list(b"cdef"),
        list(b"defg"),
        list(b"efgh"),
        list(b"fghi"),
        list(b"ghij"),
        list(b"hijk"),
    ]
    assert result["context"]["prompt_tokens_dropped"] == 2
    assert result["finish_reason"] == "length"
    assert torch.equal(rng_before, torch.random.get_rng_state())
    second = session.generate("A", max_new_tokens=1)
    assert second["completion"] == "B"
    rows = [json.loads(line) for line in destination.read_text().splitlines()]
    assert rows == [result, second]
    assert len(result["checkpoint"]["hash"]) == 64
    assert result["tokenizer_verified"]


def test_eos_and_empty_prompt(saved):
    _, _, _, run, _ = saved
    session = GenerationSession(InferenceConfig(run_dir=run, device="cpu", temperature=0))
    empty = session.generate("", max_new_tokens=1)
    assert empty["generated_token_ids"] == [0]  # EOS 256 is fed as the initial context.
    assert empty["context"]["empty_prompt_uses_eos"]
    with torch.no_grad():
        session.model.state_dict()["projection.weight"][256, ord("A")] = 100
    eos = session.generate("A")
    assert eos["generated_token_ids"] == [256]
    assert eos["finish_reason"] == "eos" and eos["completion"] == ""


def test_seeded_sampling_is_independent_of_prompt_order_and_global_rng(saved):
    _, _, _, run, _ = saved
    session = GenerationSession(InferenceConfig(run_dir=run, device="cpu", top_p=1))
    with torch.no_grad():
        session.model.state_dict()["projection.weight"].zero_()
    before = torch.random.get_rng_state().clone()
    first = session.generate("A", seed=8, max_new_tokens=40)
    session.generate("B", seed=19, max_new_tokens=40)
    repeat = session.generate("A", seed=8, max_new_tokens=40)
    different = session.generate("A", seed=9, max_new_tokens=40)
    assert first["generated_token_ids"] == repeat["generated_token_ids"]
    assert first["generated_token_ids"] != different["generated_token_ids"]
    assert torch.equal(before, torch.random.get_rng_state())
    assert session.config.seed == 1337
    with pytest.raises(ValueError, match="temperature"):
        session.generate("A", temperature=-1)


def test_top_p_filters_tail_and_small_temperature_is_finite():
    logits = torch.log(torch.tensor([0.6, 0.3, 0.1]))
    generator = torch.Generator().manual_seed(42)
    assert {sample_token(logits, 1, 0.65, generator) for _ in range(100)} == {0, 1}
    assert {sample_token(logits, 1, 0.5, generator) for _ in range(20)} == {0}
    assert sample_token(logits, 1e-300, 1, generator) == 0
    assert sample_token(logits, 0, 1, generator) == 0


@pytest.mark.parametrize(
    "change, message",
    [
        ("config_digest", "config_digest"),
        ("run_id", "step/run"),
        ("step", "step/run"),
        ("tokenizer", "tokenizer"),
        ("weights", "size mismatch"),
        ("checksum", "corrupt"),
    ],
)
def test_invalid_checkpoint_rejected(saved, tmp_path, change, message):
    _, state, metadata, _, checkpoint = saved
    state, metadata = copy.deepcopy(state), copy.deepcopy(metadata)
    if change == "weights":
        state["model"]["projection.weight"] = torch.zeros(2, 2)
    elif change == "tokenizer":
        metadata["tokenizer"]["eos_token_id"] = 999
    elif change == "step":
        metadata["step"] = 99
    elif change != "checksum":
        metadata[change] = "wrong"
    altered = CheckpointManager(tmp_path / "altered", 1).save(4, state, metadata)
    if change == "checksum":
        (altered / "state.pt").write_bytes(b"corrupt")
    with pytest.raises((ValueError, RuntimeError), match=message):
        GenerationSession(InferenceConfig(checkpoint=altered, device="cpu"))


def test_legacy_config_and_model_contract(saved, tmp_path):
    training, state, metadata, _, _ = saved
    state = dict(state)
    state.pop("config")
    checkpoint = CheckpointManager(tmp_path / "legacy", 1).save(4, state, metadata)
    config = InferenceConfig(checkpoint=checkpoint, device="cpu")
    with pytest.raises(ValueError, match="training_config"):
        GenerationSession(config)
    original = tmp_path / "original.yaml"
    dump_config(training, original)
    session = GenerationSession(replace(config, training_config=original))
    assert session.generate("A", temperature=0, max_new_tokens=1)["completion"] == "B"
    hook = session.model.register_forward_hook(lambda *args: LMOutput(logits=torch.zeros(1, 1, 3)))
    with pytest.raises(ValueError, match="shape"):
        session.generate("A")
    hook.remove()
    with torch.no_grad():
        session.model.state_dict()["projection.weight"].fill_(float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        session.generate("A")


def test_cli_batch_and_prompt_validation(saved, tmp_path):
    _, _, _, _, checkpoint = saved
    output = tmp_path / "samples.jsonl"
    config = tmp_path / "inference.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "checkpoint": str(checkpoint),
                "device": "cpu",
                "temperature": 0,
                "max_new_tokens": 2,
                "output": str(output),
            }
        )
    )
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('"A"\n"Line\\nbreak"\n""\n')
    command = [sys.executable, "-m", "nanoscope", "generate", "--config", str(config), "--json"]
    result = subprocess.run(
        command + ["--prompts", str(prompts)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert [row["prompt"] for row in rows] == ["A", "Line\nbreak", ""]
    assert rows[0]["completion"] == "BC"
    assert output.read_text().splitlines() == result.stdout.splitlines()
    single = subprocess.run(
        command + ["--prompt", "B"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(single.stdout)["completion"] == "CD"
    readable = subprocess.run(
        command[:-1] + ["--prompt", "C"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert readable.stdout == "CDE\n"
    before = output.read_text()
    prompts.write_text('"valid"\n{"bad": 1}\n')
    failed = subprocess.run(
        command + ["--prompts", str(prompts)], cwd=ROOT, capture_output=True, text=True, timeout=30
    )
    assert failed.returncode != 0 and "line 2" in failed.stderr
    assert output.read_text() == before
    for invalid in ["", "not json\n", "\n"]:
        prompts.write_text(invalid)
        with pytest.raises(ValueError):
            read_prompts(prompts)


def test_device_selection_and_unavailable_cuda(saved, monkeypatch):
    _, _, _, run, _ = saved
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = InferenceConfig(run_dir=run, device="auto")
    assert GenerationSession(config).device.type == "cpu"
    with pytest.raises(ValueError, match="CUDA was requested"):
        GenerationSession(replace(config, device="cuda"))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_generation(saved):
    _, _, _, run, _ = saved
    session = GenerationSession(InferenceConfig(run_dir=run, device="cuda", temperature=0))
    result = session.generate("A", max_new_tokens=3)
    assert result["completion"] == "BCD"
    assert result["runtime"]["device"] == "cuda"
