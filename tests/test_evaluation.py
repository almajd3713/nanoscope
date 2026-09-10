from __future__ import annotations

import copy
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from nanoscope.config import DataConfig, PartitionConfig, TokenizerConfig, load_config
from nanoscope.data import build_batch_stream
from nanoscope.data.partitions import accepts_document, document_split
from nanoscope.eval.artifacts import read_json, write_json
from nanoscope.eval.compare import build_comparison
from nanoscope.eval.config import CorpusConfig, EvalConfig, load_corpus_config, load_eval_config
from nanoscope.eval.corpus import load_corpus, prepare_corpus
from nanoscope.eval.report import write_report
from nanoscope.eval.runner import evaluate_checkpoint, load_result, score_model
from nanoscope.model import LMOutput
from nanoscope.train.acceptance import _state_differences
from nanoscope.train.checkpoint import CheckpointManager
from nanoscope.train.determinism import capture_rng_state
from nanoscope.train.trainer import train

PARTITION = PartitionConfig(buckets=10, validation_buckets=2, test_buckets=2)
DOCUMENTS = [
    f"Document {i}: reproducible experiments compare the same held-out text." for i in range(50)
]


@pytest.fixture
def corpus_config(tmp_path):
    return CorpusConfig(
        DataConfig(
            source="fixture",
            documents=DOCUMENTS,
            sequence_length=8,
            partition=replace(PARTITION, split="validation"),
        ),
        TokenizerConfig(name="byte", eos_token_id=256),
        5,
        tmp_path / "corpus",
    )


def test_partition_is_disjoint_seed_independent_and_resume_safe():
    groups = {
        role: {text for text in DOCUMENTS if document_split(text, PARTITION) == role}
        for role in ("train", "validation", "test")
    }
    assert all(groups.values())
    assert not groups["train"] & (groups["validation"] | groups["test"])
    assert not groups["validation"] & groups["test"]
    assert set.union(*groups.values()) == set(DOCUMENTS)
    data = DataConfig(documents=DOCUMENTS, sequence_length=8, partition=PARTITION)
    tokenizer = TokenizerConfig("byte", 256)
    stream = build_batch_stream(data, tokenizer, seed=7, batch_size=3)
    for _ in range(5):
        stream.next_batch()
    state = copy.deepcopy(stream.state_dict())
    expected = stream.next_batch()
    restored = build_batch_stream(data, tokenizer, seed=7, batch_size=3)
    restored.load_state_dict(state)
    assert torch.equal(expected, restored.next_batch())
    changed = build_batch_stream(
        replace(data, partition=replace(PARTITION, salt="different")),
        tokenizer,
        seed=7,
        batch_size=3,
    )
    with pytest.raises(ValueError, match="partition"):
        changed.load_state_dict(state)
    with pytest.raises(ValueError, match="no fixture documents"):
        build_batch_stream(replace(data, documents=list(groups["test"])), tokenizer, 7, 3)


def test_partition_config_validation_and_legacy_digest(tmp_path):
    base = load_config("configs/test/m0/local-smoke.yaml")
    digest = base.digest
    base.data.partition = PARTITION
    assert base.digest != digest
    raw = base.to_dict()
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert load_config(path).digest == base.digest
    raw["data"]["partition"]["split"] = "validation"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="must be train"):
        load_config(path)
    with pytest.raises(ValueError, match="training bucket"):
        PartitionConfig(buckets=2, validation_buckets=1, test_buckets=1)


def test_frozen_corpus_rebuild_hashes_and_corruption(corpus_config):
    one = prepare_corpus(corpus_config)
    two = prepare_corpus(replace(corpus_config, output=corpus_config.output.parent / "second"))
    assert one.manifest == two.manifest
    assert np.array_equal(one.tokens, two.tokens)
    assert one.manifest["scored_tokens"] == 40
    assert prepare_corpus(corpus_config).fingerprint == one.fingerprint
    with pytest.raises(ValueError, match="different recipe"):
        prepare_corpus(replace(corpus_config, sequences=6))
    with pytest.raises(ValueError, match="corpus_hash"):
        load_corpus(one.path, "a" * 64)
    token_path = one.path / "tokens.npy"
    content = bytearray(token_path.read_bytes())
    content[-1] ^= 1
    token_path.write_bytes(content)
    with pytest.raises(ValueError, match="token checksum"):
        load_corpus(one.path)


def test_corpus_is_finite_and_deduplicates(corpus_config):
    text = next(t for t in DOCUMENTS if accepts_document(t, corpus_config.data.partition))
    config = replace(
        corpus_config, data=replace(corpus_config.data, documents=[text] * 100), sequences=100
    )
    with pytest.raises(ValueError, match="exhausted"):
        prepare_corpus(config)
    assert not config.output.exists()


def test_fineweb_partition_resume_without_network(monkeypatch):
    import datasets

    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda *a, **kw: datasets.Dataset.from_dict({"text": DOCUMENTS}).to_iterable_dataset(),
    )
    data = DataConfig(
        source="fineweb",
        revision="a" * 40,
        partition=PARTITION,
        sequence_length=8,
        shuffle_buffer=7,
    )
    one = build_batch_stream(data, TokenizerConfig("byte", 256), 7, 2)
    one.next_batch()
    state = copy.deepcopy(one.state_dict())
    expected = [one.next_batch() for _ in range(20)]
    two = build_batch_stream(data, TokenizerConfig("byte", 256), 7, 2)
    two.load_state_dict(state)
    assert all(torch.equal(batch, two.next_batch()) for batch in expected)
    assert state["source"]["partition"] == asdict(PARTITION)


class TableLM(nn.Module):
    table: torch.Tensor

    def __init__(self, auxiliary=0.0):
        super().__init__()
        self.register_buffer(
            "table", torch.tensor([[0.0, 1.0, 2.0], [2.0, 0.0, 1.0], [1.0, 2.0, 0.0]])
        )
        self.dropout = nn.Dropout(0.8)
        self.auxiliary = auxiliary

    def forward(self, inputs):
        return LMOutput(self.dropout(self.table[inputs]), {"z_loss": torch.tensor(self.auxiliary)})


def test_scoring_shift_weighting_auxiliary_and_state():
    blocks = np.array([[0, 1, 2], [1, 0, 2], [2, 2, 1], [0, 0, 1], [2, 1, 0]])
    model = TableLM().train()
    model.dropout.eval()  # Mixed module flags must also survive evaluation.
    before = capture_rng_state(torch.device("cpu"))
    result = score_model(model, blocks, batch_size=2)
    assert not _state_differences(before, capture_rng_state(torch.device("cpu")))
    assert model.training and not model.dropout.training
    log_probs = model.table.log_softmax(-1)
    expected = (
        -sum(
            float(log_probs[a, b])
            for block in blocks
            for a, b in zip(block[:-1], block[1:], strict=True)
        )
        / 10
    )
    assert result["cross_entropy_nats"] == pytest.approx(expected, abs=1e-7)
    assert result["scored_tokens"] == 10
    assert score_model(TableLM(1e6), blocks, batch_size=5)["cross_entropy_nats"] == (
        pytest.approx(result["cross_entropy_nats"], abs=1e-7)
    )
    model.table.zero_()
    uniform = score_model(model, blocks, batch_size=1)
    assert uniform["cross_entropy_nats"] == pytest.approx(math.log(3), abs=1e-6)
    assert uniform["perplexity"] == pytest.approx(3, abs=1e-6)
    model.table.fill_(float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        score_model(model, blocks)
    assert model.training and not model.dropout.training


@pytest.fixture
def evaluated_runs(tmp_path, corpus_config):
    torch.set_num_threads(1)
    corpus = prepare_corpus(corpus_config)
    evaluation = EvalConfig(corpus.path, tmp_path / "evaluations", batch_size=2)
    outputs = {}
    for name, dropout in (("base", 0.1), ("variant", 0.2)):
        config = load_config("configs/test/m0/local-smoke.yaml")
        config.run.id = name
        config.run.output_dir = str(tmp_path / "runs")
        config.data = replace(corpus_config.data, partition=PARTITION)
        config.train.max_steps = 2
        config.scheduler.warmup_steps = 0
        config.checkpoint.every_steps = 1
        config.model.params["dropout"] = dropout
        result = train(config, resume="none")
        paths = []
        for step in (1, 2):
            checkpoint = result.run_dir / "checkpoints" / f"step_{step:08d}"
            paths.append(evaluate_checkpoint(checkpoint, evaluation))
        outputs[name] = paths
    study = tmp_path / "study.yaml"
    study.write_text(
        yaml.safe_dump(
            {
                "name": "fixture increments",
                "baseline": "base",
                "axis": "tokens",
                "budget": 32,
                "allow_changes": ["model.params.dropout"],
                "variants": [
                    {"name": name, "results": [str(p) for p in paths]}
                    for name, paths in outputs.items()
                ],
            },
            sort_keys=False,
        )
    )
    return outputs, study, evaluation


def test_checkpoint_evaluation_and_report(evaluated_runs, tmp_path):
    outputs, study, evaluation = evaluated_runs
    result = load_result(outputs["base"][-1])
    assert result["training"]["held_out_verified"]
    assert result["training"]["tokens_seen"] == 32
    checkpoint = Path(result["checkpoint"]["path"])
    # Re-evaluation is idempotent, including recorded timing.
    payload = outputs["base"][-1].read_bytes()
    assert evaluate_checkpoint(checkpoint, evaluation) == outputs["base"][-1]
    assert outputs["base"][-1].read_bytes() == payload
    comparison = build_comparison(study)
    assert [row["step"] for row in comparison["rows"]] == [2, 2]
    assert comparison["rows"][0]["delta_baseline"] == 0
    assert len(comparison["curves"]) == 4
    path = write_report(comparison, tmp_path / "report")
    assert path.is_file() and (path.parent / "comparison.csv").is_file()
    assert read_json(path.parent / "comparison.json") == comparison


@pytest.mark.parametrize(
    "change,expected",
    [
        ({"budget": 48}, "missing evaluated checkpoint"),
        ({"allow_changes": []}, "undeclared"),
        ({"axis": "flops", "budget": 1_000}, "parameter_tolerance"),
    ],
)
def test_comparison_rejects_unfair_studies(evaluated_runs, change, expected):
    _, study, _ = evaluated_runs
    raw = yaml.safe_load(study.read_text())
    raw.update(change)
    study.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=expected):
        build_comparison(study)


def test_legacy_checkpoint_is_diagnostic_and_config_is_verified(evaluated_runs, tmp_path):
    outputs, _, evaluation = evaluated_runs
    current = load_result(outputs["base"][-1])
    checkpoint = Path(current["checkpoint"]["path"])
    manager = CheckpointManager(tmp_path / "legacy", 1)
    state, metadata = manager.load(checkpoint, torch.device("cpu"))
    config_path = tmp_path / "training.yaml"
    config_path.write_text(yaml.safe_dump(state.pop("config")))
    metadata.pop("partition_version")
    legacy = manager.save(state["step"], state, metadata)
    with pytest.raises(ValueError, match="training-config"):
        evaluate_checkpoint(legacy, evaluation)
    path = evaluate_checkpoint(legacy, evaluation, config_path)
    assert not load_result(path)["training"]["held_out_verified"]
    raw = yaml.safe_load(config_path.read_text())
    raw["model"]["params"]["hidden_size"] = 999
    config_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="config_digest"):
        evaluate_checkpoint(legacy, evaluation, config_path)
    (legacy / "state.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt checkpoint"):
        evaluate_checkpoint(legacy, evaluation, config_path)


def test_result_checksum_rejects_edited_score(evaluated_runs):
    outputs, _, _ = evaluated_runs
    path = outputs["base"][-1]
    result = read_json(path)
    result["metrics"]["cross_entropy_nats"] = 0
    write_json(path, result)
    with pytest.raises(ValueError, match="checksum"):
        load_result(path)


def test_evaluation_config_paths_and_unknown_fields(tmp_path, corpus_config):
    path = tmp_path / "eval.yaml"
    path.write_text("corpus: corpus\noutput: output\nbatch_size: 2\n")
    config = load_eval_config(path)
    assert config.corpus == tmp_path / "corpus"
    path.write_text(path.read_text() + "mystery: true\n")
    with pytest.raises(ValueError, match="unknown"):
        load_eval_config(path)
    raw = {
        "data": asdict(corpus_config.data),
        "tokenizer": asdict(corpus_config.tokenizer),
        "sequences": 5,
        "output": "corpus",
    }
    path.write_text(yaml.safe_dump(raw))
    assert load_corpus_config(path).output == corpus_config.output
    raw["data"]["source"] = "fineweb"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="pinned"):
        load_corpus_config(path)


def test_plots_are_exported(evaluated_runs, tmp_path):
    pytest.importorskip("matplotlib")
    _, study, _ = evaluated_runs
    output = tmp_path / "plots"
    write_report(build_comparison(study), output, plots=True)
    assert len(list(output.glob("*.png"))) == 2
    assert len(list(output.glob("*.svg"))) == 2
