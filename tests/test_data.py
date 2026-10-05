import shutil

import numpy as np
import pytest
import torch
from fakes import tiny

import nanoscope.dataset as dataset
from nanoscope import paths
from nanoscope.dataset import TokenDataset, load_data, publish_data

pytestmark = pytest.mark.usefixtures("fake_data")


def _docs(tokens: np.ndarray, eos: int) -> list[list[int]]:
    ends = np.flatnonzero(tokens == eos)
    starts = np.concatenate([[0], ends[:-1] + 1])
    return [tokens[s:e].tolist() for s, e in zip(starts, ends, strict=True)]


def test_large_splits_are_sharded_on_document_boundaries(monkeypatch):
    monkeypatch.setattr(dataset, "SHARD_TOKENS", 500)
    monkeypatch.setattr(dataset, "_CHUNK_DOCS", 8)
    data = load_data(tiny())
    shards = data.train.shards
    eos = data.tokenizer.eos_token_id

    assert len(shards) > 2
    assert all(s[-1] == eos for s in shards)
    assert sum(len(_docs(np.asarray(s), eos)) for s in shards) == 200
    assert data.train.meta["tokens"] == sum(len(s) for s in shards)


def test_sampling_covers_every_shard_and_never_crosses_one(monkeypatch):
    monkeypatch.setattr(dataset, "SHARD_TOKENS", 500)
    monkeypatch.setattr(dataset, "_CHUNK_DOCS", 8)
    data = load_data(tiny())
    train: TokenDataset = data.train
    batch = train.get_batch(256, torch.Generator().manual_seed(0))

    assert batch.shape == (256, 33)
    joined = [np.asarray(s).tolist() for s in train.shards]
    hits = set()
    for row in batch.tolist():
        found = [k for k, s in enumerate(joined)
                 if any(s[j : j + 33] == row for j in range(len(s) - 32))]
        assert found, "a window must lie inside one shard"
        hits.add(found[0])
    assert hits == set(range(len(joined)))


def test_holdout_validation_comes_from_train_and_is_never_trained_on(fake_data):
    data = load_data(tiny(holdout_docs=30, eval_docs=10))
    eos = data.tokenizer.eos_token_id
    decode = data.tokenizer.decode
    val = [decode(d) for d in _docs(data.val, eos)]
    train = [decode(d) for s in data.train.shards for d in _docs(np.asarray(s), eos)]

    assert len(val) == 10 and len(train) == 200
    assert not set(val) & set(train)
    assert [v.endswith(f"day {i}.") for i, v in enumerate(val)] == [True] * 10
    assert train[0].endswith("day 30.")  # training starts after the 30 held-out documents
    assert {args[2] for args in fake_data} == {"train"}  # no validation split was requested
    with pytest.raises(ValueError, match="exceeds holdout_docs"):
        load_data(tiny(holdout_docs=5, eval_docs=10))


def test_hub_data_downloads_instead_of_tokenizing(tmp_path, monkeypatch, fake_data):
    remote = tmp_path / "remote"
    published = load_data(tiny())  # prepare once, then pretend it lives on the Hub
    shutil.copytree(paths.data_dir(), remote)
    monkeypatch.setenv("NANOSCOPE_DATA_DIR", str(tmp_path / "fresh-cache"))
    fake_data.clear()
    downloads = []

    def fake_snapshot(repo_id, repo_type, local_dir, allow_patterns):
        downloads.append((repo_id, allow_patterns[0]))
        for src in remote.glob(allow_patterns[0]):
            dest = local_dir / src.relative_to(remote)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dest)

    class FakeApi:
        def list_repo_files(self, repo_id, repo_type):
            return [p.relative_to(remote).as_posix() for p in remote.rglob("*") if p.is_file()]

    dataset._hub_cache.clear()
    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)
    data = load_data(tiny(hub_data="me/tokens"))

    assert fake_data == []  # nothing was read or tokenized
    assert [p for _, p in downloads] == [
        "fake_stories/bpe-300-100/tokenizer.json",
        "fake_stories/bpe-300-100/train-200/*",
        "fake_stories/bpe-300-100/val-10/*",
    ]
    assert np.array_equal(data.val, published.val)


def test_unreachable_hub_falls_back_to_tokenizing(monkeypatch, fake_data, capsys):
    class OfflineApi:
        def list_repo_files(self, repo_id, repo_type):
            raise ConnectionError("offline")

    dataset._hub_cache.clear()
    monkeypatch.setattr("huggingface_hub.HfApi", OfflineApi)
    data = load_data(tiny(hub_data="me/offline"))
    assert len(data.val) > 0 and fake_data
    out = capsys.readouterr().out
    assert out.count("can't reach hf.co/datasets/me/offline (ConnectionError)") == 1


def test_publish_uploads_the_cache_layout(monkeypatch):
    calls = []

    class FakeApi:
        def create_repo(self, repo_id, **kw):
            calls.append(("create", repo_id, kw["repo_type"]))

        def upload_folder(self, **kw):
            calls.append(("upload", kw["path_in_repo"], sorted(kw["allow_patterns"])))

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    url = publish_data(tiny(), "me/tokens")

    assert url == "https://huggingface.co/datasets/me/tokens"
    assert calls == [
        ("create", "me/tokens", "dataset"),
        ("upload", "fake_stories/bpe-300-100", ["tokenizer.json", "train-200/*", "val-10/*"]),
    ]


def test_prepare_json_follows_data_prep_and_records_failure(fake_data, monkeypatch):
    import json

    from helpers import assert_valid

    from nanoscope.prepare import PrepareFile

    stages = []
    real = PrepareFile.stage

    def spy(self, stage, *args, **kwargs):
        if not stages or stages[-1] != stage:
            stages.append(stage)
        return real(self, stage, *args, **kwargs)

    monkeypatch.setattr(PrepareFile, "stage", spy)
    preset = tiny()
    load_data(preset)
    assert stages == ["tokenizer", "tokenize"]
    path = paths.data_dir() / preset.name / "prepare.json"
    doc = json.loads(path.read_text())
    assert_valid("prepare", doc)
    assert doc["stage"] == "done" and doc["error"] is None and doc["preset"] == preset.name

    path.unlink()  # cached data: nothing to prepare, so nothing is written
    load_data(preset)
    assert not path.exists()

    def broken(*args):
        raise RuntimeError("disk full")

    other = tiny(name="test-broken", train_docs=150)
    monkeypatch.setattr(dataset, "_iter_texts", broken)
    with pytest.raises(RuntimeError, match="disk full"):
        load_data(other)
    failed = json.loads((paths.data_dir() / "test-broken" / "prepare.json").read_text())
    assert failed["stage"] == "failed" and failed["error"] == {
        "type": "RuntimeError", "message": "disk full"}


def test_status_data_prints_each_presets_preparation(fake_data, capsys):
    from nanoscope.cli import main

    main(["status", "--data"])
    assert "no data preparation recorded" in capsys.readouterr().out
    load_data(tiny())
    capsys.readouterr()
    main(["status", "--data"])
    out = capsys.readouterr().out
    assert out.startswith("test-tiny  done") and "updated 20" in out


def test_hub_listing_is_cached_for_five_minutes_then_retried(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(dataset, "_now", lambda: clock[0])
    dataset._hub_cache.clear()
    calls = []

    class FakeApi:
        def list_repo_files(self, repo_id, repo_type):
            calls.append(repo_id)
            if len(calls) == 1:
                raise ConnectionError("offline")
            return ["a/meta.json"]

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    assert dataset._hub_files("user/repo") == frozenset()  # offline
    clock[0] += 299
    assert dataset._hub_files("user/repo") == frozenset() and len(calls) == 1  # still cached
    clock[0] += 2
    assert dataset._hub_files("user/repo") == frozenset({"a/meta.json"})  # network is back
    assert len(calls) == 2
