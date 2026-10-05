import random

import pytest

import nanoscope.dataset as dataset

ANIMALS = ["cat", "dog", "bird", "fox", "frog"]
VERBS = ["saw", "liked", "chased", "found", "hugged"]


def _fake_texts(dataset_name, config, split, max_docs):
    rng = random.Random(0 if split == "train" else 1)
    n = max_docs or 500
    for i in range(n):
        a, b = rng.choice(ANIMALS), rng.choice(ANIMALS)
        yield f"Once upon a time, the {a} {rng.choice(VERBS)} the {b}. It was day {i}."


@pytest.fixture
def fake_data(tmp_path, monkeypatch):
    calls = []

    def texts(*args):
        calls.append(args)
        return _fake_texts(*args)

    monkeypatch.setattr(dataset, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(dataset, "_iter_texts", texts)
    monkeypatch.chdir(tmp_path)
    return calls
