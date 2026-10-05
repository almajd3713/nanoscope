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
def home(tmp_path, monkeypatch):
    """Point $NANOSCOPE_HOME at a fresh folder so runs, reports and data stay in the test."""
    folder = tmp_path / "home"
    monkeypatch.setenv("NANOSCOPE_HOME", str(folder))
    monkeypatch.delenv("NANOSCOPE_DATA_DIR", raising=False)
    monkeypatch.delenv("NANOSCOPE_WORKSPACE", raising=False)
    return folder


@pytest.fixture
def fake_data(home, monkeypatch):
    calls = []

    def texts(*args):
        calls.append(args)
        return _fake_texts(*args)

    monkeypatch.setattr(dataset, "_iter_texts", texts)
    return calls
