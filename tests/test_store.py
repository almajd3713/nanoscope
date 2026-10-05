import pytest
from fakes import tiny

from nanoscope import paths, run, store
from nanoscope.models import Bigram


def test_resolve_maps_refs_to_the_runs_root(home):
    folder = home / "runs" / "preset" / "set" / "seed-0"
    folder.mkdir(parents=True)
    assert store.resolve("preset/set/seed-0") == folder
    assert store.ref_of(folder) == "preset/set/seed-0"


def test_resolve_maps_baselines_to_the_package(home):
    assert store.resolve("baselines/tinystories-5min/gpt2") == (
        paths.baselines_dir() / "tinystories-5min" / "gpt2")
    assert store.ref_of(paths.baselines_dir() / "tinystories-5min" / "gpt2") == (
        "baselines/tinystories-5min/gpt2")


@pytest.mark.parametrize("ref", ["/etc/passwd", "../outside", "a/../../outside", "C:/windows",
                                 "a\\..\\b", ""])
def test_resolve_rejects_absolute_paths_and_traversal(home, ref):
    with pytest.raises(ValueError):
        store.resolve(ref)


def test_resolve_rejects_symlinks_that_leave_the_runs_root(home, tmp_path):
    (home / "runs").mkdir(parents=True)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (home / "runs" / "sneaky").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        store.resolve("sneaky")


def test_resolve_names_the_missing_run(home):
    with pytest.raises(FileNotFoundError, match="no run at ref 'nope/seed-0'"):
        store.resolve("nope/seed-0")


def test_ref_of_keeps_paths_outside_the_roots(tmp_path):
    assert store.ref_of(tmp_path / "mine") == (tmp_path / "mine").as_posix()


def test_list_runs_and_sets_over_two_sets_of_three_seeds(fake_data):
    run(Bigram, tiny(), device="cpu", seeds=3, progress=False)
    run(Bigram, tiny(), device="cpu", seeds=3, progress=False, d_model=8)
    assert len(store.list_runs()) == 6
    assert len(store.list_runs(state="done")) == 6
    assert store.list_runs(state="running") == []
    sets = store.list_sets()
    assert len(sets) == 2 and all(s.startswith("test-tiny-") for s in sets)
    assert len(store.list_runs(sets[0])) == 3
    assert store.list_sets(sets[0]) == [sets[0]]


def test_load_run_rebuilds_the_model_and_generates(fake_data):
    trained = run(Bigram, tiny(), device="cpu", progress=False)
    loaded = store.load_run(trained.ref)
    assert loaded.step == 20 and loaded.config["model"]["class"] == "Bigram"
    for name, tensor in trained.model.state_dict().items():
        assert (loaded.model.state_dict()[name] == tensor).all(), name
    text = loaded.generate("Once upon", max_new_tokens=15)
    assert isinstance(text, str) and text.startswith("Once upon")


def test_load_run_errors_explain_why(fake_data, monkeypatch):
    with pytest.raises(FileNotFoundError, match="shipped baseline.*not the checkpoints"):
        store.load_run("baselines/tinystories-5min/gpt2/seed-0")

    group = run(Bigram, tiny(), device="cpu", seeds=2, progress=False)
    with pytest.raises(ValueError, match=r"is a set of 2 seeds; load one, e.g. .*/seed-0"):
        store.load_run(group.ref)

    class Local(Bigram):
        pass

    Local.__module__ = "__main__"
    local = run(Local, tiny(), device="cpu", progress=False)
    with pytest.raises(ValueError, match="defined in a notebook or script.*model_source.py"):
        store.load_run(local.ref)


def test_load_run_generates_in_a_fresh_process(fake_data):
    import subprocess
    import sys

    trained = run(Bigram, tiny(), device="cpu", progress=False)
    code = (f"import nanoscope; r = nanoscope.load_run({trained.ref!r}); "
            "print('OUT:' + r.generate('Once upon', max_new_tokens=10))")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0, done.stderr
    assert "OUT:Once upon" in done.stdout
