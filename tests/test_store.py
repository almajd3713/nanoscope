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
