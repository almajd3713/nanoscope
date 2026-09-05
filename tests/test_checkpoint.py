from pathlib import Path

import torch

from nanoscope.train.checkpoint import CheckpointManager, validate_checkpoint


def test_checkpoint_round_trip_and_corrupt_fallback(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path / "run", keep_last=3)
    first = manager.save(1, {"value": torch.tensor([1])}, {"config_digest": "a"})
    second = manager.save(2, {"value": torch.tensor([2])}, {"config_digest": "a"})
    assert validate_checkpoint(first)
    assert manager.latest() == second

    (second / "state.pt").write_bytes(b"corrupt")
    assert not validate_checkpoint(second)
    assert manager.latest() == first
    state, metadata = manager.load(first, torch.device("cpu"))
    assert state["value"].item() == 1
    assert metadata["config_digest"] == "a"


def test_archives_are_not_pruned(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path / "run", keep_last=1)
    archived = manager.save(
        1, {"value": torch.tensor([1])}, {"config_digest": "a"}, archive=True
    )
    manager.save(2, {"value": torch.tensor([2])}, {"config_digest": "a"})
    manager.save(3, {"value": torch.tensor([3])}, {"config_digest": "a"})
    assert archived.exists()
    assert not (manager.root / "step_00000002").exists()
    assert (manager.root / "step_00000003").exists()


def test_corrupt_step_is_quarantined_before_replacement(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path / "run", keep_last=1)
    checkpoint = manager.save(1, {"value": torch.tensor([1])}, {"config_digest": "a"})
    (checkpoint / "state.pt").write_bytes(b"corrupt")
    replacement = manager.save(1, {"value": torch.tensor([2])}, {"config_digest": "a"})
    assert validate_checkpoint(replacement)
    assert list(manager.root.glob(".corrupt-step_00000001-*"))
