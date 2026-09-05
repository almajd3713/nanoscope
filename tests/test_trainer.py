import copy
from pathlib import Path

import pytest

from nanoscope.config import load_config
from nanoscope.train.trainer import train


def test_incompatible_resume_is_rejected(tmp_path: Path) -> None:
    config = load_config("configs/m0/local-smoke.yaml")
    config.run.id = "resume-mismatch"
    config.run.output_dir = str(tmp_path)
    train(config, resume="none", stop_after_step=1)

    changed = copy.deepcopy(config)
    changed.optimizer.learning_rate *= 2
    with pytest.raises(RuntimeError, match="configuration is incompatible"):
        train(changed, resume="auto")
