from pathlib import Path

from nanoscope.config import load_config
from nanoscope.train.acceptance import run_acceptance


def test_three_resume_run_matches_control(tmp_path: Path) -> None:
    config = load_config("configs/m0/local-smoke.yaml")
    report = run_acceptance(config, tmp_path)
    assert report["passed"], report["failures"]
    assert len(report["interruptions"]) == 3
    assert report["interruptions"] == [2, 5, 8]
