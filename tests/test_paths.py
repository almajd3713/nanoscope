from pathlib import Path

from nanoscope import paths


def test_defaults_without_home(monkeypatch):
    for var in ("NANOSCOPE_HOME", "NANOSCOPE_DATA_DIR", "NANOSCOPE_WORKSPACE"):
        monkeypatch.delenv(var, raising=False)
    assert paths.home() is None
    assert paths.runs_dir() == Path("runs")
    assert paths.reports_dir() == Path("experiments")
    assert paths.data_dir() == Path.home() / ".nanoscope" / "data"


def test_home_set_after_import_moves_everything(monkeypatch, tmp_path):
    monkeypatch.delenv("NANOSCOPE_DATA_DIR", raising=False)
    monkeypatch.delenv("NANOSCOPE_WORKSPACE", raising=False)
    monkeypatch.setenv("NANOSCOPE_HOME", str(tmp_path))
    assert paths.home() == tmp_path
    assert paths.runs_dir() == tmp_path / "runs"
    assert paths.reports_dir() == tmp_path / "experiments"
    assert paths.data_dir() == tmp_path / "data"
    assert paths.learn_dir() == tmp_path / "learn"
    assert paths.hardware_dir() == tmp_path / "hardware"
    assert paths.workspace_dir() == tmp_path / "workspace"


def test_specific_variables_win_over_home(monkeypatch, tmp_path):
    monkeypatch.setenv("NANOSCOPE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("NANOSCOPE_DATA_DIR", str(tmp_path / "tokens"))
    monkeypatch.setenv("NANOSCOPE_WORKSPACE", str(tmp_path / "mine"))
    assert paths.data_dir() == tmp_path / "tokens"
    assert paths.workspace_dir() == tmp_path / "mine"
