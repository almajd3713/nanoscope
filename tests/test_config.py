from pathlib import Path

import pytest

from nanoscope.config import ConfigError, load_config


def test_local_config_loads() -> None:
    config = load_config(Path("configs/m0/local-smoke.yaml"))
    assert config.run.id == "m0-local-smoke"
    assert config.run.seed == 1337
    assert len(config.digest) == 64


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    source = Path("configs/m0/local-smoke.yaml").read_text(encoding="utf-8")
    path = tmp_path / "bad.yaml"
    path.write_text(source.replace("  seed: 1337", "  seed: 1337\n  mystery: true"))
    with pytest.raises(ConfigError, match="unknown keys in run"):
        load_config(path)

