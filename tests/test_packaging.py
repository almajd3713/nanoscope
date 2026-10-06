"""What the built wheel and sdist contain."""
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def dist(tmp_path_factory):
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not installed")
    out = tmp_path_factory.mktemp("dist")
    subprocess.run([uv, "build", "--out-dir", str(out)], cwd=ROOT, check=True,
                   capture_output=True)
    return out


def test_wheel_contents(dist):
    (wheel,) = dist.glob("*.whl")
    names = zipfile.ZipFile(wheel).namelist()
    for wanted in ("baselines", "schemas", "curricula", "reference"):
        assert any(n.startswith(f"nanoscope/{wanted}/") for n in names), wanted
    assert any(n.endswith("licenses/LICENSE") for n in names)
    for unwanted in ("tests/", "solutions/", "notebooks/", "runs/"):
        assert not any(unwanted in n for n in names), unwanted


def test_sdist_excludes(dist):
    (sdist,) = dist.glob("*.tar.gz")
    with tarfile.open(sdist) as tar:
        names = [n.split("/", 1)[1] for n in tar.getnames() if "/" in n]
    assert "LICENSE" in names and "pyproject.toml" in names
    assert any(n.startswith("nanoscope/") for n in names)
    for unwanted in ("data/", "runs/", "studies/", "experiments/", ".kaggle-outputs",
                     ".env", "AGENTS.md", ".hypothesis", ".venv", "graphify-out"):
        assert not any(n == unwanted.rstrip("/") or n.startswith(unwanted) for n in names), unwanted
