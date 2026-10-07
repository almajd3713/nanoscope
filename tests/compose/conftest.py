"""A real docker compose stack, brought up for the tests that ask for it (`-m compose`)."""

import os
import shutil
import socket
import subprocess
import time
import uuid
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
TOKEN = "compose-test-token"


def pytest_collection_modifyitems(config, items):
    """Compose tests build images and take minutes: run them only when asked (-m compose)."""
    if "compose" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="needs docker: run with -m compose")
    for item in items:
        if "compose" in item.keywords:
            item.add_marker(skip)


class Stack:
    def __init__(self, project: str, port: int, workspace: Path):
        self.project, self.port, self.workspace = project, port, workspace
        self.url = f"http://127.0.0.1:{port}"
        self.env = {**os.environ, "NANOSCOPE_PORT": str(port), "NANOSCOPE_TOKEN": TOKEN,
                    "NANOSCOPE_WORKSPACE": str(workspace)}

    def compose(self, *args: str, check: bool = True, timeout: int = 900):
        return subprocess.run(
            ["docker", "compose", "-p", self.project, "-f", str(ROOT / "compose.yaml"), *args],
            cwd=ROOT, env=self.env, check=check, capture_output=True, text=True, timeout=timeout)

    def client(self) -> httpx.Client:
        return httpx.Client(base_url=self.url, headers={"Authorization": f"Bearer {TOKEN}"},
                            timeout=30)

    def wait_healthy(self, seconds: int = 90) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"{self.url}/api/health", timeout=3).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(1)
        raise AssertionError("the API did not become healthy:\n" + self.compose(
            "logs", "--tail", "40", check=False).stdout)

    def up(self) -> None:
        self.compose("build", "api")  # the repo's own Dockerfile; cached layers make it quick
        self.compose("up", "-d", "--pull", "never", "api", "worker")
        self.wait_healthy()

    def seed_data(self) -> None:
        """Copy the host's token cache into the data volume, then let the worker's own
        prepare-data fill in whatever is missing (the network is allowed here, not in the
        timed part of a test)."""
        from nanoscope import paths

        host = Path(os.environ.get("NANOSCOPE_COMPOSE_DATA", paths.data_dir()))
        if host.is_dir() and any(host.iterdir()):
            container = self.compose("ps", "-q", "worker").stdout.strip()
            subprocess.run(["docker", "cp", f"{host}/.", f"{container}:/data"], check=True,
                           capture_output=True)
        self.compose("exec", "-T", "worker", "nanoscope", "prepare-data", "tinystories-5min",
                     timeout=1800)

    def volume_ls(self, path: str) -> str:
        return self.compose("exec", "-T", "api", "ls", path, check=False).stdout


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def stack(tmp_path_factory):
    if shutil.which("docker") is None:
        pytest.skip("docker is not installed")
    if subprocess.run(["docker", "compose", "version"], capture_output=True).returncode:
        pytest.skip("docker compose is not available")
    workspace = tmp_path_factory.mktemp("workspace")
    workspace.chmod(0o777)  # the containers write as uid 1000, whoever runs the test
    stack = Stack(f"nanoscope-test-{uuid.uuid4().hex[:8]}", free_port(), workspace)
    try:
        stack.up()
        stack.seed_data()
        yield stack
    finally:
        stack.compose("down", "-v", "--remove-orphans", check=False)
