"""docker compose end to end: the API and a worker in real containers."""

import json
import subprocess
import time

import pytest
from compose_helpers import wait_for_state

pytestmark = pytest.mark.compose


def test_bigram_under_2min(stack):
    with stack.client() as client:
        start = time.monotonic()
        # 300 of the preset's 500 steps: the 4-CPU container needs ~135 s for all of them
        sent = client.post("/api/runs", json={
            "model": "bigram", "preset": "tinystories-5min", "seed": 0,
            "kwargs": {"max_steps": 300}})
        assert sent.status_code == 202, sent.text
        ref = sent.json()["ref"]
    detail = wait_for_state(stack, ref, {"done", "failed", "cancelled"}, seconds=120)
    elapsed = time.monotonic() - start
    assert detail["status"]["state"] == "done", detail
    assert elapsed < 120, f"took {elapsed:.0f}s"
    # the run folder is on the home volume, where the API and every worker see it
    assert "status.json" in stack.volume_ls(f"/nanoscope/runs/{ref}")


@pytest.mark.parametrize("service", ["api", "worker"])
def test_hardened(stack, service):
    """The services that can run user code get no privileges and have limits set."""
    container = stack.compose("ps", "-q", service).stdout.strip()
    info = json.loads(subprocess.run(["docker", "inspect", container], check=True,
                                     capture_output=True, text=True).stdout)[0]
    host = info["HostConfig"]
    assert info["Config"]["User"] == "1000:1000"
    assert host["CapDrop"] == ["ALL"] and not host["CapAdd"]
    assert "no-new-privileges:true" in host["SecurityOpt"]
    assert host["ReadonlyRootfs"] is True
    assert not host["Privileged"]
    assert host["Memory"] == 8 * 1024**3
    assert host["NanoCpus"] == 4 * 10**9
    assert host["PidsLimit"] == 512
    assert not any("docker.sock" in m["Source"] for m in info["Mounts"])
    ids = stack.compose("exec", "-T", service, "id", "-u").stdout.strip()
    assert ids == "1000"
