"""docker compose end to end: the API and a worker in real containers."""

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
