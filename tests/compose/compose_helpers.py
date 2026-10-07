"""Polling helpers for the compose tests."""

import json
import time


def wait_for_state(stack, ref: str, states: set[str], seconds: float) -> dict:
    """Poll GET /api/runs/<ref> until its status.state is one of `states`."""
    deadline = time.monotonic() + seconds
    last: dict = {}
    with stack.client() as client:
        while time.monotonic() < deadline:
            r = client.get(f"/api/runs/{ref}")
            if r.status_code == 200 and r.json().get("status"):
                last = r.json()
                if last["status"]["state"] in states:
                    return last
            time.sleep(1)
    status, summary = json.dumps(last.get("status")), json.dumps(last.get("summary"))[:300]
    raise AssertionError(f"{ref} did not reach {states} in {seconds:.0f}s; last: {status} "
                         f"{summary}")
