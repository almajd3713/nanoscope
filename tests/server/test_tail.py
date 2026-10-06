import json
import os

import pytest

from nanoscope.server.tail import Tail


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def run_dir(tmp_path):
    folder = tmp_path / "runs" / "p" / "m" / "seed-0"
    (folder / "checkpoints").mkdir(parents=True)
    return folder


def append(run_dir, *rows, name="metrics.jsonl", end="\n"):
    with (run_dir / name).open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + end)


def kinds(events):
    return [k for k, _ in events]


def test_state_events_follow_status_json(run_dir):
    tail = Tail(run_dir, "p/m/seed-0", clock=Clock())
    assert tail.poll() == []  # nothing yet
    (run_dir / "status.json").write_text(json.dumps(
        {"schema": 1, "state": "preparing", "updated_at": "t", "step": 0, "max_steps": 20}))
    (kind, data), = tail.poll()
    assert kind == "state" and data["ref"] == "p/m/seed-0" and data["state"] == "preparing"
    assert tail.poll() == []  # unchanged: no event
    (run_dir / "status.json").write_text(json.dumps(
        {"schema": 1, "state": "running", "updated_at": "t2", "step": 3, "max_steps": 20}))
    assert [d["state"] for k, d in tail.poll() if k == "state"] == ["running"]
    (run_dir / "status.json").write_text(json.dumps(
        {"schema": 1, "state": "failed", "updated_at": "t3", "step": 3,
         "error": {"type": "X", "message": "boom"}}))
    (kind, data), = tail.poll()
    assert data["state"] == "failed" and data["error"]["message"] == "boom"


def test_steps_are_coalesced_to_four_a_second_but_evals_never_are(run_dir):
    clock = Clock()
    tail = Tail(run_dir, "r", clock=clock)
    tail.poll()  # a live view starts at the end of the file
    append(run_dir, *[{"step": s, "loss": 1.0 / s} for s in range(1, 11)])
    first = tail.poll()
    assert kinds(first) == ["step"] and first[0][1]["step"] == 10  # only the newest row
    append(run_dir, {"step": 11, "loss": 0.1}, {"step": 12, "loss": 0.1})
    assert tail.poll() == []  # too soon: held back
    clock.advance(0.3)
    assert [d["step"] for _, d in tail.poll()] == [12]  # now sent, newest of the held ones
    # an eval and a sample are never coalesced or dropped, and keep their order
    append(run_dir, {"step": 13, "loss": 0.1}, {"step": 14, "loss": 0.1, "val_loss": 2.5,
                                                "val_bpb": 1.2},
           {"step": 15, "loss": 0.1, "val_loss": 2.4, "val_bpb": 1.1, "sample": "once upon"})
    got = tail.poll()
    assert kinds(got) == ["step", "eval", "step", "eval", "sample"]
    assert [d["step"] for _, d in got] == [13, 14, 14, 15, 15]
    assert got[1][1] == {"ref": "r", "step": 14, "val_loss": 2.5, "val_bpb": 1.2}
    assert got[4][1]["text"] == "once upon"
    assert "sample" not in got[2][1]  # a step event does not carry the (long) sample text
    assert [d["step"] for _, d in tail.finish()] == [15]  # the step held back at the end


def test_a_step_held_back_is_flushed_at_the_end(run_dir):
    clock = Clock()
    tail = Tail(run_dir, "r", clock=clock)
    tail.poll()
    append(run_dir, {"step": 1, "loss": 1.0})
    assert kinds(tail.poll()) == ["step"]
    append(run_dir, {"step": 2, "loss": 0.9})
    assert tail.poll() == []
    assert [d["step"] for _, d in tail.finish()] == [2]


def test_partial_lines_wait_and_since_step_replays(run_dir):
    clock = Clock()
    append(run_dir, {"step": 1, "loss": 1.0}, {"step": 2, "loss": 0.9, "val_loss": 3.0})
    replay = Tail(run_dir, "r", since_step=1, clock=clock)
    assert [(k, d["step"]) for k, d in replay.poll()] == [("eval", 2), ("step", 2)]
    assert replay.poll() == []  # step 1 was before since_step: never sent
    live = Tail(run_dir, "r", clock=clock)
    live.poll()
    with (run_dir / "metrics.jsonl").open("a") as f:
        f.write('{"step": 3, "loss": 0.8')  # the trainer is mid-write
    assert live.poll() == []
    with (run_dir / "metrics.jsonl").open("a") as f:
        f.write("}\n")
    assert [d["step"] for _, d in live.poll()] == [3]


def test_checkpoint_and_blockstats_events(run_dir):
    tail = Tail(run_dir, "r", clock=Clock())
    (run_dir / "checkpoints" / "step_00000010.pt").write_bytes(b"x")
    tail.poll()  # existing at connect time: not news
    (run_dir / "checkpoints" / "step_00000020.pt").write_bytes(b"x")
    (run_dir / "checkpoints" / "archive").mkdir()
    (run_dir / "checkpoints" / "archive" / "step_00000020.pt").write_bytes(b"x")
    got = tail.poll()
    assert sorted((d["name"], d["step"], d["archived"]) for _, d in got) == [
        ("archive/step_00000020.pt", 20, True), ("step_00000020.pt", 20, False)]
    assert set(kinds(got)) == {"checkpoint"}
    append(run_dir, {"step": 20, "blocks": [{"name": "blocks.0"}]}, name="blockstats.jsonl")
    (kind, data), = tail.poll()
    assert kind == "blockstats" and data["blocks"][0]["name"] == "blocks.0" and data["step"] == 20


def test_reset_when_the_file_shrinks_or_is_replaced(run_dir):
    clock = Clock()
    tail = Tail(run_dir, "r", clock=clock)
    append(run_dir, *[{"step": s, "loss": 1.0} for s in range(1, 21)])
    tail.poll()
    tail.poll()
    # a resumed run rewrites metrics.jsonl truncated to its checkpoint: smaller file, same inode
    (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 1, "loss": 1.0}) + "\n")
    clock.advance(1)
    got = tail.poll()
    assert kinds(got)[0] == "reset" and got[0][1]["reason"] == "metrics.jsonl was rewritten"
    assert [d["step"] for k, d in got if k == "step"] == [1]  # re-read from the top
    # replaced by a new file (new inode) of the same size: also a reset
    inode = os.stat(run_dir / "metrics.jsonl").st_ino
    replacement = run_dir / "metrics.new"
    replacement.write_text((run_dir / "metrics.jsonl").read_text())
    os.replace(replacement, run_dir / "metrics.jsonl")
    assert os.stat(run_dir / "metrics.jsonl").st_ino != inode
    clock.advance(1)
    assert kinds(tail.poll())[0] == "reset"


def test_frames_are_sse(run_dir):
    from nanoscope.server.tail import frames

    text, = frames([("eval", {"ref": "r", "step": 10, "val_loss": 2.0})])
    assert text.startswith("event: eval\nid: 10\ndata: ")
    assert json.loads(text.split("data: ")[1]) == {"ref": "r", "step": 10, "val_loss": 2.0}
    state, = frames([("state", {"ref": "r", "state": "done"})])
    assert "id:" not in state
