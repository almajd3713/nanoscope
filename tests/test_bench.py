import json

import pytest
from fakes import tiny
from helpers import assert_valid

from nanoscope import paths
from nanoscope.bench import bench
from nanoscope.cli import main
from nanoscope.models import Bigram

pytestmark = pytest.mark.usefixtures("fake_data")


def test_bench_save_appends_a_row_per_run():
    first = bench(Bigram, tiny(), steps=5, warmup=2, device="cpu", save=True)
    bench(Bigram, tiny(), steps=5, warmup=2, device="cpu", save=True)
    rows = [json.loads(line) for line in
            (paths.hardware_dir() / "bench.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert_valid("bench", row)
    assert rows[0]["model"] == "Bigram" and rows[0]["preset"] == "test-tiny"
    assert rows[0]["device"] == "cpu" and rows[0]["steps"] == 5 and rows[0]["compile"] == "off"
    assert rows[0]["verdict"] == first.verdict


def test_bench_without_save_writes_nothing():
    bench(Bigram, tiny(), steps=5, warmup=2, device="cpu")
    assert not (paths.hardware_dir() / "bench.jsonl").exists()


def test_bench_command_saves_with_the_flag(monkeypatch, capsys):
    from nanoscope import presets

    monkeypatch.setitem(presets._PRESETS, "test-tiny", tiny())
    main(["bench", "bigram", "--preset", "test-tiny", "--steps", "3", "--device", "cpu"])
    assert "step time" in capsys.readouterr().out
    assert not (paths.hardware_dir() / "bench.jsonl").exists()
    main(["bench", "bigram", "--preset", "test-tiny", "--steps", "3", "--device", "cpu",
          "--save"])
    (row,) = (json.loads(line) for line in
              (paths.hardware_dir() / "bench.jsonl").read_text().splitlines())
    assert_valid("bench", row)


def test_a_bench_job_saves_its_result_to_the_history(monkeypatch):
    from learn_helpers import set_preset

    from nanoscope import queue

    set_preset(monkeypatch, tiny())
    job_id = queue.enqueue("bench", {"model": "bigram", "preset": "test-tiny", "steps": 5,
                                     "device": "cpu"}, lane="interactive")
    queue.claim("w", "cpu")
    with pytest.raises(SystemExit) as stopped:
        main(["run-job", str(job_id)])
    assert stopped.value.code == 0
    rows = [json.loads(line) for line in
            (paths.hardware_dir() / "bench.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["model"] == "Bigram" and rows[0]["device"] == "cpu"
