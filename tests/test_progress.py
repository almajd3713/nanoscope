import os
import signal
import time

import pytest
from fakes import tiny

from nanoscope import Study, Tokens, run
from nanoscope.cli import main
from nanoscope.models import Bigram
from nanoscope.progress import format_snapshot, one_line, snapshot

pytestmark = pytest.mark.usefixtures("fake_data")


def test_run_shows_a_progress_bar_and_eval_lines(capsys):
    run(Bigram, tiny(), device="cpu")
    out = capsys.readouterr()
    assert "Bigram seed 0" in out.err  # the bar
    assert "step     10  val loss" in out.out and "bits per byte" in out.out


def test_progress_can_be_turned_off(capsys):
    run(Bigram, tiny(), device="cpu", progress=False)
    out = capsys.readouterr()
    assert "Bigram seed 0" not in out.err and "val loss" not in out.out


def test_snapshot_tells_done_running_and_stopped_apart():
    done = run(Bigram, tiny(), device="cpu", output_dir="runs/done", progress=False)

    def interrupt(step, row):
        if step == 5:
            os.kill(os.getpid(), signal.SIGINT)

    run(Bigram, tiny(), device="cpu", output_dir="runs/live", on_step=interrupt, progress=False)
    run(Bigram, tiny(), device="cpu", output_dir="runs/old", on_step=interrupt, progress=False)
    old = time.time() - 3600
    os.utime("runs/old/metrics.jsonl", (old, old))

    states = {r.run_dir.name: r for r in snapshot("runs")}
    assert states["done"].state == "done" and states["done"].step == 20
    assert states["done"].val_bpb == done.summary()["final_val_bpb"]
    assert states["live"].state == "running" and states["live"].step == 5
    assert states["old"].state == "stopped"

    text = format_snapshot(list(states.values()), "runs")
    assert "3 runs under runs: 1 done, 1 running, 1 stopped" in text
    assert "5/20 (25%)" in text and "60m ago" in text
    assert one_line(list(states.values()), total=4).startswith("1/4 done · running runs/live 5/20")


def test_status_command_prints_the_snapshot(capsys):
    run(Bigram, tiny(), device="cpu", progress=False)
    main(["status"])
    assert "1 runs under runs: 1 done" in capsys.readouterr().out


def test_study_numbers_its_runs_and_reports_each_result(capsys):
    study = Study("toy", preset=tiny(), seeds=1, budget=Tokens(4 * 32 * 4))
    study.add("a", Bigram)
    study.add("b", Bigram, d_model=8)
    study.run(devices=["cpu"])
    out = capsys.readouterr().out
    assert "run 1/2: a seed 0" in out
    assert "run 2/2: b seed 0 ->" in out and "bits per byte" in out


def test_status_lists_a_studys_runs_that_have_not_started():
    study = Study("toy", preset=tiny(), seeds=2, budget=Tokens(4 * 32 * 4))
    study.add("a", Bigram)
    study._write_plan(study.jobs())
    run(Bigram, study.jobs()[0].preset, device="cpu", output_dir=study.dir / "a" / "seed-0",
        progress=False)

    states = {f"{r.run_dir.parent.name}/{r.run_dir.name}": r.state for r in snapshot("runs")}
    assert states == {"a/seed-0": "done", "a/seed-1": "queued"}
    assert one_line(snapshot(study.dir)) == "1/2 done"
