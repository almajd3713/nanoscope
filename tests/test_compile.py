"""torch.compile is opt-in speed: it must not change what a run learns or how it resumes."""

import os
import signal

import pytest
import torch

from nanoscope import run
from nanoscope.models import GPT2
from tests.fakes import tiny

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(),
                                                 reason="needs CUDA")]
KW = dict(device="cuda", progress=False, n_layers=2, d_model=32, n_heads=2)
MODES = [True, "reduce-overhead"]


@pytest.mark.parametrize("mode", MODES)
def test_compiled_losses_match_eager(fake_data, mode):
    eager = run(GPT2, tiny(), output_dir="eager", **KW)
    compiled = run(GPT2, tiny(), output_dir="compiled", compile=mode, **KW)
    a, b = [r["loss"] for r in eager.metrics], [r["loss"] for r in compiled.metrics]
    assert a[0] == pytest.approx(b[0], rel=1e-3)  # same weights, same first batch
    assert b[-1] == pytest.approx(a[-1], rel=0.05)


@pytest.mark.parametrize(("first", "second"), [(False, True), (True, False)])
def test_checkpoints_resume_with_compile_on_or_off(fake_data, first, second):
    def interrupt(step, row):
        if step == 7:
            os.kill(os.getpid(), signal.SIGINT)

    run(GPT2, tiny(), output_dir="r", on_step=interrupt, compile=first, **KW)
    resumed = run(GPT2, tiny(), output_dir="r", compile=second, **KW)
    assert resumed.final_step == 20
    state = torch.load(next((resumed.run_dir / "checkpoints").glob("*.pt")), weights_only=False)
    assert not any("_orig_mod" in key for key in state["model"])
