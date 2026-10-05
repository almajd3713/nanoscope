"""M1: does the modern block beat a GPT-2 control, and which parts matter?

Every variant is matched to the GPT-2 control's non-embedding parameter count by
resizing its MLP, and all train on the same token budget, so FLOPs match too.
The baseline is the full modern block, so each "no-*" row is a leave-one-out
contribution: how much worse the model gets without that one component.

This version runs on TinyStories as a pipeline check. The M1 result proper runs on
FineWeb-Edu with mode="record" once that preset exists (phase 4).

    nanoscope study studies/m1_ablation.py --devices cuda:0
    nanoscope report studies/m1_ablation.py
"""

from nanoscope import Study, Tokens
from nanoscope.models import GPT2, Modern
from nanoscope.sizing import count_params, match_params

VOCAB = 4096  # the tinystories presets' BPE vocabulary
CONTROL = count_params(GPT2(vocab_size=VOCAB))[1]


def matched(**kwargs):
    """Modern with these settings, MLP width chosen to match the control's parameters."""
    return match_params(Modern, CONTROL, "ffn_hidden", range(64, 1024, 8),
                        vocab_size=VOCAB, **kwargs) | kwargs


study = Study("m1-ablation", preset="tinystories-5min", seeds=3, budget=Tokens(4e6),
              match="params", baseline="modern")
study.add("gpt2", GPT2)
for name, switch in {
    "modern": {},
    "no-rope": {"rope": False},
    "no-swiglu": {"swiglu": False},
    "no-rmsnorm": {"rmsnorm": False},
    "no-qk-norm": {"qk_norm": False},
    "no-gqa": {"n_kv_heads": 4},
    "no-z-loss": {"z_loss": 0.0},
}.items():
    study.add(name, Modern, **{k: v for k, v in matched(**switch).items() if k != "vocab_size"})
