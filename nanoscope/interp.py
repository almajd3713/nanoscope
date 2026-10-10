"""Interpretability helpers over a run's checkpoints.

    induction_curve(ref)    # the best head's induction score at every saved step

An induction head attends from a token in the second copy of a repeated sequence to the token
after its earlier occurrence. The prompt is `half` random token ids followed by the same ids
again, so a model can only do well on the second half by copying.
"""

from __future__ import annotations

from typing import Any

import torch

from nanoscope.blocks.attention import Attention

MAX_HALF = 32  # the prompt is 2 * half tokens; inspect's own limit is 64


def repeated_ids(vocab_size: int, half: int, seed: int = 0,
                 avoid: tuple[int, ...] = ()) -> list[int]:
    """`half` random ids, then the same ids again. `avoid` (e.g. the eos id) is never drawn."""
    generator = torch.Generator().manual_seed(seed)
    ids: list[int] = []
    while len(ids) < half:
        i = int(torch.randint(vocab_size, (1,), generator=generator))
        if i not in avoid:
            ids.append(i)
    return ids + ids


def induction_scores(model: torch.nn.Module, ids: list[int]) -> list[dict[str, Any]]:
    """Per head: the mean attention from each position of the second copy to the token after
    its earlier twin. `ids` is a sequence repeated twice (see `repeated_ids`)."""
    half = len(ids) // 2
    captured: list[tuple[str, Attention, torch.Tensor]] = []
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            def grab(m: torch.nn.Module, args: tuple[Any, ...], name: str = name) -> None:
                captured.append((name, m, args[0].detach()))  # type: ignore[arg-type]
            handles.append(module.register_forward_pre_hook(grab))
    device = next(model.parameters()).device
    try:
        with torch.no_grad():
            model(torch.tensor([ids], dtype=torch.long, device=device))
            heads = []
            for layer, (name, module, inp) in enumerate(captured):
                weights = module.attention_weights(inp)[0]  # (n_heads, T, T)
                for head in range(weights.size(0)):
                    rows = [float(weights[head, i, i - half + 1]) for i in range(half, 2 * half)]
                    heads.append({"module": name, "layer": layer, "head": head,
                                  "score": sum(rows) / len(rows)})
    finally:
        for h in handles:
            h.remove()
    return heads


def induction_curve(ref: str, half: int = 16, seed: int = 0, device: str = "cpu"
                    ) -> list[dict[str, Any]]:
    """For every saved checkpoint of the run (`checkpoint_steps` archives the early ones), the
    induction score of every head and of the best one: `[{"step", "best", "heads"}, ...]`."""
    from nanoscope.store import load_run, saved_steps

    if not 1 <= half <= MAX_HALF:
        raise ValueError(f"half must be between 1 and {MAX_HALF}, got {half}")
    first = load_run(ref, device=device)
    steps = saved_steps(first.run_dir)
    if not steps:
        raise FileNotFoundError(f"{ref} has no checkpoints")
    vocab = first.tokenizer.vocab_size
    ids = repeated_ids(vocab, half, seed, avoid=(first.tokenizer.eos_token_id,))
    curve = []
    for step in steps:
        model = load_run(ref, device=device, step=step).model.eval()
        heads = induction_scores(model, ids)
        if not heads:
            raise ValueError(f"{type(model).__name__} has no Attention blocks")
        curve.append({"step": step, "best": max(heads, key=lambda h: h["score"]), "heads": heads})
    return curve
