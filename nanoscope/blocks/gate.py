"""The build gate: a model a learner writes may only use blocks and features they unlocked.

`Decoder` and `Composite` call `enforce` when they are constructed. Shipped models (marked
with `@shipped`, and subclasses of them) are skipped, so `run(Modern, n_kv_heads=1)` always
works. Imports are gated separately, in `nanoscope.blocks.__getattr__`; this catches what an
import can't see, like grouped-query attention (`n_kv_heads < n_heads`).
"""

from __future__ import annotations

from typing import Any

from nanoscope.blocks.registry import is_shipped
from nanoscope.blocks.spec import BlockSpec


def used(values: Any) -> tuple[list[str], list[str]]:
    """(block ids, feature ids) used by these specs, found by walking their options."""
    blocks: list[str] = []
    features: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, BlockSpec):
            name = value.cls.__name__
            blocks.append(f"block:{name}")
            opts = value.options
            if name == "Attention":
                heads, kv = opts.get("n_heads"), opts.get("n_kv_heads")
                if isinstance(heads, int) and isinstance(kv, int) and kv < heads:
                    features.append("feature:gqa")
                if opts.get("qk_norm"):
                    features.append("feature:qk_norm")
                if opts.get("window") is not None:
                    features.append("feature:sliding_window")
            for option in opts.values():
                walk(option)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(values)
    return blocks, features


def enforce(model: object, specs: list[Any], extra_features: tuple[str, ...] = ()) -> None:
    """Raise `LockedBlockError` if a model that is not shipped builds something locked."""
    if is_shipped(type(model)):
        return
    from nanoscope.learn import unlocks

    if not unlocks.exists():
        return
    from nanoscope.learn.gating import LockedBlockError, check

    blocks, features = used(specs)
    ids = list(dict.fromkeys([*blocks, *features, *extra_features]))
    locked = check(ids)
    if locked:
        raise LockedBlockError(locked)
