from __future__ import annotations

from fastapi import APIRouter

from nanoscope import paths
from nanoscope.blocks.catalog import catalog
from nanoscope.learn import gating, unlocks
from nanoscope.server.models import BlocksDoc

router = APIRouter(prefix="/api", tags=["blocks"])


def lock_state(name: str, owner: str = "local") -> dict[str, object]:
    """Whether a block is locked for this learner right now, and how to unlock it."""
    unlock_id = f"block:{name}"
    lesson = gating.lock_table().get(unlock_id)
    if lesson is None:
        return {"lockable": False, "locked": False, "lesson": None, "how": None}
    entry = unlocks.read(owner)["unlocks"].get(unlock_id)
    open_policy = unlocks.policy(owner) == "open"
    return {"lockable": True, "locked": not (open_policy or entry), "lesson": lesson,
            "how": entry["how"] if entry else ("open" if open_policy else None)}


@router.get("/blocks")
def blocks() -> BlocksDoc:
    """The palette: every block with its options, tier, reference, whether it is certified
    against it, and its lock state under the current gating policy. Blocks registered in the
    workspace are listed too (found by reading the files)."""
    workspace = paths.workspace_dir()
    doc = catalog(workspace if workspace.exists() else None)
    for block in doc["blocks"]:
        block["lock"] = lock_state(block["name"])
    doc["policy"] = unlocks.policy()
    return BlocksDoc(**doc)
