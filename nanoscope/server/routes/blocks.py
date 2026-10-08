from __future__ import annotations

from fastapi import APIRouter

from nanoscope import paths, queue
from nanoscope.blocks.catalog import catalog
from nanoscope.blocks.discover import discover
from nanoscope.learn import gating, unlocks
from nanoscope.server.jobs import job_doc
from nanoscope.server.models import BlocksDoc, JobDoc

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


@router.post("/blocks/{name}/certify", status_code=202)
def certify_block(name: str) -> JobDoc:
    """Check a block of the workspace against the reference it was registered with. It is a job:
    the file has to be imported to build the block, which only a worker does."""
    workspace = paths.workspace_dir()
    found = discover(workspace)["blocks"] if workspace.exists() else []
    block = next((b for b in found if b["name"] == name), None)
    if block is None:
        raise KeyError(f"no block named {name!r} is registered in the workspace")
    if block["reference"] is None:
        raise ValueError(f"{name} has no reference to be checked against: register it with "
                         "register_block(reference=fn)")
    job_id = queue.enqueue("certify", {"file": block["file"], "block": name},
                           lane="interactive")
    assert job_id is not None
    return job_doc(queue.get(job_id))
