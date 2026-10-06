from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from nanoscope import compare as compare_runs
from nanoscope.server.models import ComparisonDoc

router = APIRouter(prefix="/api", tags=["compare"])


class CompareRequest(BaseModel):
    """What to compare: run refs (a set of seeds, or one run), the shipped baselines by name
    (`gpt2`, with `preset`), and which one the others are measured against."""

    sets: list[str]
    baseline: str | None = None
    metric: str = "val_bpb"
    preset: str | None = None


@router.post("/compare")
def compare(body: CompareRequest) -> ComparisonDoc:
    """Rows with the difference from the baseline and its 95% interval, a verdict word per row,
    every seed's curve and the precision plan: exactly `nanoscope.compare(...).to_dict()`."""
    result = compare_runs(*body.sets, baseline=body.baseline, preset=body.preset,
                          metric=body.metric)
    return ComparisonDoc.model_validate(result.to_dict())
