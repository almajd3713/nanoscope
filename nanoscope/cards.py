"""Ablation cards: a finished record-mode study as one small JSON file (card.v1).

A card holds the study's spec, every seed's final metrics and its provenance (the code commit
and the preregistration commit). It is what you share instead of a run folder, and what
`compare_cards` reads to put results from different people side by side with Welch intervals.
Explore-mode studies have no cards: only a claim made under record mode's guarantees is worth
publishing. Pushing a card to the Hub is opt-in and never automatic.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nanoscope import __version__
from nanoscope.compare import METRICS, UNITS, verdict_of
from nanoscope.schemas.upgrade import read_json
from nanoscope.statistics import summarize, unpaired_difference
from nanoscope.studyfiles import finals as study_finals
from nanoscope.studyfiles import study_dir
from nanoscope.studyspec import StudySpec

if TYPE_CHECKING:
    from nanoscope.study import Study


def export_card(study: Study) -> dict[str, Any]:
    """The card for a finished record-mode study; refuses explore mode and unfinished studies."""
    return card_from_files(study.to_spec())


def card_from_files(spec: StudySpec) -> dict[str, Any]:
    """The card for the study a spec describes, read from its run folders (no model is
    imported, so the API can serve it)."""
    if spec.mode != "record":
        raise ValueError("ablation cards are for record-mode studies only: explore results are "
                         "not preregistered, so they are not shared as claims")
    folder = study_dir(spec.name)
    manifest_path = folder / "study.json"
    if not manifest_path.exists():
        raise ValueError(f"study {spec.name!r} has not started: run it before exporting a card")
    done = study_finals(folder)
    names = [v.name for v in spec.variants]
    missing = [f"{v} seed {s}" for v in names for s in spec.seeds
               if str(s) not in done.get(v, {})]
    if missing:
        raise ValueError(f"study {spec.name!r} is not finished; no result yet for "
                         f"{', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''}")
    config = read_json(folder / names[0] / f"seed-{spec.seeds[0]}" / "config.json", "config")
    preset = config["preset"]
    return {
        "schema": 1, "nanoscope": __version__, "study": spec.name, "mode": "record",
        "spec": spec.to_dict(),
        "eval": {"preset": preset.get("name", spec.preset),
                 "dataset": preset["dataset"], "eval_docs": preset["eval_docs"],
                 "tokenizer": config["tokenizer"]},
        "baseline": spec.baseline,
        "finals": {v: done[v] for v in names},
        "provenance": read_json(manifest_path, "study"),
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def write_card(card: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    return path


def read_card(path: str | Path) -> dict[str, Any]:
    return read_json(Path(path), "card")


def push_plan(card: dict[str, Any], repo: str) -> dict[str, Any]:
    """Exactly what `push` would upload: the Hub dataset, the file's path in it and its text."""
    path_in_repo = f"cards/{card['study']}.json"
    return {"repo": repo, "repo_type": "dataset", "path_in_repo": path_in_repo,
            "content": json.dumps(card, indent=2),
            "commit_message": f"nanoscope: ablation card for {card['study']}"}


def push(card: dict[str, Any], repo: str) -> dict[str, Any]:
    """Upload the card to a Hub dataset repo (needs HF_TOKEN). Only ever called on request."""
    from huggingface_hub import HfApi

    plan = push_plan(card, repo)
    HfApi().upload_file(
        path_or_fileobj=plan["content"].encode("utf-8"), path_in_repo=plan["path_in_repo"],
        repo_id=repo, repo_type="dataset", commit_message=plan["commit_message"])
    return {k: v for k, v in plan.items() if k != "content"}


def compare_cards(cards: list[dict[str, Any]], metric: str = "val_bpb") -> dict[str, Any]:
    """Every variant of every card against the last card's baseline, with Welch intervals
    (cards are different people's seeds, so nothing is paired). Needs the same evaluation text."""
    if len(cards) < 2:
        raise ValueError("compare needs at least two cards")
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {', '.join(METRICS)}")
    texts = {(c["eval"]["dataset"], c["eval"]["eval_docs"]) for c in cards}
    if len(texts) > 1:
        raise ValueError(f"these cards were evaluated on different text: {sorted(texts)}")
    if metric == "val_loss" and len({c["eval"]["tokenizer"] for c in cards}) > 1:
        raise ValueError("val_loss is per token, so it can't compare tokenizers; use val_bpb")
    base_card = cards[-1]
    base_name = base_card["baseline"] or next(iter(base_card["finals"]))
    reference = [m[metric] for m in base_card["finals"][base_name].values()]
    rows = []
    for card in cards:
        for variant, seeds in card["finals"].items():
            values = [m[metric] for m in seeds.values()]
            is_base = card is base_card and variant == base_name
            delta = None if is_base else unpaired_difference(values, reference)
            rows.append({"label": f"{card['study']}/{variant}", "seeds": len(values),
                         "summary": summarize(values), "delta": delta,
                         "verdict": verdict_of(delta),
                         "commit": card["provenance"]["commit"]})
    return {"metric": metric, "baseline": f"{base_card['study']}/{base_name}", "rows": rows}


def format_comparison(result: dict[str, Any]) -> str:
    unit = UNITS[result["metric"]]
    table = [["variant", "seeds", result["metric"], f"Δ vs {result['baseline']}", ""]]
    for row in result["rows"]:
        d = row["delta"]
        if d is None:
            delta = "(baseline)"
        elif d["ci95_low"] is None:
            delta = f"{d['mean']:+.3f}"
        else:
            delta = f"{d['mean']:+.3f} [{d['ci95_low']:+.3f}, {d['ci95_high']:+.3f}]"
        table.append([row["label"], str(row["seeds"]), f"{row['summary']['mean']:.3f}",
                      delta, "" if d is None else row["verdict"] + " (unpaired)"])
    widths = [max(len(r[i]) for r in table) for i in range(len(table[0]))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip()
             for r in table]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join([f"{result['metric']} ({unit}; lower is better), cards compared with "
                      "Welch intervals", "", *lines])
