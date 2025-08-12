from __future__ import annotations

import json
import os
from typing import List

import wandb


def _parse_tags(raw: str | None) -> List[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "WANDB_TAGS must be a JSON array of strings, e.g. ['runpod','supervisor']."
        ) from exc
    if not isinstance(parsed, list) or not all(isinstance(t, str) for t in parsed):
        raise RuntimeError(
            "WANDB_TAGS must be a JSON array of strings, e.g. ['runpod','supervisor']."
        )
    return [t.strip() for t in parsed if t.strip()]


def _init_wandb() -> wandb.sdk.wandb_run.Run:
    api_key = os.getenv("WANDB_API_KEY")
    project = os.getenv("WANDB_PROJECT", "nalm-benchmark")
    entity = os.getenv("WANDB_ENTITY")

    if not api_key:
        raise RuntimeError("WANDB_API_KEY must be set in the environment.")
    if not entity:
        raise RuntimeError("WANDB_ENTITY must be set in the environment.")

    run_id = os.getenv("WANDB_RUN_ID")
    resume = "allow" if run_id else None
    name = os.getenv("WANDB_NAME")
    notes = os.getenv("WANDB_NOTES")
    tags = _parse_tags(os.getenv("WANDB_TAGS"))

    kwargs: dict = {"project": project, "entity": entity}
    if run_id:
        kwargs["id"] = run_id
        kwargs["resume"] = resume
    if name:
        kwargs["name"] = name
    if notes:
        kwargs["notes"] = notes
    if tags:
        kwargs["tags"] = tags

    try:
        return wandb.init(**kwargs)
    except Exception as exc:  # noqa: BLE001
        context = {
            "project": project,
            "entity": entity,
            "has_run_id": bool(run_id),
            "has_name": bool(name),
            "tags": tags,
        }
        raise RuntimeError(
            f"W&B initialization failed with context: {context}. Ensure WANDB_* env vars are correct and network access is available."
        ) from exc


# Import side-effect: initialize W&B immediately and expose a safe wrapper
run = _init_wandb()
wrapper = wandb

__all__ = ["run", "wrapper"]
