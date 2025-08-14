from __future__ import annotations

import os
from typing import Optional

import requests
import runpod

from runpod_service import wandb_setup as wandb


def on_runpod() -> bool:
    """Return True when running on a RunPod instance."""
    return os.getenv("RUNPOD_POD_ID") is not None


def stop_runpod(pod_id: Optional[str] = None, api_key: Optional[str] = None) -> bool:
    """Stop the current RunPod instance (or the provided pod_id).

    - Uses RUNPOD_POD_ID and RUNPOD_API_KEY by default.
    - Tries SDK first, falls back to REST.
    - Calls wandb.finish() before returning True.
    """
    print("Attempting to stop pod...")
    pod_id = pod_id or os.getenv("RUNPOD_POD_ID")
    api_key = api_key or os.getenv("RUNPOD_API_KEY")
    if not pod_id:
        return False
    if not api_key:
        raise ValueError("RUNPOD_API_KEY not set.")

    runpod.api_key = api_key
    if not hasattr(runpod, "stop_pod"):
        raise RuntimeError(
            "runpod.stop_pod is not available; install/upgrade runpod SDK"
        )
    runpod.stop_pod(pod_id)
    wandb.wrapper.finish()
    return True


def rename_instance(
    new_name: str, *, pod_id: Optional[str] = None, api_key: Optional[str] = None
) -> bool:
    """Rename the current RunPod instance to new_name.

    - Identifies the pod via RUNPOD_POD_ID unless pod_id is provided.
    - Requires RUNPOD_API_KEY unless api_key is provided.
    - Uses REST API: POST /v2/pods/{podId}/rename with JSON {"name": new_name}.
    """
    if not new_name or not isinstance(new_name, str):
        raise ValueError("new_name must be a non-empty string")

    pod_id = pod_id or os.getenv("RUNPOD_POD_ID")
    api_key = api_key or os.getenv("RUNPOD_API_KEY")
    if not pod_id:
        raise ValueError("RUNPOD_POD_ID not set.")
    if not api_key:
        raise ValueError("RUNPOD_API_KEY not set.")

    url = f"https://api.runpod.io/graphql?api_key={api_key}"
    # Prefer documented REST if available; fallback to GraphQL mutation
    # Using GraphQL for rename, as REST v1 doesn't expose rename.
    query = "mutation PodRename($input: PodRenameInput!) { podRename(input: $input) { id name } }"
    variables = {"input": {"podId": pod_id, "name": new_name}}
    resp = requests.post(url, json={"query": query, "variables": variables}, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"RunPod rename failed: {data['errors']}")
    renamed = (((data or {}).get("data") or {}).get("podRename") or {}).get("name")
    return renamed == new_name


def get_instance_name(
    *, pod_id: Optional[str] = None, api_key: Optional[str] = None
) -> str:
    """Return the instance (pod) name for the current RunPod instance.

    - Identifies via RUNPOD_POD_ID and RUNPOD_API_KEY by default.
    - Uses GraphQL query to fetch pod name.
    """
    pod_id = pod_id or os.getenv("RUNPOD_POD_ID")
    api_key = api_key or os.getenv("RUNPOD_API_KEY")
    if not pod_id:
        raise ValueError("RUNPOD_POD_ID not set.")
    if not api_key:
        raise ValueError("RUNPOD_API_KEY not set.")

    url = f"https://api.runpod.io/graphql?api_key={api_key}"
    query = "query Pod($id: String!) { pod(id: $id) { id name } }"
    variables = {"id": pod_id}
    resp = requests.post(url, json={"query": query, "variables": variables}, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"RunPod query failed: {data['errors']}")
    name = (((data or {}).get("data") or {}).get("pod") or {}).get("name")
    if not name:
        raise RuntimeError("RunPod returned no name for pod")
    return str(name)
