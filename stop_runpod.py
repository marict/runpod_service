from __future__ import annotations

import os
from typing import Optional

import requests
import runpod

import runpod_service.wandb_setup as wandb


def stop_runpod(pod_id: Optional[str] = None, api_key: Optional[str] = None) -> bool:
    print("Attempting to stop pod...")
    # Try via SDK then REST fallback
    pod_id = pod_id or os.getenv("RUNPOD_POD_ID")
    api_key = api_key or os.getenv("RUNPOD_API_KEY")
    if not pod_id:
        return False
    if not api_key:
        raise ValueError("RUNPOD_API_KEY not set.")

    try:
        runpod.api_key = api_key
        if hasattr(runpod, "stop_pod"):
            runpod.stop_pod(pod_id)
            wandb.finish()
            return True
    except Exception:
        pass

    try:
        url = f"https://rest.runpod.io/v1/pods/{pod_id}/stop"
        headers = {"Authorization": f"Bearer {api_key}"}
        wandb.finish()
        resp = requests.post(url, headers=headers, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to stop pod: {exc}")
        return False
