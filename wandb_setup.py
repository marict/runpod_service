from __future__ import annotations

import os
import subprocess

import wandb


def are_local():
    return os.getenv("RUNPOD_POD_ID") is None


def _open_browser(url: str) -> None:
    chrome_candidates = [
        "google-chrome",
        "google-chrome-stable",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "chrome",
        "chromium",
        "chromium-browser",
    ]
    for cmd in chrome_candidates:
        try:
            subprocess.run([cmd, url], check=True, capture_output=True, timeout=5)
            print(f"Opened W&B URL in Chrome: {url}")
            return
        except Exception:
            continue
    raise RuntimeError(f"Could not open Chrome for {url}")


def get_required_var(var_name: str) -> str:
    if not os.getenv(var_name):
        raise RuntimeError(f"{var_name} must be set in the environment.")
    return os.getenv(var_name)


# On runpod get everything from env
def init_wandb_runpod() -> wandb.sdk.wandb_run.Run:
    try:
        api_key = get_required_var("WANDB_API_KEY")
        project = get_required_var("WANDB_PROJECT")
        entity = get_required_var("WANDB_ENTITY")
        run_id = get_required_var("WANDB_RUN_ID")
        resume = "allow" if run_id else None
        name = get_required_var("WANDB_NAME")
        notes = get_required_var("WANDB_NOTES")

        return wandb.init(
            api_key=api_key,
            project=project,
            entity=entity,
            id=run_id,
            resume=resume,
            name=name,
            notes=notes,
        )
    except Exception:
        print(f"Warning: failed to initialize W&B on runpod")
        raise


# We get project and placeholder name from runpod_launcher.py
def init_wandb_local(project: str, placeholder_name: str) -> wandb.sdk.wandb_run.Run:
    try:
        api_key = get_required_var("WANDB_API_KEY")
        entity = get_required_var("WANDB_ENTITY")

        run = wandb.init(
            name=placeholder_name,
            api_key=api_key,
            project=project,
            entity=entity,
        )

        # If run is local, open browser to logs
        if are_local():
            url = f"{run.url}/logs"
            _open_browser(url)
    except Exception:
        print(f"Warning: failed to initialize local W&B run")
        raise
    return run
