from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import requests
import runpod
from graphql.language.print_string import print_string

import wandb


class RunPodError(Exception):
    """Custom exception for RunPod operations."""


# Defaults aligned with existing projects
DEFAULT_GPU_TYPE = "NVIDIA RTX 2000 Ada Generation"
DEFAULT_IMAGE = "runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04"
NETWORK_VOLUME_ID = "h3tyejvqqb"  # Same as nanoGPT-llm-extraction


@dataclass
class LaunchConfig:
    script_path: Path
    script_args: List[str]
    pod_name: Optional[str]
    gpu_type: str
    api_key: Optional[str]
    wandb_project: str
    wandb_entity: Optional[str]
    debug: bool = False


def _is_git_repo(path: Path) -> bool:
    try:
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=str(path),
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return True
    except Exception:
        return False


def _repo_root_for(path: Path) -> Optional[Path]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(path),
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return Path(result.stdout.strip())
    except Exception:
        return None


def _ensure_git_clean(repo_root: Path) -> None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover - defensive
        raise RunPodError(f"Failed to check git status: {exc}") from exc

    if result.stdout.strip():
        raise RunPodError(
            "Uncommitted changes detected in local repository. Commit or stash before launching."
        )


def _git_remote_url(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise RunPodError("Failed to get git remote 'origin' URL.") from exc


def _git_commit_hash(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise RunPodError("Failed to get current git commit hash.") from exc


def _quote_bash_for_graphql(command: str) -> str:
    # Wrap into bash -c and escape properly for GraphQL string literal
    wrapped = f"bash -c {shlex.quote(command)}"
    return print_string(wrapped)[1:-1]


def _join_shell_args(args: Iterable[str]) -> str:
    return " ".join(shlex.quote(a) for a in args)


def _build_container_script(
    *,
    has_repo: bool,
    repo_url: Optional[str],
    commit_hash: Optional[str],
    script_relpath: Optional[Path],
    original_script_arg: str,
    forwarded_args: List[str],
) -> str:
    # 0) Clean up NVIDIA/CUDA APT sources to avoid hash-mismatch errors
    nvidia_repo_cleanup = "rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list || true"

    cmds: List[str] = []
    cmds.append("set -euo pipefail")
    cmds.append("exec 2>&1")
    cmds.append("mountpoint -q /runpod-volume || echo '/runpod-volume not mounted'")
    cmds.append("echo '[RUNPOD] Starting container setup...'")
    cmds.append("echo '[RUNPOD] Cleaning up NVIDIA/CUDA APT sources...'")
    cmds.append(nvidia_repo_cleanup)
    cmds.append("apt-get update -y || true")
    cmds.append("apt-get install -y --no-install-recommends git tree htop || true")
    cmds.append("cd /workspace")
    cmds.append("export PIP_CACHE_DIR='/runpod-volume/pip-cache'")
    cmds.append("mkdir -p '$PIP_CACHE_DIR'")
    # Safer PyTorch CUDA defaults to reduce chances of segfaults / OOM
    cmds.append("export CUDA_LAUNCH_BLOCKING=1")
    cmds.append("export TORCHINDUCTOR_AUTOTUNE=0")
    cmds.append("export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512")

    if has_repo and repo_url and commit_hash and script_relpath is not None:
        # Clone, checkout exact commit, install dev requirements if present
        cmds.append("echo '[RUNPOD] Cloning repository...'")
        cmds.append(f"git clone {shlex.quote(repo_url)} repo")
        cmds.append("cd /workspace/repo")
        cmds.append(f"git checkout {shlex.quote(commit_hash)}")
        cmds.append("python -m pip install --upgrade pip setuptools wheel")
        cmds.append(
            "if [ -f requirements_dev.txt ]; then pip install -r requirements_dev.txt; fi"
        )
        script_log = f"/runpod-volume/{script_relpath.name}_$(date +%Y%m%d_%H%M%S).log"
        script_cmd = (
            f'log_file="{script_log}"; '
            f'python -u {shlex.quote(str(script_relpath))} {_join_shell_args(forwarded_args)} 2>&1 | tee "$log_file" || true'
        ).rstrip()
        cmds.append("echo '[RUNPOD] Launching script in repo...'")
        cmds.append(script_cmd)
        cmds.append("tail -f /dev/null")
    else:
        # We now always require a git repo; this path should not be reachable
        cmds.append("echo '[RUNPOD] ERROR: script is not inside a git repo' && false")

    return " && ".join(cmds)


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
    print(f"Could not open Chrome. Please manually visit: {url}")


def _init_local_wandb(
    project: str, entity: Optional[str], run_name: str
) -> Tuple[str, str]:
    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        tags=["runpod", "general"],
        notes=run_name,
    )
    url = f"{run.url}/logs"
    _open_browser(url)
    return url, run.id


def _resolve_gpu_id(gpu_type: str) -> str:
    try:
        gpus = runpod.get_gpus()
    except Exception as exc:  # pragma: no cover - network errors
        raise RunPodError(f"Failed to list GPUs: {exc}") from exc
    for gpu in gpus:
        if gpu_type in {gpu.get("id"), gpu.get("displayName")}:
            return str(gpu["id"])
    raise RunPodError(f"GPU type '{gpu_type}' not found")


def _parse_cli(argv: List[str]) -> LaunchConfig:
    example = (
        "\nExample:\n"
        "  python /Users/paul_curry/ai2/runpod_general/runpod_service.py "
        "/Users/paul_curry/ai2/runpod_general/service_test.py hello --named-arg world --pod-name service-test\n"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Launch a RunPod job that clones the script's git repo at the current commit, "
            "installs requirements_dev.txt if present, and runs the script with given args.\n"
            "Note: <script.py> must reside within a git repository with no uncommitted changes."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=example,
    )
    parser.add_argument("script", help="Path to the Python script inside a git repo")
    parser.add_argument("--pod-name", dest="pod_name")
    parser.add_argument("--gpu-type", dest="gpu_type", default=DEFAULT_GPU_TYPE)
    parser.add_argument("--api-key", dest="api_key")
    parser.add_argument(
        "--wandb-project", dest="wandb_project", default="nalm-benchmark"
    )
    parser.add_argument(
        "--wandb-entity", dest="wandb_entity", default="paul-michael-curry-productions"
    )
    parser.add_argument("--debug", action="store_true", help="Print debug info")

    # Parse known args; forward the rest to the script
    known, unknown = parser.parse_known_args(argv)

    return LaunchConfig(
        script_path=Path(known.script),
        script_args=unknown,
        pod_name=known.pod_name,
        gpu_type=known.gpu_type,
        api_key=known.api_key,
        wandb_project=known.wandb_project,
        wandb_entity=known.wandb_entity,
        debug=known.debug,
    )


def start_runpod_job(cfg: LaunchConfig) -> str:
    # Determine pod name from script basename if not provided
    script_basename = cfg.script_path.name
    pod_name = cfg.pod_name or script_basename

    # Detect git repo for script
    script_abs = cfg.script_path.resolve()
    script_dir = script_abs.parent

    has_repo = _is_git_repo(script_dir) and (_repo_root_for(script_dir) is not None)
    if not has_repo:
        raise RunPodError(
            "Script must be inside a git repository. Ensure your script path is within a repo and retry."
        )
    repo_root: Optional[Path] = _repo_root_for(script_dir) if has_repo else None
    repo_url: Optional[str] = None
    commit_hash: Optional[str] = None
    script_relpath: Optional[Path] = None

    if repo_root is not None:
        _ensure_git_clean(repo_root)
        repo_url = _git_remote_url(repo_root)
        commit_hash = _git_commit_hash(repo_root)
        try:
            script_relpath = script_abs.relative_to(repo_root)
        except ValueError:
            # If not inside repo root (shouldn't happen if has_repo True), fallback to basename
            script_relpath = Path(script_basename)

    # Init W&B locally first and open browser
    placeholder_name = f"pod-id-pending{'-' + pod_name if pod_name else ''}"
    wandb_url = ""
    wandb_run_id = ""
    try:
        wandb_url, wandb_run_id = _init_local_wandb(
            cfg.wandb_project, cfg.wandb_entity, placeholder_name
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: failed to initialize local W&B run: {exc}")

    # Build container script
    container_script = _build_container_script(
        has_repo=True,
        repo_url=repo_url,
        commit_hash=commit_hash,
        script_relpath=script_relpath,
        original_script_arg=str(cfg.script_path),
        forwarded_args=cfg.script_args,
    )

    docker_args = _quote_bash_for_graphql(container_script)
    if cfg.debug:
        print("=== DEBUG: Docker script ===")
        print(container_script)
        print("=== DEBUG: Final docker args ===")
        print(docker_args)
        print("=== END DEBUG ===")

    # Setup RunPod
    runpod.api_key = (
        cfg.api_key or os.getenv("RUNPOD_API_KEY") or getattr(runpod, "api_key", None)
    )
    if not runpod.api_key:
        raise RunPodError(
            "RunPod API key is required. Provide via --api-key or set RUNPOD_API_KEY"
        )

    gpu_type_id = _resolve_gpu_id(cfg.gpu_type)

    env_vars = {}
    # Pass through WANDB context
    if os.getenv("WANDB_API_KEY"):
        env_vars["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY", "")
    env_vars["WANDB_PROJECT"] = cfg.wandb_project
    if cfg.wandb_entity:
        env_vars["WANDB_ENTITY"] = cfg.wandb_entity
    if wandb_run_id:
        env_vars["WANDB_RUN_ID"] = wandb_run_id
        env_vars["WANDB_RESUME"] = "allow"
    # HuggingFace caches (keep parity with nanoGPT service)
    env_vars.update(
        {
            "HF_HOME": "/workspace/.cache/huggingface",
            "HF_DATASETS_CACHE": "/workspace/.cache/huggingface/datasets",
            "TRANSFORMERS_CACHE": "/workspace/.cache/huggingface/transformers",
        }
    )

    # Create the pod
    pod = runpod.create_pod(
        name=pod_name,
        image_name=DEFAULT_IMAGE,
        gpu_type_id=gpu_type_id,
        gpu_count=1,
        min_vcpu_count=6,
        min_memory_in_gb=16,
        volume_in_gb=1000,
        container_disk_in_gb=1000,
        network_volume_id=NETWORK_VOLUME_ID,
        env=env_vars,
        start_ssh=False,
        docker_args=docker_args,
    )

    pod_id = pod.get("id") if isinstance(pod, dict) else None
    if not pod_id:
        raise RunPodError("RunPod API did not return a pod id")

    # Provide a quick status/poll to confirm launch visibility
    console_url = f"https://www.runpod.io/console/pods/{pod_id}"
    print(f"RunPod console: {console_url}")

    # Rename W&B run to include actual pod id (and pod name)
    try:
        if wandb.run is not None:
            final_name = f"{pod_id}-{pod_name}" if pod_name else str(pod_id)
            wandb.run.name = final_name
            print(f"W&B run renamed to: {final_name}")
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: failed to rename W&B run: {exc}")

    print(f"Starting job '{pod_name}' (pod {pod_id}) on {cfg.gpu_type}")
    print(f"W&B: {wandb_url}")
    return str(pod_id)


def stop_runpod(pod_id: Optional[str] = None, api_key: Optional[str] = None) -> bool:
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


def main(argv: List[str]) -> None:
    # argparse within _parse_cli handles -h/--help and errors itself.
    cfg = _parse_cli(argv)
    start_runpod_job(cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
