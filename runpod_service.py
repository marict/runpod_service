from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import threading
import time
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
NETWORK_VOLUME_ID = "h3tyejvqqb"

GPU_TO_PRICE = {
    "NVIDIA RTX 2000 Ada Generation": 0.24,
}


@dataclass
class LaunchConfig:
    script_path: Path
    script_args: List[str]
    pod_name: Optional[str]
    gpu_type: str
    api_key: Optional[str]
    wandb_project: str
    debug: bool = False
    lifetime_minutes: int | None = None


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


def _normalize_repo_url(repo_url: str) -> str:
    """Return an HTTPS clone URL suitable for a headless container.

    - Converts SSH URLs (git@github.com:owner/repo.git) to HTTPS
    - If GITHUB_TOKEN is set, embeds token for private repos
    """
    url = repo_url.strip()
    if url.startswith("git@github.com:"):
        owner_repo = url.split(":", 1)[1]
        url = f"https://github.com/{owner_repo}"
    elif url.startswith("ssh://git@github.com/"):
        owner_repo = url.split("github.com/", 1)[1]
        url = f"https://github.com/{owner_repo}"

    token = os.getenv("GITHUB_TOKEN")
    if token and url.startswith("https://github.com/"):
        # GitHub accepts x-access-token as username for token auth
        url = url.replace(
            "https://github.com/", f"https://x-access-token:{token}@github.com/"
        )
    return url


def _get_current_branch(repo_root: Path) -> Optional[str]:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        branch = res.stdout.strip()
        return None if branch == "HEAD" else branch
    except subprocess.CalledProcessError:
        return None


def _branch_matches_origin(repo_root: Path, branch: str, commit_hash: str) -> bool:
    try:
        res = subprocess.run(
            ["git", "ls-remote", "origin", branch],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        line = res.stdout.strip().splitlines()[0] if res.stdout.strip() else ""
        remote_sha = line.split()[0] if line else ""
        return bool(remote_sha) and remote_sha == commit_hash
    except subprocess.CalledProcessError:
        return False


def _get_origin_branch_sha(repo_root: Path, branch: str) -> Optional[str]:
    try:
        res = subprocess.run(
            ["git", "ls-remote", "origin", branch],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        line = res.stdout.strip().splitlines()[0] if res.stdout.strip() else ""
        return line.split()[0] if line else None
    except subprocess.CalledProcessError:
        return None


def _quote_bash_for_graphql(command: str) -> str:
    # Wrap into bash -c and escape properly for GraphQL string literal
    wrapped = f"bash -c {shlex.quote(command)}"
    return print_string(wrapped)[1:-1]


def _join_shell_args(args: Iterable[str]) -> str:
    return " ".join(shlex.quote(a) for a in args)


def _build_container_script(
    *,
    repo_url: Optional[str],
    commit_hash: Optional[str],
    script_relpath: Optional[Path],
    forwarded_args: List[str],
    runpod_service_repo_url: str,
    runpod_service_commit: str,
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
    cmds.append("cd /workspace || true")
    cmds.append("REPO_DIR=/tmp/repo")
    cmds.append("export PIP_CACHE_DIR='/runpod-volume/pip-cache'")
    cmds.append("mkdir -p '$PIP_CACHE_DIR'")
    # Safer PyTorch CUDA defaults to reduce chances of segfaults / OOM
    cmds.append("export CUDA_LAUNCH_BLOCKING=1")
    cmds.append("export TORCHINDUCTOR_AUTOTUNE=0")
    cmds.append("export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512")
    # Avoid unsupported Flash/MemEfficient kernels in SDPA by default
    cmds.append("export PYTORCH_SDP_ATTENTION=math")

    # Clone, checkout exact commit, install dev requirements
    cmds.append("echo '[RUNPOD] Cloning target repository into /tmp/repo...'")
    cmds.append(f'rm -rf "$REPO_DIR" && git clone {shlex.quote(repo_url)} "$REPO_DIR"')
    cmds.append('cd "$REPO_DIR"')
    cmds.append(f"git checkout {shlex.quote(commit_hash)}")
    cmds.append("python -m pip install --upgrade pip setuptools wheel")
    cmds.append(
        '[ -f "$REPO_DIR/requirements_dev.txt" ] || { echo "[RUNPOD] ERROR: requirements_dev.txt missing at repo root: $REPO_DIR/requirements_dev.txt"; ls -la "$REPO_DIR"; exit 1; }'
    )
    cmds.append('pip install -r "$REPO_DIR/requirements_dev.txt"')

    # Always ensure runpod_service (auxiliary) repo is available
    cmds.append(f"REPO_URL={shlex.quote(repo_url)}")
    cmds.append(f"REPO_COMMIT={shlex.quote(commit_hash)}")
    cmds.append(f"RUNPOD_SERVICE_REPO_URL={shlex.quote(runpod_service_repo_url)}")
    cmds.append(f"RUNPOD_SERVICE_COMMIT={shlex.quote(runpod_service_commit)}")
    cmds.append("RUNPOD_SERVICE_DIR=/opt/runpod_service_repo")
    cmds.append(
        'if [ "$RUNPOD_SERVICE_REPO_URL" = "$REPO_URL" ] || [ -z "$RUNPOD_SERVICE_REPO_URL" ]; then '
        'RUNPOD_SERVICE_DIR="$REPO_DIR"; '
        'echo "[RUNPOD] Using target repo as runpod_service"; '
        'else echo "[RUNPOD] Cloning runpod_service repo into $RUNPOD_SERVICE_DIR..."; '
        'rm -rf "$RUNPOD_SERVICE_DIR"; git clone "$RUNPOD_SERVICE_REPO_URL" "$RUNPOD_SERVICE_DIR"; '
        '(cd "$RUNPOD_SERVICE_DIR" && git checkout "$RUNPOD_SERVICE_COMMIT" ); fi'
    )
    script_log = f"/runpod-volume/{script_relpath.name}_$(date +%Y%m%d_%H%M%S).log"
    cmds.append(f'log_file="{script_log}"')
    cmds.append("export log_file")
    # Ensure repository roots are on PYTHONPATH so top-level modules resolve (target + runpod_service)
    cmds.append('export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"')
    cmds.append(
        'if [ -f "$RUNPOD_SERVICE_DIR/runpod_service.py" ] && [ -f "$RUNPOD_SERVICE_DIR/__init__.py" ]; then '
        '  export PYTHONPATH="$RUNPOD_SERVICE_DIR:$PYTHONPATH"; '
        'else echo "[RUNPOD] ERROR: expected flat runpod_service module at $RUNPOD_SERVICE_DIR (missing runpod_service.py/__init__.py)"; ls -la "$RUNPOD_SERVICE_DIR"; exit 1; fi'
    )
    # Install runpod_service repo requirements if it is a different repo than the target
    cmds.append(
        '[ "$RUNPOD_SERVICE_DIR" = "$REPO_DIR" ] || { '
        '[ -f "$RUNPOD_SERVICE_DIR/requirements_dev.txt" ] '
        '|| { echo "[RUNPOD] ERROR: requirements_dev.txt missing at $RUNPOD_SERVICE_DIR"; ls -la "$RUNPOD_SERVICE_DIR"; exit 1; }; '
        'pip install -r "$RUNPOD_SERVICE_DIR/requirements_dev.txt"; }'
    )
    cmds.append("echo '[RUNPOD] Launching script in repo...'")
    cmds.append(
        f'python -u {shlex.quote(str(script_relpath))} {_join_shell_args(forwarded_args)} 2>&1 | tee "$log_file" || true'
    )
    cmds.append("tail -f /dev/null")
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


def _init_local_wandb(project: str, run_name: str) -> Tuple[str, str]:
    run = wandb.init(
        project=project,
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
        "  # Launch new pod\n"
        "  python /Users/paul_curry/ai2/runpod_service/runpod_service.py "
        "/Users/paul_curry/ai2/runpod_service/service_test.py hello --named-arg world --pod-name service-test\n"
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
    parser.add_argument("--pod-name")
    parser.add_argument("--gpu-type", default=DEFAULT_GPU_TYPE)
    parser.add_argument("--api-key")
    parser.add_argument("--wandb-project", default="nalm-benchmark")
    parser.add_argument(
        "--lifetime-minutes",
        type=int,
        default=None,
        help="Lifetime in minutes; if set, the launcher will stop the pod after this many minutes and print estimated cost.",
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
        debug=known.debug,
        lifetime_minutes=known.lifetime_minutes,
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
            f"Script {script_abs} must be inside a git repository. Ensure your script path is within a repo and retry."
        )
    repo_root: Optional[Path] = _repo_root_for(script_dir)
    repo_url: Optional[str] = None
    commit_hash: Optional[str] = None
    script_relpath: Optional[Path] = None

    if repo_root is not None:
        _ensure_git_clean(repo_root)
        repo_url = _git_remote_url(repo_root)
        commit_hash = _git_commit_hash(repo_root)
        branch = _get_current_branch(repo_root)
        if branch and not _branch_matches_origin(repo_root, branch, commit_hash):
            remote_sha = _get_origin_branch_sha(repo_root, branch)
            raise RunPodError(
                (
                    f"Preflight failed: local HEAD {commit_hash[:7]} on branch '{branch}' "
                    f"does not match origin/{branch} {remote_sha[:7] if remote_sha else 'missing'}.\n"
                    f"We clone from origin and then checkout that exact commit. Push your branch so that "
                    f"origin/{branch} contains {commit_hash} (e.g., `git push origin HEAD:{branch}`)."
                )
            )
        try:
            script_relpath = script_abs.relative_to(repo_root)
        except ValueError:
            script_relpath = Path(script_basename)
        # Ensure requirements_dev.txt exists at repo root prior to pod creation
        requirements_path = repo_root / "requirements_dev.txt"
        if not requirements_path.exists():
            raise RunPodError(
                f"requirements_dev.txt not found at repository root: {requirements_path}. "
                "This file is required for environment setup."
            )

    # Init W&B locally first and open browser for new pod creation
    placeholder_name = f"pod-id-pending{'-' + pod_name if pod_name else ''}"
    wandb_url = ""
    wandb_run_id = ""
    try:
        wandb_url, wandb_run_id = _init_local_wandb(cfg.wandb_project, placeholder_name)
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: failed to initialize local W&B run: {exc}")

    # Build container script
    clone_url = _normalize_repo_url(repo_url or "") if repo_url else None
    # Also prepare cloning the runpod_service repo itself so its APIs are available inside the container
    runpod_service_root = Path(__file__).resolve().parent
    if not _is_git_repo(runpod_service_root):
        raise RunPodError(
            f"runpod_service must be inside a git repository (got: {runpod_service_root})."
        )
    runpod_service_url = _git_remote_url(runpod_service_root)
    if not runpod_service_url:
        raise RunPodError("Failed to resolve runpod_service git remote URL (origin).")
    runpod_service_commit = _git_commit_hash(runpod_service_root)
    if not runpod_service_commit:
        raise RunPodError("Failed to resolve runpod_service git commit hash.")
    runpod_service_clone = _normalize_repo_url(runpod_service_url)

    container_script = _build_container_script(
        repo_url=clone_url,
        commit_hash=commit_hash,
        script_relpath=script_relpath,
        forwarded_args=cfg.script_args,
        runpod_service_repo_url=runpod_service_clone,
        runpod_service_commit=runpod_service_commit,
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

    # Create the pod with simple retries to handle transient API/capacity errors
    pod = None
    for attempt in range(1, 4):
        try:
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
            break
        except Exception as exc:  # noqa: BLE001
            if attempt >= 3:
                raise
            delay_s = 5 * attempt
            print(
                f"[RUNPOD] create_pod failed (attempt {attempt}/3): {exc}. Retrying in {delay_s}s..."
            )
            time.sleep(delay_s)

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

    # Optional lifetime enforcement with cost estimate
    if cfg.lifetime_minutes and cfg.lifetime_minutes > 0:
        if cfg.gpu_type in GPU_TO_PRICE:
            hourly_price = GPU_TO_PRICE[cfg.gpu_type]
            minutes_price = hourly_price / 60
            print(
                f"[RUNPOD] Estimated cost for {cfg.lifetime_minutes:.2f}m: ${minutes_price * cfg.lifetime_minutes:.2f}, at ${hourly_price:.2f}/h"
            )
        else:
            print(
                f"[RUNPOD] GPU type {cfg.gpu_type} does not have price in GPU_TO_PRICE"
            )

        def _stop_later():
            try:
                sleep_s = max(1, int(cfg.lifetime_minutes * 60))
                time.sleep(sleep_s)
                print("[RUNPOD] Lifetime reached; stopping pod...")
                stop_runpod(pod_id=pod_id, api_key=runpod.api_key)
            except Exception as exc:
                print(f"[RUNPOD] Failed to stop pod after lifetime: {exc}")

        t = threading.Thread(target=_stop_later, daemon=False)
        t.start()

    return str(pod_id)


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


def main(argv: List[str]) -> None:
    # argparse within _parse_cli handles -h/--help and errors itself.
    cfg = _parse_cli(argv)
    start_runpod_job(cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
