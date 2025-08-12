from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict

from runpod_service import wandb_setup as wandb


def collect_system_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    info["python_version"] = sys.version.replace("\n", " ")
    info["platform"] = platform.platform()
    info["executable"] = sys.executable
    info["cwd"] = str(Path.cwd())

    # Torch (optional) – base image includes PyTorch
    try:
        import torch  # module-level import is allowed and expected to exist in container

        info["torch_version"] = getattr(torch, "__version__", "unknown")
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name_0"] = torch.cuda.get_device_name(0)
            info["cuda_capability_0"] = ".".join(
                map(str, torch.cuda.get_device_capability(0))
            )
    except Exception as exc:  # noqa: BLE001
        info["torch_error"] = str(exc)

    # Interesting env vars
    for key in [
        "RUNPOD_POD_ID",
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_RUN_ID",
        "WANDB_RESUME",
        "HF_HOME",
        "HF_DATASETS_CACHE",
        "TRANSFORMERS_CACHE",
    ]:
        if key in os.environ:
            info[f"env_{key}"] = os.environ[key]

    # List top-level workspace and runpod-volume
    info["workspace_files"] = sorted(
        p.name for p in Path("/workspace").glob("*") if p.exists()
    )
    runpod_vol = Path("/runpod-volume")
    info["runpod_volume_present"] = runpod_vol.exists()
    if runpod_vol.exists():
        info["runpod_volume_dir_entries"] = sorted(p.name for p in runpod_vol.glob("*"))

    return info


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Service test script for runpod launcher"
    )
    parser.add_argument("positional", help="Positional argument to verify forwarding")
    parser.add_argument(
        "--named-arg", default="default", help="Named argument to verify forwarding"
    )
    args = parser.parse_args()

    wandb.init_wandb_runpod()
    print("[service_test] Starting test...")
    print(f"[service_test] Positional arg: {args.positional}")
    print(f"[service_test] Named arg: {args.named_arg}")

    sys_info = collect_system_info()
    print("[service_test] System info:")
    for k in sorted(sys_info.keys()):
        print(f"  - {k}: {sys_info[k]}")

    wandb.wrapper.config.update(
        {
            "positional": args.positional,
            "named_arg": args.named_arg,
        },
        allow_val_change=True,
    )
    wandb.wrapper.log({"service_test/heartbeat": 1, "service_test/time": time.time()})
    wandb.wrapper.log({"service_test/system_info": sys_info})

    # Write marker file to persistent volume
    out_dir = Path("/runpod-volume")
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / f"service_test_{int(time.time())}.txt"
    marker.write_text(
        f"positional={args.positional}\nnamed_arg={args.named_arg}\n",
        encoding="utf-8",
    )
    print(f"[service_test] Wrote marker file: {marker}")

    print(
        "[service_test] Done. If running under runpod_service, the pod will remain alive."
    )


if __name__ == "__main__":
    main()
