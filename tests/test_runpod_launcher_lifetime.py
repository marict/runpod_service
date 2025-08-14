from __future__ import annotations

import sys
import types
from pathlib import Path

import runpod_service.runpod_launcher as launcher


def test_build_container_script_contains_timeout_and_tail():
    # Import here to avoid global side effects during collection
    # Provide minimal dummies for external deps
    sys.modules.setdefault("runpod", types.ModuleType("runpod"))
    gql_lang_mod = types.ModuleType("graphql.language.print_string")

    def _print_string(x):
        return x

    gql_lang_mod.print_string = _print_string  # type: ignore[attr-defined]
    sys.modules.setdefault("graphql", types.ModuleType("graphql"))
    sys.modules.setdefault("graphql.language", types.ModuleType("graphql.language"))
    sys.modules["graphql.language.print_string"] = gql_lang_mod

    script_rel = Path("experiments/op_supervisor.py")
    script = launcher._build_container_script(
        repo_url="https://github.com/example/repo.git",
        commit_hash="deadbeef",
        script_relpath=script_rel,
        forwarded_args=["--foo", "bar"],
        runpod_service_repo_url="https://github.com/example/runpod_service.git",
        runpod_service_commit="cafebabe",
    )

    # The script should include the lifetime conditional and timeout wrapper
    assert (
        'if [ -n "${LIFETIME_MINUTES:-}" ] && [ "${LIFETIME_MINUTES}" -gt 0 ]' in script
    )
    assert 'timeout "${LIFETIME_MINUTES}m"' in script
    # And also the keep-alive tail path when lifetime is not set
    assert "tail -f /dev/null" in script


def test_start_runpod_job_passes_lifetime_into_env(tmp_path, monkeypatch):
    # Provide minimal dummies for external deps before import
    sys.modules.setdefault("runpod", types.ModuleType("runpod"))
    gql_lang_mod = types.ModuleType("graphql.language.print_string")

    def _print_string(x):
        return x

    gql_lang_mod.print_string = _print_string  # type: ignore[attr-defined]
    sys.modules.setdefault("graphql", types.ModuleType("graphql"))
    sys.modules.setdefault("graphql.language", types.ModuleType("graphql.language"))
    sys.modules["graphql.language.print_string"] = gql_lang_mod

    import runpod_service.runpod_launcher as launcher

    # Create a fake git repo root with requirements_dev.txt
    repo_root: Path = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "requirements_dev.txt").write_text("pytest\n")

    # Fake script inside the repo
    script_path = repo_root / "experiments" / "op_supervisor.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("print('hello')\n")

    # Monkeypatch git helpers to avoid real git
    monkeypatch.setattr(launcher, "_is_git_repo", lambda p: True)
    monkeypatch.setattr(launcher, "_repo_root_for", lambda p: repo_root)
    monkeypatch.setattr(launcher, "_ensure_git_clean", lambda p: None)
    monkeypatch.setattr(
        launcher, "_git_remote_url", lambda p: "https://github.com/example/repo.git"
    )
    monkeypatch.setattr(launcher, "_git_commit_hash", lambda p: "deadbeef")
    monkeypatch.setattr(launcher, "_get_current_branch", lambda p: "main")
    monkeypatch.setattr(
        launcher, "_branch_matches_origin", lambda *args, **kwargs: True
    )

    # Avoid hitting RunPod and GPU listing
    monkeypatch.setattr(launcher, "_resolve_gpu_id", lambda gpu: "gpu-1")

    # Capture env passed to create_pod
    captured = {}

    def fake_create_pod(**kwargs):  # type: ignore[no-redef]
        captured.update(kwargs)
        return {"id": "pod-123"}

    monkeypatch.setattr(launcher.runpod, "create_pod", fake_create_pod)
    monkeypatch.setattr(
        launcher.runpod,
        "get_gpus",
        lambda: [{"id": "gpu-1", "displayName": "NVIDIA RTX 2000 Ada Generation"}],
    )
    monkeypatch.setattr(launcher.runpod, "api_key", "test-key")

    # Stub W&B init
    class DummyRun:
        def __init__(self):
            self.url = "https://wandb.test/run"
            self.id = "run-abc"
            self.name = "pending"

    monkeypatch.setattr(launcher.wandb, "init_wandb", lambda project, name: DummyRun())

    # Prevent lifetime thread from actually starting/sleeping
    class DummyThread:
        def __init__(self, target=None, daemon=None):
            self.target = target
            self.daemon = daemon

        def start(self):  # do nothing
            return None

    monkeypatch.setattr(launcher.threading, "Thread", DummyThread)

    # Ensure required WANDB env vars exist
    monkeypatch.setenv("WANDB_API_KEY", "key")
    monkeypatch.setenv("WANDB_ENTITY", "entity")

    cfg = launcher.LaunchConfig(
        script_path=script_path,
        script_args=["--flag"],
        pod_name="test-pod",
        gpu_type="NVIDIA RTX 2000 Ada Generation",
        api_key="test-key",
        wandb_project="nalm-benchmark",
        debug=False,
        lifetime_minutes=60,
    )

    pod_id = launcher.start_runpod_job(cfg)
    assert pod_id == "pod-123"

    # Assert LIFETIME_MINUTES is injected into container env
    assert "env" in captured
    assert captured["env"]["LIFETIME_MINUTES"] == "60"
