## RunPod Service Launcher

Launch a GPU pod on RunPod and run a Python script from your git repo at the exact current commit. The launcher ensures a clean repo, installs `requirements_dev.txt`, and keeps logs on a persistent volume. A small helper is provided to stop pods.

### Requirements
- RunPod account and `RUNPOD_API_KEY`
- Your script lives in a clean git repo (HEAD pushed to origin)
- `requirements_dev.txt` at the repo root
- Python 3.10+

### Install
```bash
pip install -r requirements_dev.txt
```
Optional environment variables:
```bash
export RUNPOD_API_KEY="your_runpod_api_key"
export GITHUB_TOKEN="your_github_token"   # for private repos (optional)
export WANDB_API_KEY="your_wandb_api_key" # for Weights & Biases (optional)
```

### Quick start
```bash
python runpod_launcher.py /abs/path/to/repo/path/to/script.py \
  --pod-name my-job \
  --gpu-type "NVIDIA RTX 2000 Ada Generation" \
  --wandb-project my-project \
  --lifetime-minutes 60 \
  -- --arg1 value1 --flag
```
- Use `--` to separate launcher flags from your script's args.
- If `--pod-name` is omitted, the script filename is used.

### Common options
- `--pod-name`: Pod name (defaults to script filename)
- `--gpu-type`: GPU type or id (default: NVIDIA RTX 2000 Ada Generation)
- `--api-key`: RunPod API key (defaults to `RUNPOD_API_KEY`)
- `--wandb-project`: W&B project (default: `nalm-benchmark`)
- `--lifetime-minutes`: Auto-stop pod after N minutes
- `--debug`: Print generated container script and args

### Stop a pod programmatically
```python
from runpod_service import stop_runpod

ok = stop_runpod(pod_id="your_pod_id", api_key="your_runpod_api_key")
print("stopped" if ok else "failed")
```

### Notes
- No attach mode: this launcher only creates new pods.
- Anything written to `/runpod-volume` persists on the configured network volume.
- For private GitHub repos, set `GITHUB_TOKEN` so the container can clone.

That’s it—simple, predictable pod launches for GPU workloads.
