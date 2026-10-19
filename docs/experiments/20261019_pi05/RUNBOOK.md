# Runbook

Use a separate project directory and keep downloads, caches and results on the data volume. Follow the pinned OpenPI `uv.lock` with Python 3.11 for the policy. Keep the simulator in a separate Python 3.10 environment with `mujoco==3.2.2`, `dm-control==1.0.22`, `numpy==1.25.0`, `scipy==1.14.0`, `open3d==0.18.0`, `mediapy==1.2.0`, `h5py==3.11.0`, OpenCV, and the benchmark's required utilities. Install VLABench and `openpi-client` editably into the simulator environment.

Download only the model's `params/`, `assets/` normalization statistics and checkpoint metadata. Its current HF repository is already the checkpoint root; the older nested path in the model-card shell example is not required. Model parameters and normalization must come from the same release.

The [LeRobot CI asset mirror](https://github.com/huggingface/lerobot/blob/main/docker/Dockerfile.benchmark.vlabench) avoids Google Drive quotas. Download its pinned `obj/` and `scenes/` into VLABench's assets. Git packs the many small files efficiently; large LFS objects still need materializing.

Create a private JSON configuration with:

```json
{
  "project": "/path/to/project",
  "source": "/path/to/project/VLABench",
  "output": "/path/to/project/runs/unique-run",
  "gpus": [4, 5],
  "gpu_uuids": {"4": "GPU-allocated-4", "5": "GPU-allocated-5"},
  "scope_libraries": {"4": "/path/to/validated-egl-g4-marker.so", "5": "/path/to/validated-egl-g5-marker.so"},
  "ports": {"4": 63140, "5": 63141},
  "workers_per_gpu": 2,
  "no_progress_timeout_seconds": 900
}
```

The EGL marker libraries refer to this deployment's pre-existing NVIDIA application profiles (`EGLVisibleDGPUDevices` masks 16/32). They are resource isolation, not model code. On a different machine use its validated physical-device binding, then inspect both C and G process placement; CUDA masking alone is insufficient for rendering.

```bash
python tools/pi05_eval/run_sweep.py --config /path/to/private-config.json --mode smoke
# Inspect four real episode results, final-frame videos and gpu-scope.json.
python tools/pi05_eval/run_sweep.py --config /path/to/private-config.json --mode full
```

Each task writes `episode-NNNN.json`, error JSONs, complete native videos, `progress.json`, native `metrics.json`, and `job-result.json`. The sweep writes its complete plan, process identities, per-lane results and final summary. Three consecutive initialization/runtime exceptions stop that task; 15 minutes without progress terminates that task's owned process session. A common policy-server failure stops dependent jobs for repair. Completed outputs remain available.

On exit only this run's process sessions are terminated. The separate site resource owner checks actual GPU clearance before returning the cards to the original queued work.
