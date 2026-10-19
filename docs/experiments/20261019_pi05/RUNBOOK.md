# Runbook

Use a separate project directory and keep downloads, caches and results on the data volume. Follow the pinned OpenPI `uv.lock` with Python 3.11 for the policy. Keep the simulator in a separate Python 3.10 environment with `mujoco==3.2.2`, `dm-control==1.0.22`, `numpy==1.25.0`, `scipy==1.14.0`, `open3d==0.18.0`, `mediapy==1.2.0`, `h5py==3.11.0`, OpenCV, and the benchmark's required utilities. Install VLABench and `openpi-client` editably into the simulator environment.

The pinned policy source imports `pytest` at runtime; install its locked `pytest==8.3.5` even with `uv sync --no-dev`. The benchmark evaluator imports its VLM utilities, which also require `networkx`; this deployment uses `networkx==3.4.2`. These imports do not change model inference or task logic.

The benchmark also imports its unused OpenVLA adapter eagerly. The simulator environment therefore includes CPU-only `torch==2.7.1+cpu`, `transformers==4.53.2`, and `peft==0.15.2`; the actual policy runs in its separate JAX environment. Follow the upstream requirement to install `rrt-algorithms` **editably**: clone commit `e51d95ee489a225220d6ae2a764c4111f6ba7d85` and install that local directory with `uv pip install -e`. Its regular wheel omits nested modules, and uv does not accept editable Git URLs.

Download only the model's `params/`, `assets/` normalization statistics and checkpoint metadata. Its current HF repository is already the checkpoint root; the older nested path in the model-card shell example is not required. Model parameters and normalization must come from the same release.

The runner expects this layout under `<PROJECT>`: `VLABench/` (this repository), `openpi/` (the pinned policy source and its `.venv`), `envs/sim/`, `checkpoints/pi05-primitive-10task/`, and `cache/`. Runtime package inventories are recorded alongside this runbook; their `<PROJECT>` editable paths are deployment placeholders, not public download URLs.

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

Each valid video includes one observation before every executed action plus the terminal observation, with a separate `.last.jpg` four-camera mosaic. Run `python tools/pi05_eval/summarize.py /path/to/run/full` to produce `summary.json` and `summary.csv` for plotting. Track scores are task macro averages; missing/error counts remain explicit, particularly for the released cross-category track's 10-scene flower task.

On exit only this run's process sessions are terminated. The separate site resource owner checks actual GPU clearance before returning the cards to the original queued work.
