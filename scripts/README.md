# RoboDojo scripts

## Public entry points

| Script | Purpose |
| --- | --- |
| [robodojo.sh](robodojo.sh) | Main CLI: `doctor`, `eval`, `client`, `smoke`, `benchmark`, `dimensions`, `summarize`, `tasks` |
| [install.sh](install.sh) | User-local setup (`uv`, Python 3.11, Isaac Sim, pinned submodules); no `apt` or Conda |
| [init_assets.sh](init_assets.sh) | Download robot/object assets |
| [eval_policy.sh](eval_policy.sh) | Isaac Sim eval client (called by `robodojo.sh client` and XPolicyLab) |

## Typical eval flow

For a native install without root access:

```bash
bash scripts/install.sh --install
source .venv/bin/activate
bash scripts/init_assets.sh
bash scripts/robodojo.sh doctor --sim-env .venv
```

The installer writes only beneath this checkout (`.tools/`, `.venv/`, `.cache/`, and
the Git submodule directories). It needs existing `curl`, Git LFS, a C/C++ compiler,
`make`, and the host NVIDIA driver. It installs CMake and Ninja into `.venv`.
Because this host has CUDA 13 while RoboDojo's PyTorch wheel uses CUDA 12.8,
the installer downloads the NVIDIA CUDA 12.8 runfile and installs **toolkit only**
under `.tools/cuda-12.8` if a CUDA 12.8 toolkit is not already available. The
runfile download is several gigabytes; it never requests a driver install.
Use `--from STEP` to resume after a failed download or build. Running an evaluation
also requires a policy server and assets; Isaac Sim camera rendering must be tested
on the host GPU before assuming the benchmark works.

```text
robodojo.sh eval
  -> scripts/internal/run_policy_eval.sh
    -> policy server (localhost) + sim client

Split / multi-machine (see docs/SPLIT_EVAL.md):

robodojo.sh server  ->  scripts/internal/run_policy_server.sh  ->  policy server (bind 0.0.0.0)
robodojo.sh client  ->  scripts/eval_policy.sh  ->  src/eval_client/main.py
```

Run one or more official capability dimensions in a benchmark sweep:

```bash
bash scripts/robodojo.sh dimensions
bash scripts/robodojo.sh benchmark \
  --dimension memory,long-horizon \
  --policy-dir XPolicyLab/policy/<POLICY> \
  --ckpt <CHECKPOINT> \
  --policy-env <ENV> \
  --eval-num native
```

Available dimensions are `generalization`, `memory`, `precision`,
`long-horizon`, and `open`. Generalization includes both the 12 standard tasks
and their 12 runnable `_random` layout variants. Combine `--dimension` with
`--only` or `--tasks-file` to narrow a dimension further.

## Auto multi-GPU grouping

`robodojo.sh smoke` and `robodojo.sh benchmark` now support balanced
multi-GPU execution driven by the runtime table in `../optimal_8group.txt`.
Pass a concrete GPU id list and RoboDojo will partition the selected tasks
online instead of relying on a hard-coded task group.

Dry-run example:

```bash
bash scripts/robodojo.sh benchmark \
  --policy-dir XPolicyLab/policy/ACT \
  --ckpt test_ckpt \
  --policy-env RoboDojo \
  --eval-num 1 \
  --gpu-ids 0,2,5,7 \
  --dry-run
```

Launch only a subset of tasks:

```bash
bash scripts/robodojo.sh benchmark \
  --policy-dir XPolicyLab/policy/ACT \
  --ckpt test_ckpt \
  --policy-env RoboDojo \
  --eval-num native \
  --only imitate_sorting_sequence,pour_by_language,play_tic_tac_toe \
  --gpu-ids 0,1,3
```

## Internal (`internal/`)

Not intended for direct daily use. Called by `robodojo.sh` or policy utilities.

| File | Called by |
| --- | --- |
| [verify_install.sh](internal/verify_install.sh) | `robodojo.sh doctor` |
| [task_inventory.py](internal/task_inventory.py) | `robodojo.sh tasks` |
| [smoke_all_tasks.sh](internal/smoke_all_tasks.sh) | `robodojo.sh smoke` / `benchmark` |
| [summarize_result.py](internal/summarize_result.py) | `robodojo.sh summarize` |
| [stat_score_distribution.py](internal/stat_score_distribution.py) | Offline score histogram analysis (manual) |

## Docker

Container install and smoke tests live under [../docker/](../docker/), not here.

## Policy-specific scripts

Training, data prep, and per-policy `eval.sh` live in [../XPolicyLab/policy/](../XPolicyLab/policy/) (submodule).
