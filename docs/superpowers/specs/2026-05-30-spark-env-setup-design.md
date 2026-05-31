# Spark Environment Setup & SageAttention Build — Design

**Date:** 2026-05-30
**Status:** Approved (design), pending implementation
**Scope decision:** Fix in place + document (do NOT recreate `.venv`)

## Problem

1. **Immediate blocker:** Installing the locally-built `SageAttention/` fails to compile —
   every translation unit dies on `fatal error: Python.h: No such file or directory`.
2. **Standing confusion** the user wants resolved:
   - pip vs uv
   - `requirements.txt` vs `pyproject.toml`
   - how to manage/aggregate multiple `requirements.txt` files (each custom node ships one).

## Findings (verified 2026-05-30)

- `.venv` (5.5 GB) is built on **system Python 3.12.3** (`/usr/bin/python3`); `pyvenv.cfg`
  has `home = /usr/bin`. Its header dir `/usr/include/python3.12/` has **no `Python.h`** —
  `python3.12-dev` / `python3-dev` are **not installed**. That is the entire SageAttention
  build failure: the compiler can find torch + CUDA but not the CPython headers.
- **`pip` is not installed** in the venv — only `uv pip` works.
- The env was built by **`uv sync` from `pyproject.toml` + `uv.lock`**. Installed versions
  match the pyproject pins (`comfyui-frontend-package==1.42.8`, `comfyui-workflow-templates==0.9.44`,
  `comfyui-embedded-docs==0.4.3`, `comfy-aimdo==0.4.7` from `>=0.2.12`), **not** the
  `requirements.txt` values (1.44.19 / 0.9.85 / 0.5.1 / ==0.4.5). Therefore:
  **`pyproject.toml` (+ `uv.lock`) is canonical; `requirements.txt` is vestigial** and does
  not reflect the running environment. (`diffusers==0.37.1` is the lone extra, a custom-node dep.)
- **torch is reproducible, not a fragile nightly.** `torch==2.12.0` comes from **PyPI**; the
  aarch64 wheel (`torch-2.12.0-cp312-cp312-manylinux_2_28_aarch64.whl`) is **CUDA-13-bundled**
  by default (pulls `nvidia-cudnn-cu13` etc.), which is why `torch.__version__` is `2.12.0+cu130`.
  `triton==3.7.0` is a normal PyPI dep. The **only** local source build is SageAttention.
- GPU stack works: **NVIDIA GB10, capability (12,1) = sm_121, CUDA 13.0 available**.
- Disk: 3.1 TB free.
- The Hunyuan node (`custom_nodes/Comfy_HunyuanImage3/requirements.txt`) declares
  `transformers>=4.47.0` with **no upper bound** — installing it naively pulls transformers 5.x,
  which CLAUDE.md documents as breaking HunyuanImage3 (pinned to `4.57.3`). This is the concrete
  multi-requirements hazard.

## Decisions

| Decision | Choice |
|---|---|
| Recreate `.venv`? | **No.** Fix in place. torch/triton are reproducible; the only gap is headers. |
| Future-rebuild Python | **uv-managed Python 3.12** (`uv venv --python 3.12`) — ships its own headers, no apt/sudo, immune to this exact failure. |
| requirements/pyproject drift | **Document only.** Name `pyproject.toml` canonical; mark `requirements.txt` as not-used. No file edits to dependency declarations. |

## Deliverables

### 1. Unblock SageAttention (manual + assisted)
- **User runs (sudo):** `sudo apt install -y python3.12-dev`
  → provides `/usr/include/python3.12/Python.h`.
- Remove stale cross-machine build: `rm -rf SageAttention/build SageAttention/*.egg-info`.
- Install reusing in-tree compile, no build isolation (setup.py imports torch at build time):
  `uv pip install --no-build-isolation -e ./SageAttention`.
- **Verify:** `python -c "import sageattention"` succeeds; on a ComfyUI run with
  `--use-sage-attention`, log shows `[attention] sageattn first call ok`.

### 2. `dev-docs/environment-setup.md` (the teaching doc)
Sections:
- **Package manager:** uv only; `pip` is absent → always `uv pip …`. Env is `.venv/`.
- **Canonical deps:** `pyproject.toml` + `uv.lock`, applied via `uv sync`. `requirements.txt`
  is upstream/vestigial and does **not** match the running env — do not install from it.
- **torch:** `torch==2.12.0` from PyPI is CUDA-13-bundled on aarch64; reproducible. Pin it so a
  stray install can't upgrade it out from under the GPU stack.
- **Multiple requirements files → constraints file.** Don't merge them. Install each one under a
  shared constraints file that pins load-bearing versions:
  `uv pip install -r custom_nodes/<node>/requirements.txt -c constraints.txt`.
  The constraints force `transformers==4.57.3`, neutralizing Hunyuan's unbounded `>=4.47.0`.

### 3. `constraints.txt` (repo root)
Pins the load-bearing versions used as a `-c` file:
`transformers==4.57.3`, `torch==2.12.0`, `numpy==2.4.6`.

### 4. `scripts/setup-env.sh` (rerunnable bootstrap)
Documents the correct from-scratch order so "the right way" is captured as code:
1. `uv venv --python 3.12` (self-contained headers).
2. `uv sync` (canonical deps from pyproject + lock).
3. Install needed custom-node deps with `-c constraints.txt`.
4. Build SageAttention: `uv pip install --no-build-isolation -e ./SageAttention`.
5. Smoke check (`torch.cuda.is_available()`, device name, `import sageattention`).

### 5. Safety net
Commit `dev-docs/env/spark-env-snapshot-2026-05-30.txt` — the 102-package `uv pip freeze` —
as a known-good manifest.

## Verification / success criteria
- `import sageattention` works in `.venv`.
- ComfyUI launches with `--use-sage-attention` and logs the sageattn-OK line.
- `dev-docs/environment-setup.md`, `constraints.txt`, `scripts/setup-env.sh`, and the snapshot exist.
- No changes to `pyproject.toml`/`requirements.txt` dependency declarations.
- Working torch/GPU stack untouched.

## Out of scope
- Recreating `.venv`.
- Reconciling requirements/pyproject drift (documented only).
- Re-porting the dropped HunyuanImage3 page-cache patch (tracked separately in `dev-docs/spark-port-pending/`).
