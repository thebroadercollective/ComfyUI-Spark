# Environment Setup (DGX Spark)

This document explains how Python dependencies work in this repo, so you don't have
to rederive it each time. It answers four recurring questions: pip vs uv,
`requirements.txt` vs `pyproject.toml`, how to handle the many per-custom-node
`requirements.txt` files, and how to (re)build the locally-compiled SageAttention.

Verified against the working environment on 2026-05-30.

---

## TL;DR

- **Use `uv`, not `pip`.** `pip` is not installed in `.venv`. Run `uv pip …`.
- **`pyproject.toml` + `uv.lock` are the source of truth.** The env is built with
  `uv sync`. `requirements.txt` is upstream baggage and does **not** match what's
  installed — don't install from it.
- **torch is reproducible.** `torch==2.12.0` from PyPI ships CUDA 13 baked in on
  aarch64 (hence `2.12.0+cu130`). It is not a fragile nightly.
- **For custom-node requirements, install them under `-c constraints.txt`** so an
  unbounded dep (like Hunyuan's `transformers>=4.47.0`) can't break the fork.
- **SageAttention is the only thing compiled from local source.** It needs the
  CPython dev headers (`Python.h`) present.

---

## 1. uv vs pip

This project uses [uv](https://docs.astral.sh/uv/) for everything. The virtualenv
lives at `.venv/` under the repo root.

- `pip` is intentionally **not installed** inside `.venv`. `python -m pip …` will
  report zero packages or fail. That's expected.
- Always use the `uv` equivalents:

  | Want to… | Command |
  |---|---|
  | Install/refresh the whole env from the lock | `uv sync` |
  | Install one package | `uv pip install <pkg>` |
  | Install from a requirements file | `uv pip install -r <file>` |
  | List installed packages | `uv pip freeze` |
  | Run something in the env | `uv run <cmd>` (or activate `.venv` first) |

`uv pip …` operates on the **active** `.venv`. If you've `source .venv/bin/activate`'d
(the `comfyui` prompt), commands target it automatically.

## 2. requirements.txt vs pyproject.toml — which is canonical?

**`pyproject.toml` (+ `uv.lock`) is canonical here.** The environment was built by
`uv sync`, which reads `pyproject.toml` and pins exact versions in `uv.lock`.

Evidence (2026-05-30): the installed versions match the **pyproject** pins, not
`requirements.txt`:

| Package | Installed (= pyproject) | requirements.txt says |
|---|---|---|
| comfyui-frontend-package | 1.42.8 | 1.44.19 |
| comfyui-workflow-templates | 0.9.44 | 0.9.85 |
| comfyui-embedded-docs | 0.4.3 | 0.5.1 |
| comfy-aimdo | 0.4.7 (from `>=0.2.12`) | ==0.4.5 |

Updated (2026-08-16, after the v0.33.0 sync): the whole frontend asset trio
(`comfyui-frontend-package` 1.42.8 → **1.49.6**, `comfyui-workflow-templates`
0.9.44 → **0.11.41**, `comfyui-embedded-docs` 0.4.3 → **0.5.10**) is now
reconciled with upstream; the frontend had fallen seven minor versions behind the
backend. Only `comfy-aimdo` still drifts, by design:

| Package | Installed (= pyproject) | requirements.txt says |
|---|---|---|
| comfyui-frontend-package | **1.49.6** | 1.49.6 (reconciled) |
| comfyui-workflow-templates | **0.11.41** | 0.11.41 (reconciled) |
| comfyui-embedded-docs | **0.5.10** | 0.5.10 (reconciled) |
| comfy-aimdo | 0.4.10 (from `>=0.2.12`) | ==0.4.13 |

`comfy-aimdo` is deliberately not chased — aimdo/DynamicVRAM is off on GB10 (see
CLAUDE.md), so only its import-time API surface matters. `comfy-kitchen` was
raised to `>=0.2.31` during the same sync because `comfy/ldm/modules/attention.py`
now calls `comfy_kitchen.int8_attention_is_available()` at import time, and
`comfy-angle` was added because `comfy_extras/nodes_glsl.py` imports it.

So treat `requirements.txt` as **vestigial upstream baggage**. It is kept for
compatibility with stock-ComfyUI tooling but does **not** describe this env. Don't
`uv pip install -r requirements.txt` — it would fight the locked versions.

> The two files have drifted. We deliberately did **not** reconcile them (low-risk
> "document only" decision). If you ever do reconcile, change `pyproject.toml` and
> re-run `uv sync` / regenerate `uv.lock` — that's the canonical side.

To change a top-level dependency: edit `pyproject.toml`, then `uv sync` (or
`uv lock` then `uv sync`). To bump everything to latest allowed: `uv lock --upgrade`.

## 3. Many requirements.txt files — use a constraints file, don't merge them

Each custom node ships its own `requirements.txt`. You do **not** merge them into one
giant file. The clean way to "aggregate" is a **constraints file** plus per-node installs.

A constraints file (`-c`) installs nothing on its own; it only **caps** the versions a
resolve is allowed to pick. We keep one at the repo root: `constraints.txt`.

Install a custom node's deps like this:

```bash
uv pip install -r custom_nodes/Comfy_HunyuanImage3/requirements.txt -c constraints.txt
```

Why it matters: the Hunyuan node declares `transformers>=4.47.0` with **no upper
bound**. Installed naively, that pulls transformers 5.x, which CLAUDE.md documents as
breaking HunyuanImage3. `constraints.txt` pins `transformers==4.57.3`, so the install
is forced to honor the working version. It also protects `torch` and `numpy` the same way.

Rule of thumb: **any time you install a third-party requirements.txt, add
`-c constraints.txt`.** If a node truly needs a newer pinned package, decide
deliberately and update `constraints.txt` — don't bypass it.

## 4. Building SageAttention (local source compile)

SageAttention lives at `./SageAttention/` and is the **only** dependency built from
source. Its `setup.py` imports `torch` at build time and compiles CUDA/C++ extensions,
so it needs:

1. **The torch already in `.venv`** → build with `--no-build-isolation` (otherwise uv
   builds in a throwaway env without your torch).
2. **CPython development headers** (`Python.h`). The current `.venv` runs on **system
   Python 3.12** (`/usr/bin/python3`), whose headers come from the OS package
   `python3.12-dev`. If that package is missing you get:
   `fatal error: Python.h: No such file or directory`.

Install/refresh the headers (one-time, needs sudo):

```bash
sudo apt install -y python3.12-dev
# provides /usr/include/python3.12/Python.h
```

Then build:

```bash
rm -rf SageAttention/build SageAttention/*.egg-info   # clear any stale/cross-machine build
uv pip install --no-build-isolation -e ./SageAttention
```

Verify:

```bash
uv run python -c "import sageattention; print('sageattention OK')"
```

At runtime, launching ComfyUI with `--use-sage-attention` logs
`[attention] sageattn first call ok` on the first attention call, and an aggregate
`sageattn_calls` / `sageattn_fallbacks` summary at process exit. If `sageattn_calls=0`
at shutdown, the kernel swap didn't engage — see CLAUDE.md's SageAttention notes.

> **Future rebuilds avoid the header problem entirely.** `scripts/setup-env.sh`
> creates the venv with a **uv-managed** Python 3.12 (`uv venv --python 3.12`), which
> ships its own headers — no apt/sudo needed. The current `.venv` predates that and
> uses system Python, so for *now* we just apt-install the headers.

## 5. Reproducibility / safety net

- `uv.lock` pins exact versions of everything resolvable from registries.
- `dev-docs/env/spark-env-snapshot-2026-05-30.txt` is a known-good `uv pip freeze`
  (102 packages) captured 2026-05-30 — a manual restore reference if the lock ever
  drifts or a resolve goes wrong.
- A from-scratch rebuild is scripted in `scripts/setup-env.sh`.
