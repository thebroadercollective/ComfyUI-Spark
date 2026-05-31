#!/usr/bin/env bash
#
# setup-env.sh — rebuild the DGX Spark ComfyUI environment from scratch, the right way.
#
# This captures the correct order of operations so it's code, not tribal knowledge.
# It is rerunnable. See dev-docs/environment-setup.md for the full explanation of
# uv vs pip, pyproject vs requirements, and the constraints-file technique.
#
# Run from the repo root:  ./scripts/setup-env.sh
#
set -euo pipefail

cd "$(dirname "$0")/.."
REPO="$(pwd)"
echo "==> Repo: $REPO"

# 1. Create the virtualenv on a uv-MANAGED Python 3.12.
#    uv downloads a self-contained CPython that ships its own headers (Python.h),
#    so SageAttention compiles without any apt/sudo. This is the key improvement
#    over the original system-Python venv.
echo "==> [1/5] Creating .venv on uv-managed Python 3.12"
uv venv --python 3.12

# 2. Install the canonical dependencies from pyproject.toml + uv.lock.
#    This is the source of truth for this repo (NOT requirements.txt).
echo "==> [2/5] Installing locked dependencies (uv sync)"
uv sync

# 3. Install custom-node dependencies UNDER the shared constraints file so an
#    unbounded dep (e.g. Hunyuan's `transformers>=4.47.0`) can't break the fork.
#    Add a line here for each custom node that ships a requirements.txt you need.
echo "==> [3/5] Installing custom-node deps under constraints.txt"
if [ -f custom_nodes/Comfy_HunyuanImage3/requirements.txt ]; then
  uv pip install -r custom_nodes/Comfy_HunyuanImage3/requirements.txt -c constraints.txt
fi

# 4. Build SageAttention from local source.
#    --no-build-isolation so the build sees the torch we just installed.
#    On a uv-managed Python the headers are present; no apt package required.
echo "==> [4/5] Building SageAttention (local source)"
if [ -d SageAttention ]; then
  rm -rf SageAttention/build SageAttention/*.egg-info
  uv pip install --no-build-isolation -e ./SageAttention
else
  echo "    (SageAttention/ not present; skipping)"
fi

# 5. Smoke test.
echo "==> [5/5] Smoke test"
uv run python - <<'PY'
import torch
print("torch          :", torch.__version__)
print("cuda available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device         :", torch.cuda.get_device_name(0))
    print("capability     :", torch.cuda.get_device_capability(0))
try:
    import sageattention  # noqa: F401
    print("sageattention  : import OK")
except Exception as e:
    print("sageattention  : FAILED ->", e)
PY

echo "==> Done. Launch ComfyUI with the flags documented in CLAUDE.md."
