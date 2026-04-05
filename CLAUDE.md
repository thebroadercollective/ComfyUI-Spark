# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ComfyUI-Spark is a fork of [ComfyUI](https://github.com/comfyanonymous/ComfyUI) (v0.18.1) optimized for the NVIDIA DGX Spark (GB10, sm_121, 128GB unified LPDDR5x memory). The core optimization goal is eliminating redundant memory copies during model loading — on unified memory, the standard ComfyUI loading pipeline creates 2-3x memory usage because it assumes separate RAM/VRAM pools.

## Common Commands

```bash
# Run ComfyUI (DGX Spark optimized flags)
uv run python main.py --listen 0.0.0.0 \
  --disable-dynamic-vram --reserve-vram 1 --disable-pinned-memory \
  --disable-mmap --dont-upcast-attention \
  --bf16-unet --bf16-vae --bf16-text-enc

# Lint
uv run ruff check .

# Install test dependencies (not in main deps)
uv pip install pytest pytest-aiohttp pytest-asyncio

# Run all tests
uv run python -m pytest

# Run unit tests only
uv run python -m pytest tests-unit/

# Run a single test file
uv run python -m pytest tests-unit/comfy_test/some_test.py

# Run tests by marker
uv run python -m pytest -m "not inference and not execution"

# Install dependencies
uv sync
```

## Architecture

### Entry Point & Server
- `main.py` — Application entry point. Parses CLI args, initializes CUDA environment, loads nodes, starts server.
- `server.py` — aiohttp web server with WebSocket support for real-time progress updates.
- `execution.py` — Workflow execution engine. Walks the node graph, manages execution order, handles caching.

### Core Model Pipeline (`comfy/`)
The model loading and inference pipeline flows through these key files:

- **`comfy/cli_args.py`** — All CLI flags parsed here. Memory-relevant flags: `--disable-mmap`, `--disable-pinned-memory`, `--disable-dynamic-vram`, `--reserve-vram`, `--bf16-unet/vae/text-enc`, `--highvram`, `--gpu-only`, `--lowvram`, `--novram`.
- **`comfy/model_management.py`** (~1800 lines) — Central memory management. Controls VRAM state (HIGH/NORMAL/LOW/NO/SHARED), model loading/unloading between CPU and GPU, memory estimation, and the soft/hard memory limits that trigger offloading. The `VRAMState` enum and `load_models_gpu()` function are critical.
- **`comfy/model_patcher.py`** (~1700 lines) — `ModelPatcher` wraps loaded models, handles weight patching (LoRA, etc.), device movement, and memory tracking. `patch_model()` / `unpatch_model()` manage weight modifications.
- **`comfy/sd.py`** (~1850 lines) — High-level model loading. `load_diffusion_model()`, `load_clip()`, `load_vae()` orchestrate loading safetensors files into model architectures. Uses `comfy.utils.load_torch_file()` for the actual file I/O.
- **`comfy/utils.py`** (~1450 lines) — `load_torch_file()` is the universal file loading function. For safetensors, it uses `safetensors.safe_open()` with mmap. When `--disable-mmap` is set, it does `tensor.to(device=device, copy=True)` which forces a duplicate on unified memory.
- **`comfy/ops.py`** (~1200 lines) — Custom PyTorch module wrappers (`Linear`, `Conv2d`, etc.) that handle dtype casting and device placement during inference.
- **`comfy/model_detection.py`** — Identifies model architecture from state dict key patterns.
- **`comfy/supported_models.py` / `supported_models_base.py`** — Model architecture definitions and configurations.

### Node System
- **`nodes.py`** (~2500 lines) — Built-in node definitions (CheckpointLoader, KSampler, VAEDecode, etc.). Node classes define `INPUT_TYPES`, `RETURN_TYPES`, and a main function.
- **`comfy_extras/`** — Additional node modules (samplers, controlnet, audio, flux, etc.). Each `nodes_*.py` registers node classes.
- **`comfy_api_nodes/`** — Cloud API partner nodes.
- **`custom_nodes/`** — User-installed custom node packages. Each subdirectory with `__init__.py` is auto-loaded at startup.

### Execution Pipeline (`comfy_execution/`)
- `graph.py` / `graph_utils.py` — DAG topological sort and execution ordering.
- `caching.py` / `cache_provider.py` — Node output caching strategies (classic, LRU, RAM-pressure, none).

### Key Data Flow: Model Loading
1. User triggers a loader node (e.g., `CheckpointLoaderSimple` in `nodes.py`)
2. Node calls `comfy.sd.load_diffusion_model()` or similar
3. Which calls `comfy.utils.load_torch_file()` to read the safetensors file
4. State dict is passed to model detection, then used to instantiate the model architecture
5. `ModelPatcher` wraps the model for memory management
6. `model_management.load_models_gpu()` handles device placement when inference begins

### DGX Spark Unified Memory Context
On the Spark, "CPU memory" and "GPU memory" are the same 128GB physical pool. The `--unified-memory` flag (auto-detected on GB10/sm_121) enables optimizations that eliminate two of three duplication layers:
1. **mmap page cache** — `safetensors.safe_open()` uses mmap internally (at the Rust level). Page cache pages persist even after tensor materialization.
2. **`copy=True` duplication** — ~~Eliminated~~. On unified memory, `load_torch_file()` loads directly to CUDA via `safe_open(device="cuda")`, skipping the copy.
3. **`load_state_dict(assign=False)` copy** — ~~Eliminated~~. `ModelPatcher.should_assign_weights()` returns `True` on unified memory, using `assign=True` to directly assign loaded tensors as model parameters.

See `dev-docs/dgx-spark-comfyui-loader-plan.md` for the original analysis and `docs/superpowers/specs/2026-04-05-unified-memory-loading-design.md` for the implemented design.

## Development Rules

- Use `uv` for all Python environment management. The venv is at `.venv/` under the project root.
- Linting uses `ruff` (config in `pyproject.toml`). Key ignores: E501 (line length), E722 (bare except), E402 (import order).
- Test framework is `pytest` with markers: `inference`, `execution`. Test requirements are in `tests-unit/requirements.txt`.
- Python target: >=3.10. Type annotations use 3.10+ syntax.
- `comfy/utils.py` cannot import `comfy.model_management` at module level (circular import). Use lazy import inside functions.
- Weight assignment behavior is controlled by `ModelPatcher.should_assign_weights()` — do not check `UNIFIED_MEMORY` directly at load_state_dict call sites.
- Pyright reports many pre-existing errors (missing imports for torch, numpy, etc.) due to venv resolution. These are not real issues.
- The `--unified-memory` flag is auto-detected on GB10 (sm_121) hardware via `is_unified_memory_system()` in `comfy/model_management.py`.
