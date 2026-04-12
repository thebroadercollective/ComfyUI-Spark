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

# Run with CPU text encoder loading (needed for large models like Flux2.dev)
uv run python main.py --listen 0.0.0.0 \
  --disable-dynamic-vram --reserve-vram 1 --disable-pinned-memory \
  --disable-mmap --dont-upcast-attention \
  --bf16-unet --bf16-vae --bf16-text-enc --cpu-text-enc

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

- **`comfy/cli_args.py`** — All CLI flags parsed here. Memory-relevant flags: `--disable-mmap`, `--disable-pinned-memory`, `--disable-dynamic-vram`, `--reserve-vram`, `--bf16-unet/vae/text-enc`, `--cpu-text-enc`, `--highvram`, `--gpu-only`, `--lowvram`, `--novram`.
- **`comfy/cache_policy.py`** — Configurable cache-drop policy for model loading. `CachePhase` enum defines seven load-path phase hooks. `maybe_drop(phase)` checks the active preset/override/watermark and calls `soft_empty_cache_unified()` + optional `gc.collect()`. CLI flags: `--cache-aggressiveness {off,low,normal,high,paranoid}` (default: normal), `--cache-drop-at <phases>` (comma-separated override), `--cache-drop-threshold-gb <gb>` (pressure watermark). Call sites are in `sd.py`, `utils.py`, `model_management.py`. The module imports `model_management` at module level (safe: one-way dependency) and `cli_args.args`.
- **`comfy/model_management.py`** (~1800 lines) — Central memory management. Controls VRAM state (HIGH/NORMAL/LOW/NO/SHARED), model loading/unloading between CPU and GPU, memory estimation, and the soft/hard memory limits that trigger offloading. The `VRAMState` enum and `load_models_gpu()` function are critical. Contains memory observability helpers: `memory_report(*, device=None) -> str` returns `"alloc X.XG res X.XG free X.XG | sys avail X.XG used X.XG"` combining torch allocator + psutil views; `memory_delta(before, after) -> str` returns `"Δtorch +X.XG Δavail +X.XG"` from two snapshot strings; `soft_empty_cache_unified(force=False)` skips `ipc_collect()` on unified memory. Use these for any new load-path logging.
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

### Load-Path Observability
The load path emits INFO-level memory-accounting logs at phase boundaries. Key tags in chronological order: `LOAD` (file bookends + throttled progress ticks in `utils.py`), `CHECKPOINT` / `CHECKPOINT_SLICE` (entry/done bookends in `sd.py`), `DETECTED` (model config identification), `MODEL_INIT` / `UNET_LOADED` / `VAE_LOADED` / `CLIP_LOADED` (per-component snapshots), `LOAD_MODELS_GPU` / `POST_GC` / `MODEL_GPU_READY` (inference-start sequence in `model_management.py`), `CACHE_DROP` (from `cache_policy.maybe_drop()`). Progress ticks are gated on file size ≥5GB AND INFO level. SageAttention logs `[attention] sageattn first call ok` on first successful call and an aggregate `[attention] shutdown` summary at process exit via `atexit`.

`PAGE_CACHE_DROP` lines appear after each file load when `--drop-page-cache` is passed or on unified memory (auto-enabled). Uses `os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)` — non-privileged, per-file, Linux-only. This drops the OS page cache for the loaded safetensors file, which on unified memory competes with model tensors for the same physical pool. The helper `drop_file_page_cache(filepath)` lives in `model_management.py`. This is distinct from `soft_empty_cache_unified()` which clears the PyTorch CUDA allocator cache — the two caches are independent: `torch.cuda.empty_cache()` does not touch OS page cache, and `posix_fadvise(DONTNEED)` does not touch the CUDA allocator.

### DGX Spark Unified Memory Context
On the Spark, "CPU memory" and "GPU memory" are the same 128GB physical pool. The `--unified-memory` flag (auto-detected on GB10/sm_121) enables optimizations that eliminate two of three duplication layers:
1. **mmap page cache** — `safetensors.safe_open()` uses mmap internally (at the Rust level). Page cache pages persist even after tensor materialization.
2. **`copy=True` duplication** — ~~Eliminated~~. On unified memory, `load_torch_file()` loads directly to CUDA via `safe_open(device="cuda")`, skipping the copy. **Important**: The CUDA override only applies when no explicit `device` is passed to `load_torch_file()` (`device=None` → use unified default). Passing an explicit `device=torch.device("cpu")` bypasses the CUDA override — this is how `--cpu-text-enc` and the DualCLIPLoader `device="cpu"` option work, avoiding CUDA allocator OOM when loading large multi-component models (e.g., Flux2.dev).
3. **`load_state_dict(assign=False)` copy** — ~~Eliminated~~. `ModelPatcher.should_assign_weights()` returns `True` on unified memory, using `assign=True` to directly assign loaded tensors as model parameters.
4. **`assign=True` dtype normalization** — With `assign=False`, loaded tensor data is copied into pre-allocated parameter buffers, implicitly converting to the model's dtype. With `assign=True`, checkpoint dtypes are preserved as-is. Checkpoints with mixed dtypes (e.g., weights in bf16, biases in fp32) will produce mixed-dtype parameters. Models using `disable_weight_init` ops (like the VAE — hardcoded at `comfy/ldm/modules/diffusionmodules/model.py:10`) have no runtime dtype casting, so mixed dtypes cause `RuntimeError`. Fix: call `.to(target_dtype)` after `load_state_dict(assign=True)` to normalize. Models using `manual_cast` ops (like the diffusion model via `pick_operations()`) are safe because they cast at runtime.

See `dev-docs/dgx-spark-comfyui-loader-plan.md` for the original analysis and `docs/superpowers/specs/2026-04-05-unified-memory-loading-design.md` for the implemented design.

## Development Rules

- Use `uv` for all Python environment management. The venv is at `.venv/` under the project root.
- Linting uses `ruff` (config in `pyproject.toml`). Key ignores: E501 (line length), E722 (bare except), E402 (import order).
- Test framework is `pytest` with markers: `inference`, `execution`. Test requirements are in `tests-unit/requirements.txt`.
- Python target: >=3.10. Type annotations use 3.10+ syntax.
- `comfy/utils.py` cannot import `comfy.model_management` at module level (circular import). Use lazy import inside functions.
- `comfy/cache_policy.py` imports `comfy.model_management` and `comfy.cli_args` at module level. Call sites in `sd.py`, `utils.py`, `model_management.py` use lazy `import comfy.cache_policy as cache_policy` inside functions to avoid circular imports. Do not add a module-level import of `cache_policy` in these files.
- `comfy/cli_args.py` has `_VALID_CACHE_PHASES` (a frozenset of strings mirroring `CachePhase` enum values) used for argparse-time validation. A drift assertion in `cache_policy.py` confirms they stay in sync at import time.
- Weight assignment behavior is controlled by `ModelPatcher.should_assign_weights()` — do not check `UNIFIED_MEMORY` directly at load_state_dict call sites.
- Pyright reports many pre-existing errors (missing imports for torch, numpy, etc.) due to venv resolution. These are not real issues.
- The `--unified-memory` flag is auto-detected on GB10 (sm_121) hardware via `is_unified_memory_system()` in `comfy/model_management.py`.
- Unified memory gotcha: `safe_open(device="cuda")` loads ALL tensors to CUDA, including non-weight metadata (e.g., tokenizer vocab). Any code calling `.numpy()` on such tensors must add `.cpu()` first — e.g., `tensor.cpu().numpy()`. On unified memory `.cpu()` is essentially free.
- `load_torch_file(device=None)` uses a sentinel pattern: `None` means "use default behavior (CUDA on unified memory)", an explicit device means "honor this device". Do not add boolean flags to override device — pass the device explicitly instead.
- Text encoder device placement is controlled by `text_encoder_device()` in `model_management.py` — `load_clip()` passes this to `load_torch_file(device=...)`. The `--cpu-text-enc` flag and node-level `device="cpu"` both flow through this path.
- Unified memory gotcha: `load_state_dict(assign=True)` preserves checkpoint dtypes — if a checkpoint has mixed dtypes, the model will too. Any `load_state_dict(assign=True)` call site must follow with `.to(target_dtype)` to normalize. See `comfy/sd.py` VAE loading for the pattern.
- The VAE decoder uses `disable_weight_init` ops (hardcoded, not from `pick_operations()`), meaning `comfy_cast_weights=False` — no runtime dtype casting. The diffusion model uses `manual_cast` ops which cast at runtime. Keep this asymmetry in mind when debugging dtype errors.
- Unified memory gotcha: `load_state_dict(assign=True)` replaces pre-allocated parameter buffers — the originals become unreferenced but stay resident until `gc.collect()`. For large models (Flux2.dev FP16 = ~24GB) this transient can OOM. Fixed: `comfy/model_base.py::load_model_weights` calls `gc.collect()` gated on `assign=True` immediately after `del to_load`. Do not remove this; do not add `torch.cuda.empty_cache()` here (that's `cache_policy`'s domain).
- SageAttention runtime verification: `--use-sage-attention` swaps `optimized_attention = attention_sage` at module init (`comfy/ldm/modules/attention.py:725`). Runtime confirmation comes from (1) `[attention] sageattn first call ok` INFO log on the first successful `sageattn()` call, (2) aggregate `sageattn_calls` / `sageattn_fallbacks` counters logged at process exit via `atexit`. If the shutdown summary shows `sageattn_calls=0`, the swap failed upstream — check model-specific attention dispatch.
- Feature branches use git worktrees at `.worktrees/<branch-name>` (gitignored). Create with `git worktree add .worktrees/<name> -b <branch>`, clean up with `git worktree remove .worktrees/<name>`.
- Transformers >=5.x causes explosive memory growth on HunyuanImage3. Pinned to 4.x (exact version needs verification via `pip show transformers`). See `dev-docs/transformers-5x-memory-regression.md` for investigation plan.
- Unified memory gotcha: On Spark there are TWO independent memory caches that consume the shared pool. (1) **PyTorch CUDA allocator cache** (`torch.cuda.memory_reserved - memory_allocated`) — freed tensors that torch holds for reuse; cleared by `soft_empty_cache_unified()` / `torch.cuda.empty_cache()`. (2) **OS page cache** (`buff/cache` in `free -h`) — mmap'd file pages from `safetensors.safe_open()`; cleared by `posix_fadvise(POSIX_FADV_DONTNEED)` per-file or `echo 3 > /proc/sys/vm/drop_caches` (requires root). The gap between `torch free` and `sys avail` in `memory_report` output is primarily OS page cache. `--drop-page-cache` (auto-enabled on unified memory) addresses layer 2; `--cache-aggressiveness` addresses layer 1.
- Unified memory gotcha: `ModelPatcher.patch_weight_to_device` has two allocation hazards on unified memory. (1) The default `weight.to(offload_device)` backup is a full second allocation in the same pool — when `should_assign_weights()` is True and `inplace_update` is False, store the original `weight` in `self.backup` by reference. (2) The legacy patch pipeline builds `temp_weight = cast_to_device(..., copy=True)` as an fp32 full-model scratch and then `set_attr_param`s a fresh rounded tensor — that was another full-model allocation (the 2x floor that OOM'd Flux2+LoRA). Fix: on the `should_assign_weights()` path, if all patches for the key are purely additive constant deltas (plain LoRA/LoHa/LoKr/GLoRA without DoRA, or raw `diff` patches — see `_patches_are_invertible` in `comfy/model_patcher.py`), compute `delta = calculate_weight(patches, torch.zeros_like(weight, dtype=fp32), key)`, stochastic-round to the weight dtype, and `weight.add_(delta)` in place. Store an invertible marker (`WeightBackup(weight=None, invertible=True)`) in `self.backup`; `unpatch_model` re-derives the delta from `self.patches[key]` and calls `weight.sub_(delta)` via `_invert_fast_path_weight`. Non-invertible patches (`set`, `model_as_lora`, DoRA/OFT/BOFT/bypass, non-None `function`, `offset`, `strength_model != 1.0`, shape mismatches) fall back to the legacy copy path for correctness. Any new code path that reads `bk.weight` from `self.backup` must first check `bk.invertible` (always present on `WeightBackup`) and use `_invert_fast_path_weight` / `_compute_rounded_delta` or skip as appropriate. Critical asymmetry: `ModelPatcher.clone()` shares `self.backup` *by reference* across clones (via `get_clone_model_override`) but *copies* `self.patches`. An invertible `WeightBackup` written by a LoRA clone will later be observed by the base patcher during `unpatch_model`, where `self.patches[key]` no longer exists — so the patch list needed to recompute the delta must be stored *inside* the `WeightBackup` itself (`bk.patches`), not read from `self.patches`. All invertible callsites (`_invert_fast_path_weight`, `_compute_invertible_unpatched`, `_compute_rounded_delta`) take an explicit `patches` argument threaded from `bk.patches` for this reason.
