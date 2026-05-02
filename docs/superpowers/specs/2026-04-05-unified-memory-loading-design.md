# Unified Memory Model Loading Optimization — Design Spec

**Date:** 2026-04-05
**Target:** NVIDIA DGX Spark (GB10, sm_121, 128GB unified LPDDR5x)
**Goal:** Reduce model loading memory from ~3x model size to ~1x model size

---

## Problem

ComfyUI's model loading pipeline creates 2-3 copies of tensor data in memory on unified memory systems like the DGX Spark. A 60GB model consumes ~180GB peak during loading due to three layers of duplication:

| Layer | Cause | Code Location | Extra Memory |
|-------|-------|---------------|-------------|
| 1 | mmap page cache from safetensors Rust internals | `safetensors.safe_open()` | ~model size |
| 2 | `tensor.to(device, copy=True)` with `--disable-mmap` | `comfy/utils.py:138` | ~model size |
| 3 | `load_state_dict(assign=False)` copies into model buffers | `comfy/model_base.py:328`, `comfy/sd.py:1665,1806,395,824` | ~model size |

On discrete GPU systems, Layers 1-2 exist in RAM while the model lives in VRAM — separate pools. On unified memory, all three layers compete for the same 128GB.

## Solution

Load safetensors tensors directly to CUDA (eliminating Layer 2) and use `assign=True` in `load_state_dict` (eliminating Layer 3). Layer 1 (page cache) remains but is kernel-reclaimable.

**Expected peak:** ~120GB for a 60GB model (page cache + CUDA tensors)
**Expected steady state:** ~60GB (after page cache eviction)

All changes are gated behind unified memory detection and have zero impact on discrete GPU systems.

---

## Changes

### 1. Unified Memory Detection

**File:** `comfy/model_management.py`

Add detection function and module-level flag:

```python
def is_unified_memory_system():
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return props.major == 12 and props.minor == 1  # GB10 = sm_121
```

Set `UNIFIED_MEMORY` flag at module level, also checkable via new `--unified-memory` CLI flag in `comfy/cli_args.py`.

### 2. Direct CUDA Loading in `load_torch_file()`

**File:** `comfy/utils.py` (lines 122-167)

On unified memory, change safetensors loading to use `device="cuda"` instead of `device="cpu"`, and skip the `copy=True` operation:

```python
if UNIFIED_MEMORY:
    load_device_str = "cuda"
else:
    load_device_str = device.type

with safetensors.safe_open(ckpt, framework="pt", device=load_device_str) as f:
    for k in f.keys():
        tensor = f.get_tensor(k)
        if DISABLE_MMAP and not UNIFIED_MEMORY:
            tensor = tensor.to(device=device, copy=True)
        sd[k] = tensor
```

Tensors land directly on CUDA. The safetensors Rust library still mmaps the file internally, so page cache accumulates, but the explicit `copy=True` duplication is eliminated.

### 3. `assign=True` for Model Weight Loading

**Files:** `comfy/model_base.py`, `comfy/sd.py`, `comfy/clip_vision.py`

At every `load_state_dict()` call that currently uses `assign=model_patcher.is_dynamic()`, change to:

```python
assign = model_patcher.is_dynamic() or UNIFIED_MEMORY
```

Affected call sites:

| File | Line | Component |
|------|------|-----------|
| `comfy/model_base.py` | 328 | Diffusion model (via `load_model_weights`) |
| `comfy/sd.py` | 1665 | Diffusion model (checkpoint loader) |
| `comfy/sd.py` | 1806 | Diffusion model (standalone loader) |
| `comfy/sd.py` | ~395 | CLIP text encoder |
| `comfy/sd.py` | ~824 | VAE first_stage_model |
| `comfy/clip_vision.py` | ~53 | CLIP Vision model |

With `assign=True`, `load_state_dict` directly assigns loaded CUDA tensors as model parameters instead of copying them into pre-allocated buffers. The state dict tensors ARE the model parameters — zero copy.

### 4. Device Handling Adjustments

**`unet_inital_load_device()`** — `comfy/model_management.py:897-916`

On unified memory, return CUDA device to ensure model is instantiated on CUDA. This ensures `assign=True` places CUDA tensors correctly and later `m.to(device_to)` calls in the patcher are no-ops.

**`load_diffusion_model_state_dict()`** — `comfy/sd.py:1804-1806`

Skip the `model.to(offload_device)` call before weight loading on unified memory. This pre-move would place empty buffers on CPU, conflicting with CUDA tensor assignment.

### 5. Fallback Safety

Wrap the unified memory loading path in try/except. On failure, fall back to the standard loading path with a warning log. This ensures ComfyUI remains functional even if the optimized path encounters unexpected issues.

---

## Scope Boundaries

**In scope:**
- Safetensors loading via `comfy.utils.load_torch_file`
- All model types that flow through the standard loading pipeline (diffusion, VAE, CLIP, CLIP Vision, ControlNet, upscale models, etc.)
- The CLI flags: `--disable-dynamic-vram --reserve-vram 1 --disable-pinned-memory --disable-mmap --dont-upcast-attention --bf16-unet --bf16-vae --bf16-text-enc`

**Out of scope (deferred):**
- Page cache dropping (setuid binary) — not needed if kernel reclamation works
- Streaming/chunked loading for 80GB+ models — Phase 2 if needed
- HuggingFace `from_pretrained()` paths — separate loading pipeline
- Custom nodes with their own loading logic — they benefit from `load_torch_file` changes if they use it

## Files Modified

| File | Change |
|------|--------|
| `comfy/cli_args.py` | Add `--unified-memory` flag |
| `comfy/model_management.py` | Add `is_unified_memory_system()`, `UNIFIED_MEMORY` flag, modify `unet_inital_load_device()` |
| `comfy/utils.py` | Modify `load_torch_file()` for direct CUDA loading |
| `comfy/model_base.py` | Modify `load_model_weights()` to accept unified memory assign override |
| `comfy/sd.py` | Propagate `assign=True` on unified memory at ~6 call sites, skip pre-move in `load_diffusion_model_state_dict()` |
| `comfy/clip_vision.py` | Propagate `assign=True` on unified memory |

## Verification

1. Existing test suite passes unchanged (non-unified-memory paths unaffected)
2. Load FLUX.2-dev BF16 (~60GB) on DGX Spark — verify peak memory < 128GB via `free -h`
3. Steady-state memory after loading should be ~60GB (model size)
4. Run txt2img inference — output should be identical to standard loading
5. Load model, unload, reload — verify no memory leaks
6. Test with LoRA application — verify patching works correctly with assigned weights
