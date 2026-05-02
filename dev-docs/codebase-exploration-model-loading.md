# ComfyUI Model Loading Pipeline — Deep Analysis for DGX Spark Optimization

**Date:** April 5, 2026
**Purpose:** Document the full model loading pipeline to identify all points of memory duplication relevant to unified memory optimization.

---

## 1. Overview: Loading Pipeline Flow

When a user triggers a loader node (e.g., `CheckpointLoaderSimple`), the following chain executes:

```
nodes.py: CheckpointLoaderSimple.load_checkpoint()
  → comfy/utils.py: load_torch_file(ckpt_path)        # Step A: Read file from disk
  → comfy/sd.py: load_state_dict_guess_config(sd)      # Step B: Detect model type
    → comfy/model_detection.py: model_config_from_unet()
    → comfy/model_base.py: model.load_model_weights()   # Step C: Assign weights to model
      → nn.Module.load_state_dict(sd, assign=...)       # Step D: PyTorch weight assignment
    → comfy/sd.py: VAE(sd=vae_sd)                       # Step E: VAE instantiation
    → comfy/sd.py: CLIP(state_dict=clip_sd)             # Step F: CLIP instantiation
  → comfy/model_management.py: load_models_gpu()        # Step G: Device placement
    → comfy/model_patcher.py: load()                    # Step H: Weight patching & movement
```

---

## 2. Step A: File I/O — `comfy/utils.py:122-167`

### Path 1: AIMDO enabled (Dynamic VRAM active)
- `load_safetensors()` (line 85-119) memory-maps the file via `comfy_aimdo.model_mmap.ModelMMAP`
- Tensors are created as read-only views via `torch.frombuffer()` — zero-copy at load time
- **NOT relevant to our case** — AIMDO is disabled by `--disable-dynamic-vram`

### Path 2: Standard safetensors (our case with `--disable-mmap`)
```python
with safetensors.safe_open(ckpt, framework="pt", device=device.type) as f:
    for k in f.keys():
        tensor = f.get_tensor(k)             # Tensor backed by mmap'd file pages
        if DISABLE_MMAP:
            tensor = tensor.to(device=device, copy=True)  # COPY #1: Forces physical copy
        sd[k] = tensor
```

**Memory impact with `--disable-mmap`:**
- `safe_open()` internally uses mmap at the Rust level — file pages enter kernel page cache
- `f.get_tensor(k)` returns a tensor backed by mmap'd pages
- `tensor.to(device=device, copy=True)` allocates NEW memory and copies data
- The mmap'd tensor is released when the `safe_open` context manager exits
- BUT the kernel page cache retains the file-backed pages

**Result:** Model data exists in TWO places:
1. Kernel page cache (file-backed mmap pages from safetensors Rust internals)
2. The actual tensors in the state_dict (from `copy=True`)

On unified memory, both consume from the same 128GB pool.

### Path 3: Standard safetensors WITHOUT `--disable-mmap`
- Same as above but WITHOUT the `copy=True` — tensors remain backed by mmap pages
- Memory is single-copy BUT tensors are read-only mmap views
- When tensors are later moved to GPU or copied during model instantiation, the mmap pages persist in page cache

---

## 3. Step B-C: Model Detection & Weight Assignment

### State dict splitting (`comfy/sd.py:1614-1712`)
The unified state dict from load_torch_file is split into:
- **UNET weights** — extracted via `sd.pop(k)` (removes from original dict, zero-copy)
- **VAE weights** — extracted via `state_dict_prefix_replace(..., filter_keys=True)` (new dict, tensor references only)
- **CLIP weights** — extracted via `model_config.process_clip_state_dict(sd)` (new dict, tensor references only)

**No tensor duplication here** — only dict structure operations.

### Model instantiation (`comfy/sd.py:1661-1665`)
```python
inital_load_device = model_management.unet_inital_load_device(parameters, unet_dtype)
model = model_config.get_model(sd, diffusion_model_prefix, device=inital_load_device)
model.load_model_weights(sd, diffusion_model_prefix, assign=model_patcher.is_dynamic())
```

**Critical:** `inital_load_device` determination:
- With `--disable-dynamic-vram`: `is_dynamic()` returns `False`
- With our flags: `unet_inital_load_device()` checks free GPU mem vs free CPU mem
  - If `DISABLE_SMART_MEMORY` is not set and VRAM state is NORMAL_VRAM:
    returns GPU if model fits, else CPU
  - On DGX Spark: GPU and CPU are same physical memory, so this is about PyTorch device tracking

### Weight assignment (`comfy/model_base.py:320-335`)
```python
def load_model_weights(self, sd, unet_prefix="", assign=False):
    to_load = {}
    for k in keys:
        if k.startswith(unet_prefix):
            to_load[k[len(unet_prefix):]] = sd.pop(k)  # References, not copies
    m, u = self.diffusion_model.load_state_dict(to_load, strict=False, assign=assign)
```

**The `assign` parameter is THE critical factor:**

- **`assign=False` (our case, since `is_dynamic()=False`):**
  PyTorch's `load_state_dict` with `assign=False` calls `.copy_()` on each parameter buffer.
  The model was instantiated with empty/random parameter buffers. Each buffer gets filled via
  `param.data.copy_(loaded_tensor)`. This means:
  - Model parameters: allocated during model instantiation (meta device or actual allocation)
  - State dict tensors: the loaded data
  - **COPY #2:** `copy_()` copies state dict tensor data into model parameter buffers
  - After `load_model_weights()`, the `to_load` dict and its tensors can be GC'd

- **`assign=True` (dynamic/AIMDO case):**
  PyTorch directly assigns the loaded tensor as the parameter — no copy.

---

## 4. Step D: Model Instantiation Device

Models are instantiated with `device=inital_load_device`. On the Spark with our flags:

```python
def unet_inital_load_device(parameters, dtype):
    # aimdo_enabled is False (--disable-dynamic-vram)
    # vram_state is NORMAL_VRAM (no --highvram, --lowvram, etc.)
    # DISABLE_SMART_MEMORY is False (no --disable-smart-memory)
    
    model_size = dtype_size(dtype) * parameters
    mem_dev = get_free_memory(torch_dev)  # CUDA free mem
    mem_cpu = get_free_memory(cpu_dev)    # RAM free mem
    
    if mem_dev > mem_cpu and model_size < mem_dev:
        return torch_dev  # Load directly to GPU
    else:
        return cpu_dev    # Load to CPU first
```

On unified memory, `mem_dev` and `mem_cpu` report from the same pool but may differ
due to CUDA allocator reservations. The model will likely load to CPU first.

If model loads to CPU, then `load_state_dict(assign=False)` copies tensors into CPU-resident
model parameters. Later, `load_models_gpu()` moves the model to GPU — which on unified
memory is a no-op physically but PyTorch still tracks device placement.

---

## 5. Step E-F: VAE and CLIP Loading

### VAE (`comfy/sd.py:437-560`)
- VAE is instantiated with `load_state_dict(sd, strict=False, assign=self.patcher.is_dynamic())`
- Same `assign=False` issue — copies tensors into model buffers

### CLIP (`comfy/sd.py:208-266`)
- Similar pattern — `load_state_dict` with `assign=` based on `is_dynamic()`

---

## 6. Step G-H: Device Placement & Weight Patching

### `load_models_gpu()` (`comfy/model_management.py:718-822`)
When model needs to run inference:
1. Calculates memory budget
2. Frees other models if needed
3. Calls `loaded_model.model_load()` → `model_patcher.load()`

### `model_patcher.load()` (`comfy/model_patcher.py:766-893`)
For each module, decides: load to GPU fully, or use LowVramPatch (lazy loading).

**For modules loaded completely** (line 846-862):
```python
for param in params:
    self.patch_weight_to_device(key, device_to=device_to)  # Potential COPY #3
m.to(device_to)  # Moves remaining buffers
```

### `patch_weight_to_device()` (`comfy/model_patcher.py:684-712`)
If the key has LoRA/adapter patches:
```python
# Backup original weight to offload device
self.backup[key] = (weight.to(device=self.offload_device, copy=inplace_update), ...)  # COPY if inplace

# Cast weight for LoRA computation
temp_weight = cast_to_device(weight, device_to, temp_dtype, copy=True)  # COPY #3

# Calculate patched weight
out_weight = comfy.lora.calculate_weight(self.patches[key], temp_weight, key)

# Replace model parameter
set_attr_param(self.model, key, out_weight)  # or copy_to_param
```

**Without LoRA patches:** `patch_weight_to_device` returns early (line 686-687), no extra copies.

---

## 7. Summary: Memory Copies During Loading (DGX Spark, No LoRA)

For a standard checkpoint load with our CLI flags:

| Copy | Location | What | Size | Avoidable? |
|------|----------|------|------|------------|
| Page Cache | safetensors Rust lib | mmap file-backed pages | ~60GB | Yes, with non-mmap loading |
| Copy #1 | `utils.py:138` | `tensor.to(copy=True)` from mmap → CPU tensor | ~60GB | Yes, use `copy=False` |
| Copy #2 | `model_base.py:328` | `load_state_dict(assign=False)` copies into model params | ~60GB | Yes, use `assign=True` |
| Total Peak | | Page cache + state dict + model params | ~180GB | Could be ~60GB |

**After GC of intermediates:**
- Page cache: ~60GB (persists until dropped or evicted)
- Model parameters: ~60GB (needed for inference)
- Total steady state: ~120GB for a 60GB model

---

## 8. CLI Flag Effects on Loading

### `--disable-dynamic-vram`
- Prevents AIMDO initialization → `aimdo_enabled = False`
- `CoreModelPatcher` stays as base `ModelPatcher` (not `ModelPatcherDynamic`)
- `is_dynamic()` returns `False` → `load_state_dict(assign=False)` → copies weights

### `--reserve-vram 1`
- Sets `EXTRA_RESERVED_VRAM = 1GB`
- Affects `minimum_inference_memory()` = 0.8GB + 1GB = 1.8GB reserved
- Reduces available memory budget for model loading

### `--disable-pinned-memory`
- `MAX_PINNED_MEMORY = -1` → no pinned memory allocation
- Prevents host-locked memory buffers (irrelevant on unified memory)

### `--disable-mmap`
- Sets `DISABLE_MMAP = True` in `comfy/utils.py`
- Triggers `tensor.to(device=device, copy=True)` for every tensor during safetensors load
- Intended to avoid mmap issues but actually ADDS a copy on unified memory

### `--bf16-unet`, `--bf16-vae`, `--bf16-text-enc`
- Forces bf16 dtype for respective components
- Prevents upcasting to fp32 which would double memory
- Good for unified memory — keeps data compact

### `--dont-upcast-attention`
- Prevents attention computation from upcasting to fp32
- Saves significant temporary memory during inference

---

## 9. Key Files Reference

| File | Lines | Purpose |
|------|-------|---------|
| `comfy/utils.py` | 85-167 | File I/O, safetensors/torch loading |
| `comfy/sd.py` | 1614-1712 | Checkpoint guess config, splits state dict |
| `comfy/sd.py` | 1715-1819 | Diffusion model loading |
| `comfy/model_base.py` | 320-335 | Weight assignment to model |
| `comfy/model_management.py` | 574-593 | model_load() orchestration |
| `comfy/model_management.py` | 718-822 | load_models_gpu() |
| `comfy/model_management.py` | 897-916 | Initial load device selection |
| `comfy/model_management.py` | 1290-1320 | cast_to() / cast_to_device() |
| `comfy/model_patcher.py` | 684-712 | patch_weight_to_device() |
| `comfy/model_patcher.py` | 766-893 | load() — selective weight loading |
| `comfy/cli_args.py` | 36-270 | All CLI flag definitions |
| `comfy/ops.py` | 82-160 | VBAR cast operations (AIMDO only) |
