# DGX Spark ComfyUI Safetensors Loading Optimization

## Research & Implementation Plan

**Date:** April 5, 2026
**Target Hardware:** NVIDIA DGX Spark (GB10, sm_121, 128GB unified LPDDR5x memory)
**Target Software:** ComfyUI (current stable), Python 3.12, PyTorch cu130
**Initial Target Model:** FLUX.2-dev BF16 (~60GB on disk)
**Future Target:** EricRollei's INT8 HunyuanImage3 Instruct-Distil v2 (~80GB on disk)

---

## 1. Problem Statement

When ComfyUI loads safetensors files on the DGX Spark, models consume roughly **2x their expected memory**, frequently causing OOM kills during loading — even when the model would fit comfortably in memory for inference. The root cause is that ComfyUI's loading pipeline assumes separate RAM and VRAM pools, but on the Spark's unified memory architecture, "RAM" and "VRAM" are the same physical memory.

### Observed Symptoms

- `free -h` shows buff/cache growing by tens of GB during safetensors loading
- Dropping caches (`echo 3 > /proc/sys/vm/drop_caches`) frees equivalent memory from buff/cache, increases "free" column, but does NOT change "available" column
- OOM kills occur during model loading, never during inference
- Models that successfully load (via workarounds) run inference fine within memory limits
- The phaserblast ComfyUI-DGXSparkSafetensorsLoader custom node successfully loads models that fail via the standard loader, confirming the model itself fits in memory

---

## 2. Root Cause Analysis

There are **three layers of memory duplication** during safetensors loading on unified memory:

### Layer 1: mmap Page Cache Residue

`safetensors.safe_open()` uses memory-mapped I/O internally (at the Rust library level). On the Spark, the kernel's page cache fills with file-backed mmap pages. These pages persist even after tensor data has been materialized, because the kernel considers them reclaimable cache. However, the CUDA allocator does not trigger page cache reclamation when requesting GPU memory, leading to an OOM before the kernel has a chance to evict the cache pages.

**Evidence:** With `--disable-mmap` and `copy=False` patch applied, `free -h` still shows buff/cache growing by tens of GB during loading. Dropping caches frees this memory. The "available" column does not change (kernel considers it reclaimable), but CUDA allocations fail anyway.

**Key insight:** ComfyUI's `--disable-mmap` flag does NOT prevent mmap at the safetensors Rust library level. It only controls ComfyUI's post-deserialization copy behavior. `safe_open()` still uses mmap internally.

### Layer 2: tensor.to(copy=True) Duplication

In `comfy/utils.py`, when `--disable-mmap` is set, the code path does:
```python
tensor = tensor.to(device=device, copy=True)
```
On the Spark, "moving" a tensor from CPU to CUDA is a no-op in terms of physical location (same memory), but `copy=True` forces a duplicate allocation. The fix (already known) is changing to `copy=False`.

### Layer 3: Byte Buffer Overhead

When using `safetensors.torch.load(open(ckpt, 'rb').read())` (the non-mmap fallback), the entire file is read into a Python bytes object. This bytes object coexists with deserialized tensors until garbage collected, briefly requiring ~2x model size.

### Why fastsafetensors (phaserblast's loader) Works

The `fastsafetensors` library bypasses mmap entirely. It performs direct I/O reads into GPU memory via DLPack, creating zero-copy tensor views. No page cache accumulation, no duplicate allocations. This is why it successfully loads 60GB FLUX.2-dev when the standard loader fails.

**Limitation:** fastsafetensors relies on DLPack for tensor instantiation, and DLPack's dtype mapping doesn't include FP8 types (`float8_e4m3fn`, `float8_e5m2`). The fastsafetensors authors acknowledge this as a gap, not a fundamental limitation. Standard INT8 (`int8`) may already be supported by DLPack — this needs verification for the HunyuanImage3 case.

---

## 3. Existing Workarounds and Their Limitations

### 3a. `--disable-mmap` + `copy=False` patch in comfy/utils.py

- **What it does:** Prevents the explicit copy duplication (Layer 2)
- **What it doesn't do:** Does NOT prevent mmap at the safetensors library level (Layer 1)
- **Result:** Helps but insufficient for large models; page cache still grows

### 3b. ComfyUI-DGXSparkSafetensorsLoader (phaserblast)

- **What it does:** Uses fastsafetensors for zero-copy direct I/O loading, bypasses all three layers
- **Limitations:**
  - Requires replacing loader nodes in every workflow/template
  - No support for quantized models (FP8/INT8) due to DLPack dtype limitations
  - No memory management / unload capability (must restart ComfyUI to free VRAM)
  - Rough/experimental

### 3c. Manual cache dropping

- **What it does:** `echo 3 > /proc/sys/vm/drop_caches` reclaims mmap page cache residue
- **Limitations:** Requires root (or special permission setup), manual, can't help if OOM occurs during loading before you can drop

### 3d. luix93's optimized Docker container

- **What it does:** Bundles copy=False patch, disable-mmap, SageAttention, comfy-kitchen, optimized flags
- **Limitations:** Docker overhead, doesn't address the fundamental mmap page cache issue

---

## 4. Implementation Plan

### Goal

Create a monkey-patch module that overrides `comfy.utils.load_torch_file` to provide Spark-optimized safetensors loading. This approach:

- Works with ALL existing loader nodes (no workflow modifications needed)
- Works with custom nodes that call `comfy.utils.load_torch_file`
- Addresses all three layers of memory duplication
- Can be activated via CLI flag or auto-detected on GB10 hardware
- Supports quantized models (INT8 initially, FP8 as stretch goal)

### Phase 1: System Setup — Unprivileged Cache Dropping

Since dropping page cache requires root, and ComfyUI should NOT run as root, set up a **setuid helper binary**.

#### Implementation

Create `/usr/local/bin/drop_caches`:

```c
// drop_caches.c
#include <fcntl.h>
#include <unistd.h>
int main(void) {
    sync();
    int fd = open("/proc/sys/vm/drop_caches", O_WRONLY);
    if (fd >= 0) { write(fd, "3\n", 2); close(fd); }
    return 0;
}
```

Build and install:

```bash
sudo gcc -o /usr/local/bin/drop_caches drop_caches.c
sudo chown root:root /usr/local/bin/drop_caches
sudo chmod 4755 /usr/local/bin/drop_caches
```

#### Verification

From the non-root ComfyUI user:
```bash
# Check buff/cache before
free -h
# Run the helper
/usr/local/bin/drop_caches
# Check buff/cache after — should decrease
free -h
```

### Phase 2: Monkey-Patch Module for comfy.utils.load_torch_file

#### Architecture

The module should:

1. **Detect unified memory at startup** — check for GB10 (sm_121) via `torch.cuda.get_device_properties()`
2. **Replace `comfy.utils.load_torch_file`** with a Spark-optimized version
3. **Avoid mmap entirely** — use `safetensors.torch.load()` with raw file reads instead of `safe_open()`
4. **Eliminate copy duplication** — `tensor.to(device, copy=False)` on unified memory
5. **Proactively drop page cache** — call the setuid helper after each safetensors file loads
6. **Aggressively free intermediate buffers** — `del` byte buffers, `gc.collect()` after deserialization
7. **Preserve all existing behavior** for non-Spark systems (the monkey-patch should be a no-op on discrete GPU systems)

#### Key Design Decisions

**Loading strategy:** Use `safetensors.torch.load(raw_bytes)` which deserializes from an in-memory byte buffer rather than mmap. This completely avoids Layer 1. The byte buffer is a Python object that can be explicitly `del`'d and garbage collected.

**Memory budget concern:** For a 60GB safetensors file, `safetensors.torch.load(raw_bytes)` briefly requires the raw bytes (~60GB) plus the deserialized tensor dict (~60GB) = ~120GB. On a 128GB Spark (~119GB usable), this is tight. Consider:

- **Chunked loading fallback:** If total file size > threshold (e.g., 50% of available memory), fall back to `safe_open()` with per-tensor loading + aggressive cache dropping after each tensor. This avoids the 2x spike from reading the entire file at once.
- **Streaming approach:** Use `safetensors.safe_open()` but load tensors one at a time, dropping page cache periodically (e.g., every N tensors or every N GB loaded). This is a middle ground between the full-file-read approach and the zero-copy fastsafetensors approach.

**Recommended approach for the initial implementation:**

For files that fit comfortably (file size < 40% of total memory ~= 50GB):
- Full file read with `safetensors.torch.load()`

For larger files (file size >= 40% of total memory):
- Use `safetensors.safe_open()` with per-tensor loading
- Drop page cache every N tensors (tunable, start with every 100 tensors or every 5GB)
- Use `copy=False` for `.to()` calls

#### Module Structure

```
ComfyUI/
├── custom_nodes/
│   └── comfyui-spark-loader/      # or placed at ComfyUI root level
│       ├── __init__.py             # Auto-patches on import
│       ├── spark_loader.py         # Core loading logic
│       └── requirements.txt        # (empty or minimal)
```

#### Core Loading Logic — spark_loader.py

```python
"""
Spark-optimized safetensors loader for DGX Spark unified memory.

Replaces comfy.utils.load_torch_file with a version that:
- Avoids mmap-based loading (prevents page cache bloat on unified memory)
- Uses copy=False for tensor device moves (no-op on unified memory)
- Proactively drops page cache during loading of large models
- Falls back to standard loading on non-Spark hardware
"""

import os
import gc
import subprocess
import logging
import torch
import safetensors
import safetensors.torch

logger = logging.getLogger("spark_loader")

# --- Configuration ---
CACHE_DROP_BINARY = "/usr/local/bin/drop_caches"
# Files smaller than this threshold use full-file-read (fast, but 2x peak memory)
FULL_READ_THRESHOLD_RATIO = 0.40  # fraction of total system memory
# For large-file streaming mode, drop cache every N bytes loaded
CACHE_DROP_INTERVAL_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB


def is_unified_memory_system():
    """Detect DGX Spark / GB10 unified memory architecture."""
    try:
        if not torch.cuda.is_available():
            return False
        props = torch.cuda.get_device_properties(0)
        # GB10 is compute capability 12.1 (sm_121)
        if props.major == 12 and props.minor == 1:
            return True
    except Exception:
        pass
    return False


def get_total_system_memory_bytes():
    """Get total system memory from /proc/meminfo."""
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    # Value is in kB
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 128 * 1024 * 1024 * 1024  # fallback: assume 128GB


def drop_page_cache():
    """Drop kernel page cache using setuid helper binary."""
    try:
        subprocess.run(
            [CACHE_DROP_BINARY],
            check=False,
            capture_output=True,
            timeout=10
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        logger.debug(f"Cache drop skipped: {e}")


def spark_load_safetensors_small(path, device):
    """
    Load safetensors via full file read (no mmap).
    Best for files that fit comfortably in memory alongside the deserialized tensors.
    """
    logger.info(f"Spark loader: full-read mode for {os.path.basename(path)}")

    with open(path, 'rb') as f:
        raw_bytes = f.read()

    sd = safetensors.torch.load(raw_bytes)

    # Immediately free the byte buffer
    del raw_bytes
    gc.collect()

    # Move tensors to target device without copying (no-op on unified memory)
    if device is not None and str(device) != "cpu":
        sd = {k: v.to(device, copy=False) for k, v in sd.items()}

    # Drop any residual page cache
    drop_page_cache()

    return sd, None  # (state_dict, metadata) — metadata not available in this path


def spark_load_safetensors_large(path, device, return_metadata=False):
    """
    Load safetensors via safe_open with periodic cache dropping.
    For files too large to hold both the raw bytes and deserialized tensors.
    Uses safe_open (which mmap's internally) but aggressively drops cache
    to prevent page cache accumulation from triggering OOM.
    """
    file_size = os.path.getsize(path)
    logger.info(
        f"Spark loader: streaming mode for {os.path.basename(path)} "
        f"({file_size / (1024**3):.1f} GB)"
    )

    metadata = None
    sd = {}
    bytes_loaded = 0
    last_cache_drop = 0

    # Initial cache drop to start clean
    drop_page_cache()

    device_str = device.type if hasattr(device, 'type') else str(device)

    with safetensors.safe_open(path, framework="pt", device=device_str) as f:
        if return_metadata:
            metadata = f.metadata()

        keys = list(f.keys())
        for i, key in enumerate(keys):
            tensor = f.get_tensor(key)

            # On unified memory, avoid copy=True which is the default
            # when the tensor's device doesn't match the target
            if device is not None and str(device) != "cpu":
                tensor = tensor.to(device, copy=False)

            sd[key] = tensor

            # Estimate bytes loaded (element_size * numel)
            bytes_loaded += tensor.element_size() * tensor.numel()

            # Periodically drop cache to prevent accumulation
            if bytes_loaded - last_cache_drop >= CACHE_DROP_INTERVAL_BYTES:
                drop_page_cache()
                last_cache_drop = bytes_loaded
                logger.debug(
                    f"  Cache drop at {bytes_loaded / (1024**3):.1f} GB "
                    f"({i+1}/{len(keys)} tensors)"
                )

    # Final cache drop
    drop_page_cache()
    gc.collect()

    return sd, metadata


def spark_load_torch_file(ckpt, safe_load=False, device=None, return_metadata=False):
    """
    Drop-in replacement for comfy.utils.load_torch_file optimized for
    DGX Spark unified memory. Falls through to the original implementation
    for non-safetensors files.
    """
    if device is None:
        device = torch.device("cpu")

    # Only handle safetensors files — let everything else use the original loader
    if not (ckpt.lower().endswith(".safetensors") or ckpt.lower().endswith(".sft")):
        return _original_load_torch_file(ckpt, safe_load=safe_load,
                                          device=device,
                                          return_metadata=return_metadata)

    file_size = os.path.getsize(ckpt)
    total_mem = get_total_system_memory_bytes()
    threshold = total_mem * FULL_READ_THRESHOLD_RATIO

    try:
        if file_size < threshold:
            sd, metadata = spark_load_safetensors_small(ckpt, device)
        else:
            sd, metadata = spark_load_safetensors_large(
                ckpt, device, return_metadata=return_metadata
            )
    except Exception as e:
        logger.warning(
            f"Spark loader failed, falling back to standard loader: {e}"
        )
        return _original_load_torch_file(ckpt, safe_load=safe_load,
                                          device=device,
                                          return_metadata=return_metadata)

    if return_metadata:
        return sd, metadata
    return sd


# Placeholder — set during patching
_original_load_torch_file = None
```

#### Initialization / Monkey-Patching — __init__.py

```python
"""
ComfyUI Spark Loader — Automatic monkey-patch for DGX Spark unified memory.

Place this directory in ComfyUI/custom_nodes/ and it will auto-activate
on GB10 hardware. On non-Spark systems, it does nothing.
"""

import logging
from .spark_loader import is_unified_memory_system, spark_load_torch_file

logger = logging.getLogger("spark_loader")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def _apply_patch():
    import comfy.utils
    from . import spark_loader

    # Save reference to original function
    spark_loader._original_load_torch_file = comfy.utils.load_torch_file

    # Replace with Spark-optimized version
    comfy.utils.load_torch_file = spark_load_torch_file

    logger.info("Spark loader: patched comfy.utils.load_torch_file for unified memory")

if is_unified_memory_system():
    logger.info("Spark loader: GB10 unified memory detected, activating optimizations")
    _apply_patch()
else:
    logger.info("Spark loader: non-Spark system detected, no patches applied")
```

### Phase 3: Testing Protocol

#### Test 1: Baseline (no patches)

```bash
# Start ComfyUI without any Spark optimizations
python main.py --listen 0.0.0.0

# Load FLUX.2-dev BF16 via standard Load Diffusion Model node
# Monitor in another terminal:
watch -n 2 'free -h; echo "---"; cat /proc/meminfo | grep -E "Cached|Buffers"'

# Expected: OOM or massive cache bloat
```

#### Test 2: Monkey-patch module active

```bash
# Ensure comfyui-spark-loader is in custom_nodes/
# Ensure /usr/local/bin/drop_caches setuid binary is installed
# Start ComfyUI with bf16 flags:
python main.py --listen 0.0.0.0 \
  --bf16-unet --bf16-vae --bf16-text-enc \
  --dont-upcast-attention \
  --disable-pinned-memory

# Load FLUX.2-dev BF16 via standard Load Diffusion Model node
# Monitor memory: cache should stay controlled, model should load
```

#### Test 3: Verify inference works

After successful loading, run a standard FLUX.2-dev txt2img generation to confirm the loaded model produces correct output.

#### Test 4: Compare with ComfyUI-DGXSparkSafetensorsLoader

Load the same model via phaserblast's node and compare:
- Peak memory during loading
- Total memory after loading
- Loading time
- Inference output (should be identical)

### Phase 4: EricRollei's HunyuanImage3 Custom Node (Future)

This is deferred but the approach is:

1. EricRollei's node uses `AutoModelForCausalLM.from_pretrained()` which has its own loading pipeline separate from `comfy.utils.load_torch_file`
2. The monkey-patch module will NOT intercept this path
3. To optimize this, modify his node's `from_pretrained()` call:
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       model_path,
       device_map="cuda:0",
       torch_dtype=torch.bfloat16,
       low_cpu_mem_usage=True,  # meta tensors, shard-by-shard loading
   )
   ```
4. Add post-load cache dropping
5. Investigate `HF_SAFETENSORS_NO_MMAP=1` environment variable
6. Test whether the monkey-patch's `spark_load_safetensors_large` could be adapted to work with HF's shard loading

---

## 5. Recommended ComfyUI Launch Configuration

For reference, the full optimized launch configuration for DGX Spark:

### System-Level (persistent across reboots)

```bash
# /etc/sysctl.d/99-dgx-spark.conf
vm.swappiness=10
```

### Pre-Launch

```bash
# Disable swap (prevents silent lockups on unified memory)
sudo swapoff -a

# Cap GPU clocks (prevents power spike hard crashes)
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc 300,2100

# Set CUDA kernel cache (3x speedup on subsequent runs)
export CUDA_CACHE_MAXSIZE=4294967296

# Skip NCCL overhead (single GPU)
export NCCL_P2P_DISABLE=1
```

### ComfyUI Launch

```bash
python main.py \
  --listen 0.0.0.0 \
  --bf16-unet \
  --bf16-vae \
  --bf16-text-enc \
  --dont-upcast-attention \
  --disable-pinned-memory \
  --disable-dynamic-vram \
  --reserve-vram 1 \
  --disable-mmap \
  --use-sage-attention  # only if SageAttention compiled for sm_121
```

### Flags NOT to use

- `--highvram` — forces all models GPU-pinned, causes OOM on unified memory
- `--gpu-only` — fights the unified memory fabric
- `--cache-none` — unnecessary with the monkey-patch handling memory properly
- `PYTORCH_NO_CUDA_MEMORY_CACHING=1` — causes fragmentation and OOM
- `CUDA_CACHE_DISABLE=1` — kills kernel cache, 3x slower reruns

---

## 6. Reference Links

- [ComfyUI Issue #10896 — Double memory on DGX Spark](https://github.com/comfyanonymous/ComfyUI/issues/10896)
- [phaserblast/ComfyUI-DGXSparkSafetensorsLoader](https://github.com/phaserblast/ComfyUI-DGXSparkSafetensorsLoader)
- [fastsafetensors library](https://github.com/foundation-model-stack/fastsafetensors)
- [fastsafetensors paper — DLPack dtype limitation](https://arxiv.org/html/2505.23072v1)
- [NVIDIA Forum — Unlocking the Spark in ComfyUI](https://forums.developer.nvidia.com/t/unlocking-the-power-of-the-spark-in-comfyui-no-crashes/360336)
- [NVIDIA Forum — ComfyUI optimized setup for DGX Spark (luix93)](https://forums.developer.nvidia.com/t/comfyui-setup-optimized-for-dgx-spark/364846)
- [luix93/DGX-Spark-ComfyUI Docker](https://github.com/luix93/DGX-Spark-ComfyUI)
- [SparkyUI — ComfyUI Docker with flag rationale](https://github.com/ecarmen16/SparkyUI/)
- [NVIDIA build.nvidia.com — ComfyUI troubleshooting](https://build.nvidia.com/spark/comfy-ui/troubleshooting)
- [ComfyUI Dynamic VRAM discussion](https://github.com/Comfy-Org/ComfyUI/discussions/12699)
- [ComfyUI Blog — Dynamic VRAM explanation](https://blog.comfy.org/p/dynamic-vram-in-comfyui-saving-local)
- [ComfyUI Blog — NVFP4 optimizations](https://blog.comfy.org/p/new-comfyui-optimizations-for-nvidia)
- [ComfyUI Server Config reference](https://docs.comfy.org/interface/settings/server-config)
