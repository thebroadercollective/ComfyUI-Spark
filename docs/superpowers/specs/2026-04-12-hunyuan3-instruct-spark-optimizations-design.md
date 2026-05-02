# HunyuanImage3 Instruct Loader — DGX Spark Unified Memory Optimizations

## Context

The HunyuanImage3 custom node (`custom_nodes/Comfy_HunyuanImage3/hunyuan_instruct_nodes.py`) uses HuggingFace `AutoModelForCausalLM.from_pretrained()` to load models — a completely separate path from ComfyUI's `load_torch_file()`. This means none of our Spark unified memory optimizations (page cache dropping, `soft_empty_cache_unified()`, memory observability) apply to it.

On DGX Spark (128GB unified LPDDR5x), the INT8-v2 model is ~83GB across 18 safetensors shards. During loading, `from_pretrained()` uses mmap internally (via safetensors' Rust binding), creating up to ~83GB of OS page cache that competes with CUDA allocations for the same physical memory pool. The existing code has a `time.sleep(15)` with a "DROP CACHES NOW!!!!" comment (line 1713), confirming the author encountered this pressure but lacked a programmatic solution.

**Goal:** Apply the same page cache management, cache clearing, and memory observability patterns from our core loading pipeline to the HunyuanImage3 custom loader.

## Changes

### 1. Background Thread Page Cache Dropper

**What:** A daemon thread that periodically drops OS page cache for all safetensors shards during `from_pretrained()`, plus a final sweep after it returns.

**Why:** `from_pretrained()` is opaque — we can't inject page cache drops between shard loads. A background thread with a 30-second interval keeps the page cache ceiling low during the entire ~83GB load, preventing kernel reclaim pressure spikes.

**How:**

A helper function (context manager or start/stop pair) in `hunyuan_instruct_nodes.py`:

```python
def _page_cache_dropper(safetensors_files, stop_event, interval=30):
    """Background thread: drop page cache for safetensors files every `interval` seconds."""
    while not stop_event.wait(interval):
        for path in safetensors_files:
            model_management.drop_file_page_cache(path)
        logger.info("PAGE_CACHE_DROP_TICK | dropped %d shards", len(safetensors_files))
```

Usage pattern wrapping each `from_pretrained()` call site:

```python
import glob, threading

safetensors_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))

if model_management.UNIFIED_MEMORY and safetensors_files:
    stop_event = threading.Event()
    dropper = threading.Thread(
        target=_page_cache_dropper,
        args=(safetensors_files, stop_event),
        daemon=True,
    )
    dropper.start()

model = AutoModelForCausalLM.from_pretrained(model_path, ...)

if model_management.UNIFIED_MEMORY and safetensors_files:
    stop_event.set()
    dropper.join(timeout=5)
    # Final sweep
    for path in safetensors_files:
        model_management.drop_file_page_cache(path)
    logger.info("PAGE_CACHE_DROP | final sweep, %d shards", len(safetensors_files))
```

**Guard:** Entire mechanism gated on `comfy.model_management.UNIFIED_MEMORY`. No-op on non-Spark systems.

**Call sites:** All five `from_pretrained()` calls in `load_model()`:
- NF4 (line 1628)
- INT8 block-swap (line 1652)
- INT8 single-GPU (lines 1716, 1732)
- INT8 explicit-map (line 1773)
- BF16 block-swap (line 1798)
- BF16 explicit-map (line 1823)

To avoid duplicating the start/stop boilerplate at each call site, wrap it in a context manager:

```python
@contextmanager
def _mid_load_page_cache_drop(model_path):
    """Drop page cache periodically during from_pretrained() on unified memory."""
    if not model_management.UNIFIED_MEMORY:
        yield
        return
    safetensors_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not safetensors_files:
        yield
        return
    stop_event = threading.Event()
    dropper = threading.Thread(
        target=_page_cache_dropper,
        args=(safetensors_files, stop_event),
        daemon=True,
    )
    dropper.start()
    try:
        yield
    finally:
        stop_event.set()
        dropper.join(timeout=5)
        for path in safetensors_files:
            model_management.drop_file_page_cache(path)
        logger.info("PAGE_CACHE_DROP | final sweep, %d shards", len(safetensors_files))
```

Each call site becomes:

```python
with _mid_load_page_cache_drop(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, ...)
```

### 2. Replace `torch.cuda.empty_cache()` with `soft_empty_cache_unified()`

**What:** In `load_model()`, replace raw `torch.cuda.empty_cache()` + `torch.cuda.synchronize()` pairs with `comfy.model_management.soft_empty_cache_unified()`.

**Why:** On unified memory, `soft_empty_cache_unified()` skips `ipc_collect()` (meaningless on single-process unified systems) and handles synchronize internally.

**Call sites in `load_model()`:**
- Force-reload cleanup: lines 1543-1544, 1552-1553
- Pre-load cleanup: lines 1618-1619

**Import:** `comfy.model_management` — lazy import inside `load_model()` to avoid load-order issues with the custom node system.

### 3. Remove `time.sleep(15)` Hack

**What:** Remove the `time.sleep(15)` at line 1713 and the "DROP CACHES NOW!!!!" log message.

**Why:** The background thread page cache dropper makes this unnecessary. The sleep was a manual workaround for page cache pressure.

**Replace with:** Nothing — the page cache dropper context manager already wraps the subsequent `from_pretrained()` call, and the pre-load `soft_empty_cache_unified()` handles CUDA cache.

### 4. Memory Observability

**What:** Add `memory_report()` / `memory_delta()` logging at key points in `load_model()`.

**Why:** Better diagnostics for understanding memory behavior during model loading, consistent with our core pipeline's observability.

**Insertion points:**
1. **Before loading** (after pre-load cleanup, ~line 1620): `before = memory_report()`
2. **After `from_pretrained()` + page cache drop**: `after = memory_report()` + log delta
3. **Replace GPU logging** (lines 1943-1949): Replace manual `torch.cuda.memory_allocated()` / `mem_get_info()` with `memory_report()` output

**Log format:** Consistent with existing tags: `HUNYUAN_LOAD | <message> | <memory_report>`

### 5. Out of Scope

- **`from_pretrained()` parameters** — device_map, torch_dtype, low_cpu_mem_usage are correct as-is
- **Block swap logic** — Orthogonal to page cache; already well-optimized
- **Generation/edit/fusion nodes** — Their `torch.cuda.empty_cache()` calls work fine; changing them adds risk for minimal gain
- **`cache_policy.maybe_drop()` integration** — The HunyuanImage3 loader doesn't go through ComfyUI's standard model pipeline, so phase hooks don't map cleanly. Explicit drops are clearer.

## Files Modified

| File | Changes |
|------|---------|
| `custom_nodes/Comfy_HunyuanImage3/hunyuan_instruct_nodes.py` | All changes — page cache dropper, soft_empty_cache_unified, remove sleep, memory observability |

## Functions Reused from ComfyUI-Spark

| Function | Module | Purpose |
|----------|--------|---------|
| `UNIFIED_MEMORY` | `comfy.model_management` | Gate all optimizations |
| `drop_file_page_cache()` | `comfy.model_management` | Per-file page cache eviction |
| `soft_empty_cache_unified()` | `comfy.model_management` | CUDA cache clearing (skips ipc_collect on unified) |
| `memory_report()` | `comfy.model_management` | Memory snapshot string |
| `memory_delta()` | `comfy.model_management` | Delta between two snapshots |

## Verification

1. **Load INT8-v2 model** with optimizations enabled on DGX Spark
2. **Check logs** for `PAGE_CACHE_DROP_TICK` entries during loading (expect ~2-3 at 30s intervals for a ~90s load)
3. **Check logs** for `PAGE_CACHE_DROP | final sweep` after loading completes
4. **Check logs** for `HUNYUAN_LOAD` entries with memory snapshots
5. **Compare** `sys avail` before/after page cache drop — should reclaim 10-40GB depending on kernel reclaim during load
6. **Verify** no `time.sleep(15)` in output
7. **Run inference** to confirm model works correctly after optimized loading
8. **Monitor** `free -h` during load to observe page cache (buff/cache column) staying lower than before
