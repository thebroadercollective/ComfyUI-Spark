# Follow-up: `free_memory()` unconditionally full-unloads non-dynamic models

Status: **open / low-priority** (harmless on the Spark's unified-memory path; documented
for a future cleanup and as a rebase tripwire).

Discovered 2026-05-31 while root-causing the RAM-pressure active-model eviction bug (fixed
in `comfy_execution/caching.py`; see CLAUDE.md "RAM-pressure cache" bullet and
`tests-unit/comfy_test/test_ram_pressure_model_protection.py`). This is a **separate**
latent divergence found in the same investigation, deliberately left out of that fix to
avoid scope creep.

## What

`comfy/model_management.py::free_memory()` (around lines 824-832):

```python
for x in can_unload_sorted:
    i = x[-1]
    memory_to_free = 1e32
    if current_loaded_models[i].model.is_dynamic() and (not DISABLE_SMART_MEMORY or device is None):
        memory_to_free = 0 if device is None else memory_required - get_free_memory(device)
        if for_dynamic:
            memory_required -= current_loaded_models[i].model.loaded_size()
            memory_to_free = 0
    if memory_to_free > 0 and current_loaded_models[i].model_unload(memory_to_free):
        unloaded_model.append(i)
```

Upstream's condition is `if not DISABLE_SMART_MEMORY or device is None:` — i.e. the
"only free the actual shortfall" smart-memory logic (`memory_to_free = memory_required -
get_free_memory(device)`, which goes **≤ 0 and skips the unload when there is already
enough free memory**) applies to *all* models. The fork added
`current_loaded_models[i].model.is_dynamic() and ...`, so for **non-dynamic** models the
branch is skipped, leaving `memory_to_free = 1e32` → always `> 0` → unconditional full
`model_unload`. With `--disable-dynamic-vram` (the documented Spark launch flag) every
model is non-dynamic, so **every** `free_memory()` call fully "unloads" each candidate
resident model on the target device, regardless of whether there is already plenty of
free memory.

`git diff upstream/master -- comfy/model_management.py` shows this is a fork change, not
upstream behavior.

## Why it is harmless on unified memory (today)

`model_unload(1e32)` → (1e32 is not `< loaded_size`, so the partial-unload branch is
skipped) → `self.model.detach(unpatch_weights=True)` → `unpatch_model(self.offload_device)`
→ `self.model.to(self.offload_device)`. On unified memory `offload_device == load_device
== cuda` (`unet_offload_device()` / `vae_offload_device()` return `get_torch_device()`),
so `self.model.to(cuda)` is a **no-op move**: the nn.Module weights stay physically
resident in the one shared pool, and the node-output cache's `ModelPatcher` keeps them
alive. The model is only removed from `current_loaded_models` (bookkeeping); the next
`load_models_gpu([...])` re-finds the cached patcher and reloads instantly (no disk read).

This is why, in the reported repro, every VAE-decode's `free_memory()` "unloaded" the
diffusion model with **zero** memory or latency cost. The catastrophic 60GB free came from
a different path entirely — `RAMPressureCache.ram_release(free_active=True)` doing
`del self.cache[key]`, which drops the patcher *object* — now fixed.

## Why it could still matter

- **Non-unified hardware / discrete GPU:** `offload_device` is CPU there, so this would
  force a real CPU↔GPU round-trip on every `free_memory()` even when no shortfall exists —
  a behavioral regression vs upstream. The fork targets the Spark, but the code is shared.
- **Churn / log noise:** it pops/re-inserts `current_loaded_models` and toggles loaded
  bookkeeping more than necessary; spurious "unloaded"/"loaded" sequences.
- **Rebase tripwire:** the `is_dynamic()` gate is a fork delta in an upstream hot path. A
  rebase that touches `free_memory()` may silently revert it (restoring upstream behavior,
  which is actually *more* correct here) or conflict.

## Recommended fix (when prioritized)

Extend the smart-memory branch to cover unified memory (and arguably restore upstream's
all-models behavior), so non-dynamic resident models are only unloaded for a genuine
shortfall:

```python
if (UNIFIED_MEMORY or current_loaded_models[i].model.is_dynamic()) and (not DISABLE_SMART_MEMORY or device is None):
    memory_to_free = 0 if device is None else memory_required - get_free_memory(device)
    ...
```

Caveats to verify before shipping:
- Confirm `model_unload(memory_to_free)` with a *finite* `memory_to_free < loaded_size`
  routes into `partially_unload(offload_device, ...)` and that on unified
  (`offload_device == load_device`) this stays a no-op for memory rather than re-introducing
  the per-step cast-on-demand "lowvram churn" the fork explicitly removed (see the
  `unified_full_load` rationale in `load_models_gpu`). If partial-unload-on-unified is not a
  clean no-op, prefer simply *never* unloading non-dynamic resident models on unified
  (they are the single resident copy; there is no second pool to free into).
- Add a unit test mirroring `test_ram_pressure_model_protection.py`: under no real
  shortfall, `free_memory()` must not unload an already-resident model.

## Pointers

- `comfy/model_management.py` `free_memory()` (~814-846), `load_models_gpu()` (~916/924),
  `LoadedModel.model_unload()` (~750), `unet_offload_device()`/`vae_offload_device()`.
- `comfy/model_patcher.py` `detach()` (~1384), `unpatch_model()` (~1215),
  `partially_unload()` (~1258).
