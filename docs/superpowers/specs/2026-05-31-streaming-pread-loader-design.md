---
Status: Reviewed — approved to implement with changes
---

# Streaming pread Loader — Eliminating Mid-Load Page-Cache Duplication on Unified Memory

**Date:** 2026-05-31
**Status:** Reviewed (adversarial review 2026-05-31) — approved to implement with the changes below
**Author:** Spark port team
**Component:** `comfy/utils.py` (`load_torch_file`)
**Related:** `docs/superpowers/specs/2026-04-05-unified-memory-loading-design.md`

## Review Outcome (2026-05-31, two rounds)

An adversarial design review (Opus 4.8 devil's advocate) ran experiments on the actual
GB10 hardware across **two rounds** and returned **"proceed."** The diagnosis was confirmed
by direct measurement; the I/O mechanism was measured to be **faster** than `safe_open`, not
slower (4.2s vs 6.5s on a cold 4GiB read, and it bounds + evicts page cache mid-load while
`safe_open`'s mmap pins the whole file).

**Round 1** required: dtype generality, correctness invariants, stronger verification — all
folded in. **Round 2** produced one decisive simplification, now adopted: **collapse the
reconstruction to a single path** ("always tier-2"). The original draft had a fast tier-1
`uint8.view(dtype).view(shape)` path with a tier-2 fallback. Measurement showed tier-1 saves
only **~2.8 ms per 512 MB tensor (~330 ms across a 60 GB model)** — noise against a problem
measured in hundreds of seconds — while carrying ~80% of the design's correctness surface
(offset-0 alignment invariant, size-match assertion, packed-dtype `_x2` hazard, and a
tier-1→tier-2 partial-failure CUDA leak). So we drop tier-1 entirely: every tensor is
reconstructed by handing its bytes to **safetensors' own** `safetensors.torch.load()` via a
one-tensor in-memory blob. This is *more* general and *less* code. We own only header parsing
+ I/O; the library owns all dtype/shape/packing reconstruction.

**Two alternatives were considered and rejected:**

1. **`vm.swappiness` / `swapoff` sysctl (zero code).** Rejected by the project owner on
   principle: *the system should not be reaching swap in the first place, and if it
   genuinely does, swap must remain available to prevent OOM.* This is actually an argument
   **for** this loader: the right fix prevents the memory pressure that drives the kernel to
   swap, rather than papering over the symptom or removing the OOM safety net. Swap stays
   enabled at its current 16GB; this work removes the transient that fills it.
2. **`madvise(MADV_DONTNEED)` on safetensors' own mapping.** Rejected: the Rust binding does
   not expose the mmap address/fd (already noted in CLAUDE.md); reaching it would require
   parsing `/proc/self/maps` and `ctypes`-calling `madvise` on a foreign mapping — far more
   fragile than owning the I/O.
3. **`O_DIRECT`.** Rejected as primary (kept as a possible later optimization): safetensors
   `data_offsets` are not guaranteed 4K-aligned, so each tensor would need an aligned
   bounce-buffer read-and-trim. `pread`+`fadvise` already reaches a bounded, fully-evicted
   transient at full disk bandwidth, so the alignment complexity isn't worth it.

**Generality requirement (project owner):** the design must be general across model
architectures and dtype mixes, **not** specialized to Flux.2 Dev. This drove the most
significant change vs the draft — see [Dtype Generality](#dtype-generality).

## Post-Implementation Refinements (2026-05-31, after first live run)

Shipped and verified on a full Flux.2 Dev run (no swap, free never < ~20GB, ~90s faster). Three refinements followed from the live log:

1. **CPU target too.** The streamer was generalized from CUDA-only to any unified large load, including the explicit-CPU `--cpu-text-enc` text-encoder path (the ~33GB mistral encoder, which as `cpu-explicit`/`safe_open` leaked ~33GB of un-evictable page cache). Function renamed `_stream_safetensors_to_cuda` → `_stream_safetensors_to_device(..., target_device)`; gate now admits `stream_target ∈ {cuda, cpu}`; `method=` is `unified-cuda-stream` / `unified-cpu-stream`. Byte-identical to `safe_open(device='cpu')` (tested).
2. **Log FREE, not just available.** Load logs now show `free X avail Y` (ticks) and `sys free X avail Y used Z` (bookends, via `memory_report`); `memory_delta` adds `Δfree`. `free` excludes buff/cache and is the real pressure signal; `available` masked it (it was pinned at ~116G while cache silently filled).
3. **Unconditional unified full-load.** The first-generation `LOAD_BUDGET` showed `fits_fully=False` because the residency probe (`_model_weights_on_device`) raced the accounting. Replaced with `unified_full_load = bool(UNIFIED_MEMORY)` (helper removed): the model is always already resident here (loaded straight to the pool), a too-big model would have OOM'd during load, and this branch never runs for CPU targets — so full-load is unconditionally correct on unified. First-gen now logs `fits_fully=True` / `loaded completely`.

Note (not changed): the post-text-encoder OS page cache observed lingering in the first run is addressed by #1 — mistral now streams with bounded, evicted cache instead of the safe_open leak.

## Executive Summary

On the DGX Spark, loading a large safetensors model (e.g. `flux2-dev.safetensors`,
60 GB) into the unified pool transiently consumes **~2× the file size** of physical
memory — once as the CUDA-resident tensors and once as un-evictable OS page cache held
by `safetensors.safe_open`'s live mmap. For a 60 GB model this drives the 128 GB pool
to <1 GB free, triggers heavy swap usage (observed: swap fills to its 15 GB limit), and
stretches the load to **~320 s** via kernel direct-reclaim thrash. The steady state is
fine (after load, page cache is dropped and ~48 GB is free); the problem is purely
*transient*, *during* the load.

The existing mitigation — `PAGE_CACHE_DROP_TICK`, which calls `posix_fadvise(DONTNEED)`
on a separate fd at each progress tick — is **empirically a no-op while the load is in
progress**, because `fadvise(DONTNEED)` cannot evict pages that are currently mapped into
a live mmap (see [Evidence](#evidence)). The post-load `PAGE_CACHE_DROP` works only because
the `with safe_open(...)` block has by then munmap'd the file.

**Proposal:** for the unified-memory large-file path, replace `safetensors.safe_open`
(which mmaps the whole file) with a manual streaming reader that uses `preadv()` to read each
tensor's bytes into a fresh per-tensor host buffer, reconstructs the tensor by handing those
bytes to safetensors' own `safetensors.torch.load()` (a one-tensor in-memory blob), copies it
to CUDA, and periodically calls `fadvise(DONTNEED)`. Because `read()`/`pread()` pages are
*not* mapped into a page table, `fadvise(DONTNEED)` evicts them effectively — keeping the
transient page cache bounded to `DROP_INTERVAL + largest_tensor` (≈ <2 GB) throughout the
load. This eliminates the 2× transient, the swap thrash, and the 320 s stall, leaving the
CUDA copy as the only resident copy of the bytes. We own only header parsing + I/O; all
dtype/shape/packing reconstruction is **delegated to safetensors**, so coverage stays general
and exactly equal to `safe_open`'s — see [Dtype Generality](#dtype-generality).

## Background: The Problem

### Observed behavior (2026-05-31, back-to-back Flux.2 Dev generations)

From `free -h` sampled at 1 Hz during the `flux2-dev.safetensors` load:

- `buff/cache` climbs steadily as the file is read.
- `PAGE_CACHE_DROP_TICK` lines appear at every progress tick but `buff/cache` does **not**
  fall in response.
- At ~60 % loaded, free memory drops below 1 GB and **swap usage begins climbing rapidly**
  even though `buff/cache` is still 20–30 GB (i.e. the kernel is swapping anonymous pages
  rather than dropping the clean, but mmap-pinned, file cache).
- Swap fills to its 15 GB limit; only then does `buff/cache` start to fall (forced LRU
  reclaim of the mapped file pages under extreme pressure), letting free hover ~1 GB.
- The model finishes loading; **after** load `buff/cache` collapses to near-zero and ~48 GB
  free is restored.

### Confirming log excerpt

```
LOAD file=flux2-dev.safetensors size=60.0G method=unified-cuda | ... sys avail 115.2G
  LOAD ... 4.6G/60.0G  (8%)  | sys avail 106.1G   # -9.1G for +4.6G copied
  LOAD ... 9.1G/60.0G  (15%) | sys avail  97.0G   # -9.1G for +4.5G copied
  ...
  LOAD ... 59.8G/60.0G (100%)| sys avail  10.1G
LOAD done ... elapsed=320.6s ... Δavail -58.8G
```

Each ~4.5 GB of tensor data copied to CUDA costs ~9 GB of system availability — the extra
~4.5 GB is the second (page-cache) copy of the same bytes, held by the live mmap.

### Root cause

`load_torch_file` (`comfy/utils.py:198`) does:

```python
with safetensors.safe_open(ckpt, framework="pt", device="cuda") as f:
    for k in f.keys():
        sd[k] = f.get_tensor(k)   # tensor materialized to CUDA
        # ... PAGE_CACHE_DROP_TICK: fadvise(DONTNEED) on a separate fd ...
```

`safe_open` mmaps the **entire** file for the lifetime of the `with` block. `get_tensor`
reads through that mapping (faulting pages into page cache) and copies to CUDA. The
mid-load `fadvise(DONTNEED)` (`comfy/utils.py:240-244`) targets a separate read-only fd,
but on Linux `fadvise(DONTNEED)` → `invalidate_mapping_pages()` **skips pages that are
mapped into any page table**. Since the mmap is live for the whole load, the ticks evict
nothing. The pages become evictable only at `__exit__` (munmap), which is when the
post-load `PAGE_CACHE_DROP` finally succeeds.

This is the same 2× duplication the fork eliminated for the *steady state* via direct
`safe_open(device="cuda")` loading — but it re-appears as a *transient* because the mmap
source itself is cached and pinned during the copy.

## Evidence

A standalone experiment (run on this host, 2026-05-31) measured global page cache
(`/proc/meminfo` `Cached + Buffers`) around a 4 GiB file:

| Step | Page cache |
|---|---|
| baseline after `fadvise(DONTNEED)` | 339 MB |
| after faulting in 4 GiB via a live `mmap` | 4436 MB (+4096) |
| **`fadvise(DONTNEED)` on a separate fd while mmap live** | **4436 MB (evicted 0)** |
| after `munmap` + `fadvise(DONTNEED)` | 340 MB (evicted all 4096) |

This is a direct reproduction of the `PAGE_CACHE_DROP_TICK` scenario and conclusively
shows the mid-load drop evicts nothing while the mapping is live, but evicts everything
once the mapping is gone. `pread()`/`read()` pages (not mapped) do not have this
restriction.

## Goals / Non-Goals

**Goals**
- Keep transient page cache bounded (target: < `DROP_INTERVAL + largest_tensor`, ≈ <2 GB)
  throughout a large unified-cuda load.
- Eliminate the swap thrash and the ~320 s stall for models in the ~30–120 GB range.
- Leave the CUDA-resident copy as the single copy of the bytes (no behavioral change to the
  resulting state dict / dtypes).
- Be a drop-in for the existing `unified and not explicit_device` path only.

**Non-Goals**
- Changing the `aimdo`, `cpu-explicit`, `cuda-explicit`, or `mmap` / torch-pickle paths.
- Changing dtype handling, `assign=True` semantics, or `should_assign_weights()`.
- Specializing to any model architecture or dtype mix — the path must be **general** and its
  dtype coverage must equal `safe_open`'s (see [Dtype Generality](#dtype-generality)).
- Solving the case where a model is genuinely larger than the physical pool (that is an
  OOM regardless; `--drop-page-cache` is the only lever and beyond pool size you are out).
- Multi-GPU / non-CUDA devices.

## Design

### Scope / gate

Engage the streaming reader only when **all** of:
- `unified and not explicit_device` (the current `method == "unified-cuda"` branch), AND
- `not comfy.memory_management.aimdo_enabled` (aimdo has its own loader), AND
- `total_size >= STREAM_THRESHOLD` (default **5 GB**, the same threshold already used to gate
  ticks / mid-load drops), AND
- `hasattr(os, "preadv")` and `hasattr(os, "posix_fadvise")` (Linux), AND
- safetensors' private dtype resolver is importable (see [Dtype Generality](#dtype-generality)).

Otherwise fall back to the existing `safe_open` path unchanged. Any error inside the
streaming reader falls back to `safe_open` (defensive: correctness over optimization), after
cleaning up partial state (see [Partial-failure cleanup](#partial-failure-cleanup)).

### Dtype Generality

**This is the most important change from the draft and the answer to "must be general, not
Flux.2-specific."** The draft proposed reusing the fork's local `_TYPES` map
(`comfy/utils.py:66`). That map has only **12** entries — verified missing `F8_E8M0` and
`float4` (`F4`), among others — whereas the `safe_open` path it replaces resolves dtypes
through **safetensors' own** map. Keying the streamer on the fork's subset would silently
route any checkpoint using a dtype we forgot (MXFP4 / NVFP4 / E8M0 quantized models —
exactly the kind that lands on a Spark) down the slow fallback path, **reintroducing the
320 s behavior with no error.** That is the opposite of general.

Verified facts (installed `torch 2.12.0+cu130`, `safetensors 0.7.0`):
- `torch.float8_e8m0fnu` and `torch.float4_e2m1fn_x2` **exist** in this torch.
- The fork's `_TYPES` resolves neither.
- `safetensors.torch._getdtype("F8_E8M0")` **also** raises `KeyError` in 0.7.0, and an
  in-memory `safetensors.torch.load()` of an `F8_E8M0` blob raises `KeyError` too — i.e.
  `safe_open` itself cannot load an E8M0 checkpoint in this version.
- Therefore the **correct generality bar is "exactly match `safe_open`'s coverage,"** no more
  and no less: any dtype `safe_open` cannot load, our fallback to `safe_open` reproduces the
  *identical* error (zero regression); any dtype `safe_open` *can* load, we must also load.

**Design rule:** validate dtypes against **safetensors' own** dtype map, not a hand-rolled
one. Import `safetensors.torch._TYPES` (the dict) at module load behind a `try/except`; if
that private symbol is ever absent (version bump), disable the stream path entirely
(`safe_open` fallback) rather than guess. We use it for **membership testing only** in the
pre-flight (`info["dtype"] in _TYPES`) — the actual string→dtype *conversion* is done by
`safetensors.torch.load()` during reconstruction, so we never call a private function, only
read a private dict. (Depending on a dict's contents is strictly less surface than depending
on a function's name + call + exception semantics; `_getdtype` is just a lookup into this
same map.) This makes the streamer's coverage **identical to and auto-tracking** `safe_open`
across safetensors upgrades. Do **not** reuse the fork's `_TYPES` and do **not** introduce a
second hand-maintained dtype map.

Verified on this host: `safetensors.torch._TYPES` is a 16-entry `dict` containing `BF16`,
`F32`, `F8_E4M3`, … and **not** `F8_E8M0` — exactly `safe_open`'s coverage.

### safetensors format (what we parse)

```
[8 bytes: little-endian u64 header_len]
[header_len bytes: UTF-8 JSON header]
[tensor data region]
```
The JSON header maps `name -> {dtype, shape, data_offsets:[begin,end]}`; offsets are relative
to the start of the data region (absolute file offset = `8 + header_len + begin`). A
`"__metadata__"` key may be present (returned only if `return_metadata`). We parse only the
8-byte length + JSON header ourselves; **all dtype/shape/packing reconstruction is
delegated** (see below) so we never hand-roll format math.

### Reconstruction strategy (single path, fully delegated)

For each tensor we own only the I/O; **all** byte→typed-tensor reconstruction is delegated to
the safetensors library — there is no hand-rolled `.view()` path:

1. **`preadv`** the tensor's raw bytes into a fresh `bytearray(nbytes)` (offset-0, no shared
   buffer).
2. **Build a one-tensor in-memory safetensors blob** from the original header entry:
   `struct.pack("<Q", len(hdr_json)) + hdr_json + raw`, where `hdr_json` is the JSON for just
   this tensor with `data_offsets = [0, nbytes]`.
3. **`cpu_tensor = safetensors.torch.load(bytes(blob))[name]`** — the library does all
   dtype/shape/packing reconstruction (including packed `float4_e2m1fn_x2` and any future
   layout), byte-identically to `safe_open`.
4. **`sd[name] = cpu_tensor.to("cuda")`**, then free `raw`/`blob`/`cpu_tensor`.

**Why single-path (round-2 decision):** an earlier draft had a fast `uint8.view(dtype)
.view(shape)` tier with this as a fallback. Measured cost of using `load()` for *everything*
is **+2.8 ms / 512 MB (~330 ms for a 60 GB model)** — negligible against a hundreds-of-seconds
problem — and it deletes the offset-0 alignment invariant, the size-match assertion, the
packed-dtype hazard, and a tier-fallthrough CUDA-leak. The only cost is a transient **2×
largest_tensor** in host RAM for one tensor at a time (the blob's `raw` + the library's output
tensor), freed each iteration — trivial on a 128 GB pool. **On any error → whole-file
`safe_open` fallback** (see [Partial-failure cleanup](#partial-failure-cleanup)).

Confirmed on this host: `safetensors.torch.load(bytes)` round-trips a hand-built blob
byte-identically for BF16 and F32, and correctly handles zero-size (`[0]`) and zero-dim
(`[3,0,4]`) tensors.

### Algorithm

```
from safetensors.torch import _TYPES as _ST_TYPES   # guarded at module import; missing -> stream path disabled
import safetensors.torch

# pre-flight (before reading ANY tensor bytes):
fd = os.open(ckpt, O_RDONLY)
hdr_len = struct.unpack("<Q", os.pread(fd, 8, 0))[0]
header  = json.loads(os.pread(fd, hdr_len, 8))
data_base = 8 + hdr_len
tensors = [(n, i) for n, i in header.items() if n != "__metadata__"]
# B2: validate EVERY dtype up front (membership only); unknown -> WARNING + whole-file fallback
for n, info in tensors:
    if info["dtype"] not in _ST_TYPES:
        logging.warning("stream loader: unknown dtype %s for %s; falling back to safe_open",
                        info["dtype"], n)
        os.close(fd); return _safe_open_load(...)   # identical coverage, zero regression

read_order = sorted(tensors, key=lambda kv: kv[1]["data_offsets"][0])  # sequential I/O
sd = {}
bytes_since_drop = 0
try:
    for name, info in read_order:
        begin, end = info["data_offsets"]; nbytes = end - begin
        raw = bytearray(nbytes)                                # fresh, offset 0
        if nbytes:
            _preadv_exact(fd, raw, data_base + begin)          # loop until nbytes (short reads)
        # one-tensor blob: header for THIS tensor only, offsets rebased to [0, nbytes]
        one = {name: {"dtype": info["dtype"], "shape": info["shape"], "data_offsets": [0, nbytes]}}
        hb = json.dumps(one).encode("utf-8")
        blob = struct.pack("<Q", len(hb)) + hb + bytes(raw)
        sd[name] = safetensors.torch.load(blob)[name].to("cuda")   # library owns reconstruction
        del raw, blob
        bytes_since_drop += nbytes
        if bytes_since_drop >= DROP_INTERVAL:                 # 512 MB (see Tuning)
            os.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)   # effective: no live mmap
            bytes_since_drop = 0
            emit throttled LOAD progress + PAGE_CACHE_DROP_TICK (before/after sys avail)
    os.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
finally:
    os.close(fd)
# re-key the returned dict in sorted order (matches safe_open's keys); read_order was I/O only.
```

### Partial-failure cleanup

If the reader raises mid-loop, drop the partially-built `sd` (releasing already-allocated
CUDA tensors) and force the allocator to release before retrying via `safe_open` — otherwise
the fallback runs with N tensors' worth of dead CUDA reservation:
```
except Exception as err:
    del sd
    comfy.model_management.soft_empty_cache_unified(force=True)
    logging.warning("stream loader failed (%s); falling back to safe_open", err)
    return _safe_open_load(...)
```

### Key invariants

The always-delegated reconstruction (no `.view()`) removes the alignment/size-match invariants
the draft needed. What remains:

- **Fresh per-tensor buffer:** each tensor reads into its own `bytearray(nbytes)`; no shared
  buffer (avoids any offset/aliasing subtlety, and bounds the host transient to 2× a single
  tensor).
- **Key-set / metadata parity:** `set(sd) == set(safe_open keys)`; `__metadata__` handled
  identically (returned iff `return_metadata`); `sd` returned in **sorted-key order** to match
  safe_open (verified that `safetensors.safe_open().keys()` yields sorted keys) (offset
  order is used only for I/O locality).
- **Dtype coverage parity:** pre-flight membership against `safetensors.torch._TYPES` ⇒ any
  dtype we accept is one `safetensors.torch.load()` can reconstruct; anything else → fallback.

### Tuning

- **DROP_INTERVAL = 512 MB** (was 2 GB in draft). Measured: a 512 MB cap costs no wall-time
  vs 2 GB but tightens the transient. The real transient bound is **`DROP_INTERVAL +
  largest_single_tensor`** accumulated since the last drop — *not* "a few hundred MB"
  unconditionally; state it honestly. On a 128 GB pool even a couple-GB transient is fine,
  but smaller is strictly safer and free.
- **Per-tensor pageable allocation** (no reuse, no pinning — pinning is disabled on the Spark
  and unnecessary; H2D is not the bottleneck, measured ~18 GB/s).

### Logging

Preserve the existing observability tags so the new path is comparable to the old:
- `LOAD file=... method=unified-cuda-stream` (new method tag to distinguish in logs).
- Reuse the existing throttled `LOAD x/total (pct) tensors=...` progress tick.
- Emit `PAGE_CACHE_DROP_TICK` reporting `sys avail` **before/after** so the (now real) effect
  is visible in the log itself.
- Keep the `LOAD done ... Δavail` bookend.
- Emit a `logging.warning` on any fallback to `safe_open` (unknown dtype, parse/read error),
  so a silent return to the slow path is impossible to miss.

Expected post-fix signature: `sys avail` stays roughly flat (minus the growing CUDA
allocation, which `memory_report`'s torch side tracks separately) instead of falling ~2× the
bytes copied.

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Unknown/new dtype → silent slow fallback (regression with no error) | **B2:** resolve via safetensors' own `_getdtype` (coverage == `safe_open`); pre-flight validate every header dtype; unknown → `logging.warning` (visible) + whole-file `safe_open`. CLAUDE.md note: coverage tracks safetensors, not a fork-local map. |
| `.view()` alignment / packed-dtype (`float4_e2m1fn_x2`) / shape math | **B4:** fresh per-tensor offset-0 allocation + asserted size match for tier 1; tier-2 per-tensor `safetensors.torch.load(blob)` delegates packed/exotic reconstruction to the library; tier-3 whole-file `safe_open`. |
| Short reads / `preadv` partial returns | `_pread_exact` loops until `nbytes`; EOF/short mismatch → raise → cleanup → fallback. |
| Partial failure leaves dead CUDA reservation before fallback | **N5:** `del sd` + `soft_empty_cache_unified(force=True)` before `safe_open` retry. |
| Performance regression vs mmap readahead | Sort by offset for sequential I/O. Measured: pread+fadvise was *faster* than `safe_open` cold (4.2s vs 6.5s/4GiB) even without pressure; gap widens under the 60 GB pressure case. Verify wall-time vs baseline on the live load. |
| `sd` iteration order changes | **N3:** read in offset order for locality but return `sd` in sorted-key order to match `safe_open` (safe_open yields sorted keys, verified); a test asserts `list(keys)` equality, not just the set. |
| Second parser in a rebase-prone file | We parse only the 8-byte+JSON header (trivial, stable format), own **no** dtype map (inherited from safetensors), and own **no** byte reconstruction (delegated). `safe_open` fallback ⇒ a rebase dropping the branch degrades to slow-but-correct; the unknown-dtype `warning` surfaces any silent perf regression. |
| safetensors private `_getdtype`/`load` API changes | Guard the import; missing symbol → disable stream path (fall back). A smoke test asserts the import resolves. |

## Verification Plan

1. **Unit / correctness (general, not Flux-specific):** for ≥1 **mixed-dtype** real model,
   load via both `safe_open` and the streamer and assert: (a) `set(keys)` equal incl.
   `__metadata__` handling; (b) per-tensor `dtype`, `shape`, `is_contiguous()`,
   `storage_offset()==0` equal; (c) **byte-equality for every tensor** (uint8 view), not a
   sample. Add to `tests-unit` per CLAUDE.md `load_torch_file` notes (import
   `comfy.model_management` before monkeypatch; `caplog.at_level(INFO)` for tick behavior).
2. **Edge cases:** zero-size tensor (`nbytes==0`); a tensor forced down tier-2; an
   unknown-dtype header → asserts `warning` + `safe_open` fallback + result equals
   `safe_open`; simulated short read → asserts cleanup + fallback.
3. **Downstream assign path:** run a small model through `load_diffusion_model` →
   `load_state_dict(assign=True)` with the streamer; assert no dtype/device surprise and that
   a subsequent `.cpu().numpy()` on a non-weight tensor still works (CLAUDE.md hazard).
4. **Live load (the real test):** restart ComfyUI, load Flux.2 Dev, sample `free -h` at 1 Hz.
   **Pass criteria:** `buff/cache` stays bounded (< `DROP_INTERVAL + largest_tensor`, ≈ <2 GB)
   throughout; **swap usage does not climb** (swap stays enabled as the OOM net, but must not
   be touched); `sys avail` does not fall toward <1 GB; **load wall-time drops markedly from
   ~320 s** toward I/O-bound (60 GB ÷ NVMe bandwidth — the Samsung MZALC4T0 should deliver
   multi-GB/s, so target well under a minute).
5. **Output regression:** generated image **bit-identical** (hash) to a pre-change baseline at
   a fixed seed; `LOAD_BUDGET` shows `fits_fully=True` / `loaded completely` (with the
   already-shipped budget fix).

## Rollout

- Implement behind the gate; default **on** for unified + large files (the only affected
  config is the Spark itself).
- New log method tag `method=unified-cuda-stream`.
- Land as a single focused commit on `master` after a passing live load + green tests.
- Update CLAUDE.md "Load-Path Observability" and the unified-memory gotchas with: the new
  method tag, the mmap-defeats-fadvise rationale, and that the streamer's dtype coverage is
  **inherited from safetensors** (do not reintroduce a fork-local dtype map).
