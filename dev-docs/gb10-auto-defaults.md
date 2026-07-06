# GB10 auto-defaults + `--disable-dynamic-vram` deprecation-proofing

Status: **implemented** (Tasks A–D landed on `feat/gb10-auto-defaults`; this note is the
implementation record, written as Task E). Plan:
`/home/ai/.claude/plans/currently-i-am-launching-polished-feigenbaum.md`. Commits (base
`3f4185fd`): `423d6b8f` (A), `ae472612` + `4ccf603b` fix (B), `5dfbd9dc` (C), `9323f671` +
`42c2d483` fix (D).

## Problem

The documented Spark launch command required 7+ hand-passed flags to engage the fork's
unified-memory optimizations. A bare `python main.py` silently routed loads through
upstream's aimdo demand-paged loader — the exact 2x memory duplication the fork exists to
eliminate (see CLAUDE.md's "why aimdo is the wrong paradigm" note). Separately, upstream is
nominally deprecating `--disable-dynamic-vram` (the flag the fork's loader depended on),
which was a latent risk: if a future sync deletes the flag, the fork's single-copy loader
silently stops engaging.

## What auto-applies

On GB10 detection (or `--spark-defaults on`), gated uniformly by
`comfy.spark_defaults.enabled()`:

1. `CUDA_CACHE_MAXSIZE=4294967296` — `os.environ.setdefault(...)` in `apply_early()`
   (`comfy/spark_defaults.py`), so a user's own export always wins.
2. Dynamic-VRAM/aimdo off — `comfy.cli_args.enables_dynamic_vram()` returns `False` via the
   `spark_defaults.enabled()` clause, without reading `--disable-dynamic-vram` at all.
3. `--reserve-vram 1` equivalent — `model_management.py`'s `EXTRA_RESERVED_VRAM` init block
   gets an `elif UNIFIED_MEMORY and comfy.spark_defaults.enabled(): EXTRA_RESERVED_VRAM = 1
   * 1024**3` arm (only reached when the user didn't pass `--reserve-vram` explicitly).
4. Pinned-memory off — new `pinned_memory_disabled()` helper in `model_management.py`,
   OR'd into all three read sites (the `MAX_PINNED_MEMORY` init guard in
   `model_management.py`, and `get_pin`/`pin_memory` in `comfy/pinned_memory.py`).
5. bf16 text-encoder — `text_encoder_dtype()` in `model_management.py` gets a
   `UNIFIED_MEMORY and comfy.spark_defaults.enabled(): return torch.bfloat16` clause,
   checked *after* the five explicit `--fp8*/--fp16/--bf16/--fp32-text-enc` flags so an
   explicit choice always wins by construction.
6. `--cache-aggressiveness` auto-`high` — `cli_args.py`'s default flips from the literal
   `"normal"` to a `None` sentinel; `cache_policy.py::_resolved_preset()` resolves `None` to
   `"high"` when `comfy.spark_defaults.enabled() and mm.UNIFIED_MEMORY and not
   comfy.memory_management.aimdo_enabled`, else `"normal"`.

One startup INFO line (`comfy.spark_defaults.log_summary()`, called after `setup_logger()`
in `main.py`) announces every applied default plus a reminder that fp32-TE users should pass
`--fp32-text-enc`.

## What stays manual, and why

`--bf16-unet`, `--bf16-vae`, `--use-sage-attention`, `--cpu-text-enc` are never
auto-applied:

- **`--bf16-unet`/`--bf16-vae`** are a *global* override that fires before the flagless,
  checkpoint-aware dtype whitelist gets a say. Forcing bf16 would upcast a plain-fp8
  checkpoint 2x, override Wan's fp16 preference, and break fp32-only audio VAEs (which have
  no runtime dtype cast — see CLAUDE.md's `disable_weight_init` note). The flagless path
  already picks the right per-architecture dtype without help.
- **`--use-sage-attention`** is a lossy INT8 quality change, and can hard-exit if the local
  SageAttention build goes stale — not something to silently flip on for every user.
- **`--cpu-text-enc`** is a latency regression for small text encoders (only worth it for
  huge ones, e.g. Flux2's ~33GB mistral TE) and drags clip_vision/audio-encoder loads onto
  CPU with it (`text_encoder_device()` is shared across all of them) — a per-workflow
  tradeoff, not a blanket default.

## Detection mechanism

`comfy.spark_defaults.detect_gb10()` (`functools.lru_cache`-memoized, so probed once):

1. `args.unified_memory` manual force (mirrors the existing `is_unified_memory_system()`
   manual-force precedent in `model_management.py`).
2. DMI probe: `/sys/class/dmi/id/product_name` contains `dgx_spark` (case-insensitive
   substring).
3. `nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader` fallback: matches
   `gb10` (substring) or compute_cap `12.1` (word-boundary regex) on any line.

Each step is independently wrapped in `try/except Exception: pass` so a probe failure just
falls through to the next; the whole function returns `False` if everything fails or nothing
matches. This runs pre-torch (see below), so it can't use `torch.cuda.get_device_properties`
the way `is_unified_memory_system()` does — hence the separate DMI/nvidia-smi probes.

`comfy.spark_defaults.enabled()` combines `detect_gb10()` with the `--spark-defaults
{auto,on,off}` kill-switch (`cli_args.py`, default `auto`): `off` forces `False`
unconditionally (the disable-everything path); `on` forces `True` without probing hardware;
`auto` follows `detect_gb10()`.

A drift check lives in `model_management.py` (once torch is up): if
`comfy.spark_defaults.detect_gb10() != UNIFIED_MEMORY` (the torch-free probe disagrees with
`is_unified_memory_system()`'s sm_121 check), it logs a `SPARK: probe disagreement`
WARNING — a canary for the two detection mechanisms drifting apart.

## Deprecation-proofing rationale

The core idea: **gate on detection, not on the flag being deprecated.**
`enables_dynamic_vram()` (`cli_args.py`) checks `args.enable_dynamic_vram` first (escape
hatch, unchanged, always wins), then `comfy.spark_defaults.enabled()` — if GB10 is
detected, it returns `False` unconditionally and never reads `args.disable_dynamic_vram`.
Only when `spark_defaults.enabled()` is `False` (non-GB10 boxes, or `--spark-defaults off`)
does it fall through to the legacy expression that does read `--disable-dynamic-vram`. This
means:

- A bare `python main.py` on GB10 keeps aimdo off with **zero flags**.
- `--disable-dynamic-vram` still works (back-compat; still accepted by argparse) but the
  fork's loader no longer *depends* on it surviving upstream's deprecation.
- If upstream deletes the flag entirely in a future sync, the GB10 path is unaffected — only
  the legacy fallback expression (reached solely on non-GB10/kill-switch paths) would need a
  rebase touch-up.

**Self-check tripwire** (defense-in-depth, given this exact code region's rebase-conflict
history — see `upstream_sync.md`): a `main.py` WARNING placed right after the aimdo
activation block fires whenever `comfy.model_management.UNIFIED_MEMORY and
comfy.memory_management.aimdo_enabled and comfy.spark_defaults.enabled() and not
args.enable_dynamic_vram` all hold — i.e. aimdo ended up active on unified memory despite
the gate that's supposed to prevent it. It's gated on `spark_defaults.enabled()` (not just
`UNIFIED_MEMORY`) specifically so an explicit `--spark-defaults off` opt-out running aimdo
(correct stock behavior) doesn't false-positive — this gate was added in a fix pass
(`4ccf603b`) after the first version fired a false alarm on exactly that path. The intent:
convert a silent post-rebase regression of `enables_dynamic_vram()`'s gating into a loud
one.

## Consumer-side pattern: no `args` mutation

Every auto-default reads `comfy.spark_defaults.enabled()` live at its own call site; none of
them write into `args`. This matches the existing `--drop-page-cache` auto-enable precedent
in the codebase. Two reasons this mattered enough to call out:

- **Pinned-memory gate must not be module-body-order-dependent.** `pinned_memory_disabled()`
  is a function (not a flag captured once at import time), OR'd into all three read sites,
  so it's immune to rebase reordering of the module bodies.
- **Cache-aggressiveness's `None` sentinel needed explicit resolution**, not a
  `getattr(args, "cache_aggressiveness", "normal")` default — that pattern silently returns
  `None` (not `"normal"`) once the attribute exists-but-is-`None`, so `_resolved_preset()`
  resolves the sentinel explicitly and both `_active_phases()`/`_gc_phases()` and the
  `CACHE_DROP` log call through it, so they can never disagree.

## Kill-switch uniformity (fix passes)

Two consumer gates initially missed the `spark_defaults.enabled()` check and were caught in
fix passes before merge, both fixed to keep `--spark-defaults off` == stock upstream
behavior end-to-end:

- **Task B** (`4ccf603b`): the `main.py` self-check WARNING originally fired even when
  `--spark-defaults off` correctly left aimdo running (the user's explicit choice) —
  added the `comfy.spark_defaults.enabled()` guard.
- **Task D** (`42c2d483`): `cache_policy.py::_resolved_preset()`'s auto-`high` branch
  originally gated on `UNIFIED_MEMORY and not aimdo_enabled` alone, which meant
  `--spark-defaults off` would still resolve to `"high"` on a unified box — added
  `comfy.spark_defaults.enabled()` to the condition so it now resolves to `"normal"`,
  matching the other four auto-defaults' kill-switch behavior.

## Cross-references

- Plan: `/home/ai/.claude/plans/currently-i-am-launching-polished-feigenbaum.md`
- Task reports: `.superpowers/sdd/task-{A,B,C,D,E}-report.md`
- CLAUDE.md: Common Commands (new minimal launch command), and the Development Rules
  bullets on `comfy/spark_defaults.py`'s torch-free contract, the dynamic-vram gate
  rewrite, and the auto-default/collateral/kill-switch notes.
- `upstream_sync.md`: rebase-checklist item for re-verifying the gate, the self-check
  WARNING, and the consumer-side gates after each sync.
- Tests: `tests-unit/comfy_test/test_spark_defaults.py`,
  `tests-unit/comfy_test/test_dynamic_vram_gate.py`,
  `tests-unit/comfy_test/test_spark_consumer_defaults.py`, and the `TestResolvedPreset`
  class in `tests-unit/comfy_test/test_cache_policy.py`.
