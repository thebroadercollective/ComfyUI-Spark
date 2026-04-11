"""Configurable cache-drop policy for model loading on unified memory.

Centralizes the decision of when to call soft_empty_cache_unified() and
gc.collect() during the load path. Three layered controls:

- Preset levels (off / low / normal / high / paranoid) are curated bundles
  of phase hooks. normal is the default and matches pre-existing behavior
  plus a POST_CHECKPOINT_LOAD drop and a PRE_INFERENCE drop.
- Explicit phase list via --cache-drop-at overrides the preset if provided.
- A pressure-driven watermark via --cache-drop-threshold-gb fires on any
  registered phase regardless of preset when psutil.virtual_memory().available
  falls below the threshold.

Call sites invoke maybe_drop(CachePhase.X, reason=...) at known phase
boundaries. This file owns the policy; call sites land in T5.
"""

import enum
import gc
import logging
import psutil

import comfy.model_management as mm
from comfy.cli_args import args


class CachePhase(enum.Enum):
    PRE_CHECKPOINT_LOAD = "pre_checkpoint_load"
    POST_FILE_LOAD = "post_file_load"
    POST_CHECKPOINT_SLICE = "post_checkpoint_slice"
    POST_MODEL_INIT = "post_model_init"
    POST_CHECKPOINT_LOAD = "post_checkpoint_load"
    PRE_INFERENCE = "pre_inference"
    POST_INFERENCE = "post_inference"


# Preset -> active phases. A phase present in the set triggers a drop at
# that callsite when the preset is active.
_PRESET_PHASES: dict[str, frozenset[CachePhase]] = {
    "off": frozenset(),
    "low": frozenset({CachePhase.POST_CHECKPOINT_LOAD}),
    "normal": frozenset({
        CachePhase.POST_CHECKPOINT_LOAD,
        CachePhase.PRE_INFERENCE,
    }),
    "high": frozenset({
        CachePhase.POST_FILE_LOAD,
        CachePhase.POST_MODEL_INIT,
        CachePhase.POST_CHECKPOINT_LOAD,
        CachePhase.PRE_INFERENCE,
    }),
    "paranoid": frozenset(CachePhase),
}


# Preset -> phases that also trigger gc.collect() in addition to the
# allocator-level soft_empty_cache_unified(). Must be a subset of the
# preset's active phases.
_PRESET_GC_PHASES: dict[str, frozenset[CachePhase]] = {
    "off": frozenset(),
    "low": frozenset(),
    "normal": frozenset({CachePhase.POST_CHECKPOINT_LOAD}),
    "high": frozenset({
        CachePhase.POST_MODEL_INIT,
        CachePhase.POST_CHECKPOINT_LOAD,
    }),
    "paranoid": frozenset(CachePhase),
}


# Module-level tracker so maybe_drop() logs at-most-once per phase on failure.
_drop_failures_seen: set[CachePhase] = set()


def _parse_phase_override(phase_list_str: str) -> frozenset[CachePhase]:
    """Parse --cache-drop-at comma-separated phase names.

    Raises ValueError if any name does not match a CachePhase value.
    """
    names = [s.strip() for s in phase_list_str.split(",") if s.strip()]
    out: set[CachePhase] = set()
    valid = {p.value for p in CachePhase}
    for n in names:
        lower = n.lower()
        if lower not in valid:
            raise ValueError(
                f"Unknown cache phase '{n}'. Valid phases: {sorted(valid)}"
            )
        out.add(CachePhase(lower))
    return frozenset(out)


def _active_phases() -> frozenset[CachePhase]:
    override = getattr(args, "cache_drop_at", None)
    if override:
        return _parse_phase_override(override)
    preset = getattr(args, "cache_aggressiveness", "normal")
    return _PRESET_PHASES.get(preset, _PRESET_PHASES["normal"])


def _gc_phases() -> frozenset[CachePhase]:
    override = getattr(args, "cache_drop_at", None)
    if override:
        # With an explicit override, we don't second-guess: every phase in the
        # override triggers gc.collect() as well. Users asking for a specific
        # phase list are debugging and want the full hammer.
        return _parse_phase_override(override)
    preset = getattr(args, "cache_aggressiveness", "normal")
    return _PRESET_GC_PHASES.get(preset, _PRESET_GC_PHASES["normal"])


def _watermark_bytes() -> int | None:
    gb = getattr(args, "cache_drop_threshold_gb", None)
    if gb is None or gb <= 0:
        return None
    return int(gb * (1024 ** 3))


def _watermark_triggered() -> bool:
    threshold = _watermark_bytes()
    if threshold is None:
        return False
    try:
        return psutil.virtual_memory().available < threshold
    except Exception:
        return False


def _maybe_drop_impl(phase: CachePhase, *, reason: str = "") -> None:
    """Real body of maybe_drop(). Wrapped by maybe_drop() for error suppression.

    - If the phase is in the preset's active set, drop (soft_empty_cache_unified).
    - If the phase is in the preset's gc set, also gc.collect().
    - If the watermark is configured and current sys avail is below it, drop
      and gc.collect() regardless of preset.
    - If neither condition fires, return without side effects.
    """
    active = _active_phases()
    gc_set = _gc_phases()
    watermark = _watermark_triggered()

    preset_fires = phase in active
    if not preset_fires and not watermark:
        return

    if watermark and not preset_fires:
        trigger = "watermark"
    elif watermark:
        trigger = "preset+watermark"
    else:
        trigger = "preset"

    before = mm.memory_report()
    mm.soft_empty_cache_unified()
    should_gc = watermark or (phase in gc_set)
    if should_gc:
        gc.collect()
    after = mm.memory_report()

    preset_name = getattr(args, "cache_aggressiveness", "normal")
    logging.info(
        "CACHE_DROP phase=%s preset=%s trigger=%s%s gc=%s | %s",
        phase.value,
        preset_name,
        trigger,
        f" reason={reason}" if reason else "",
        "yes" if should_gc else "no",
        mm.memory_delta(before, after),
    )


def maybe_drop(phase: CachePhase, *, reason: str = "") -> None:
    """Drop caches at a phase boundary if the active policy allows it.

    Never raises: call sites on the hot path of model loading do not need
    their own try/except. Failures are logged at WARNING at-most-once per
    phase (tracked via module-level `_drop_failures_seen`).
    """
    try:
        _maybe_drop_impl(phase, reason=reason)
    except Exception as e:
        if phase not in _drop_failures_seen:
            _drop_failures_seen.add(phase)
            logging.warning(
                "cache_policy.maybe_drop failed for %s: %s "
                "(suppressing further errors for this phase)",
                phase.value,
                e,
            )
