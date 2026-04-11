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
from comfy.cli_args import _VALID_CACHE_PHASES as _cli_valid
from comfy.cli_args import args


class CachePhase(enum.Enum):
    PRE_CHECKPOINT_LOAD = "pre_checkpoint_load"
    POST_FILE_LOAD = "post_file_load"
    POST_CHECKPOINT_SLICE = "post_checkpoint_slice"
    POST_MODEL_INIT = "post_model_init"
    POST_CHECKPOINT_LOAD = "post_checkpoint_load"
    PRE_INFERENCE = "pre_inference"
    POST_INFERENCE = "post_inference"


# Keep the CLI validator (cli_args._VALID_CACHE_PHASES) in sync with CachePhase.
# cli_args cannot import CachePhase without a circular import, so it mirrors
# the phase names as a plain frozenset of strings. This assertion re-anchors
# that duplication to the enum at import time; any drift fails fast.
assert _cli_valid == frozenset(p.value for p in CachePhase), (
    f"cli_args._VALID_CACHE_PHASES {_cli_valid} drifted from "
    f"CachePhase {frozenset(p.value for p in CachePhase)}"
)


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


# Enforce the GC-subset invariant at import time. Any future edit that adds
# a phase to _PRESET_GC_PHASES without also adding it to _PRESET_PHASES would
# silently become a no-op (the early return in _maybe_drop_impl skips the
# block because `preset_fires = phase in active` is False); turn that into
# a fail-fast startup error instead.
for _preset_name, _gc_set in _PRESET_GC_PHASES.items():
    assert _gc_set.issubset(_PRESET_PHASES[_preset_name]), (
        f"_PRESET_GC_PHASES[{_preset_name!r}] must be a subset of "
        f"_PRESET_PHASES[{_preset_name!r}]; violation: "
        f"{_gc_set - _PRESET_PHASES[_preset_name]}"
    )


# Module-level tracker so maybe_drop() logs at-most-once per phase on failure.
_drop_failures_seen: set[CachePhase] = set()


# Cache of the parsed --cache-drop-at override. Populated on first access by
# _get_override_phases(); keyed implicitly by the `args.cache_drop_at` string,
# which argparse populates once at startup and never mutates afterward.
_parsed_override: frozenset[CachePhase] | None = None
_parsed_override_raw: str | None = None


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


def _get_override_phases() -> frozenset[CachePhase] | None:
    """Return the parsed --cache-drop-at set, or None if no override is set.

    Caches the parse result on first call. argparse has already validated the
    raw string via cli_args._cache_drop_at_type, so _parse_phase_override
    should never raise here — but it's kept defensive for hand-set args
    objects in tests.
    """
    global _parsed_override, _parsed_override_raw
    raw = getattr(args, "cache_drop_at", None)
    if not raw:
        return None
    if raw != _parsed_override_raw:
        _parsed_override = _parse_phase_override(raw)
        _parsed_override_raw = raw
    return _parsed_override


def _active_phases() -> frozenset[CachePhase]:
    override = _get_override_phases()
    if override is not None:
        return override
    preset = getattr(args, "cache_aggressiveness", "normal")
    return _PRESET_PHASES.get(preset, _PRESET_PHASES["normal"])


def _gc_phases() -> frozenset[CachePhase]:
    override = _get_override_phases()
    if override is not None:
        # With an explicit override, we don't second-guess: every phase in the
        # override triggers gc.collect() as well. Users asking for a specific
        # phase list are debugging and want the full hammer. (This behavior
        # is documented in the --cache-drop-at help text.)
        return override
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
