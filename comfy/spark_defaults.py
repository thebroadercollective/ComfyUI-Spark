"""GB10 (NVIDIA DGX Spark) auto-defaults: detection + kill-switch + earliest wiring.

This module is imported extremely early in main.py -- before torch is imported and
before CUDA is initialized (early CUDA init would break the CUDA_VISIBLE_DEVICES /
--default-device env setup that happens later). It MUST stay torch-free and
import-cheap: stdlib only (os, subprocess, re, logging, functools). Never import
comfy.model_management or comfy.cache_policy at module level (both pull in torch,
directly or transitively).

`enabled()` is the shared interface later tasks (auto-defaults for dynamic-vram,
reserve-vram, pinned-memory, bf16-text-enc, cache-aggressiveness) consume to decide
whether to apply their own GB10-specific default. Task A implements ONLY: detection,
the enabled()/kill-switch interface, and the single CUDA_CACHE_MAXSIZE side effect.
"""

import functools
import logging
import os
import re
import subprocess

log = logging.getLogger(__name__)

_DMI_PRODUCT_NAME_PATH = "/sys/class/dmi/id/product_name"

# Populated by apply_early(); read by log_summary(). Human-readable description of
# every GB10 auto-default that will be applied (across all tasks, not just this one).
_summary = ""


def _read_dmi_product_name() -> str:
    """Read /sys/class/dmi/id/product_name. Raises on any failure (caller handles)."""
    with open(_DMI_PRODUCT_NAME_PATH) as f:
        return f.read().strip()


def _probe_nvidia_smi() -> str:
    """Run `nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader`.

    Raises on any failure (caller handles). Bounded by a timeout so a hung
    nvidia-smi can't stall startup.
    """
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
        timeout=5,
    )
    if isinstance(out, bytes):
        out = out.decode("utf-8", errors="replace")
    return out


@functools.lru_cache(maxsize=1)
def detect_gb10() -> bool:
    """Return True if running on DGX Spark / GB10 hardware.

    Detection order, each step wrapped so it can never raise and falls through to
    the next on any failure:
      1. Manual force via --unified-memory.
      2. DMI product_name probe (primary): substring "DGX_Spark" (case-insensitive).
      3. nvidia-smi fallback: GPU name contains "GB10" OR compute_cap is "12.1".
    Returns False if every probe fails or none match.
    """
    try:
        from comfy.cli_args import args

        if args.unified_memory:
            return True
    except Exception:
        pass

    try:
        product_name = _read_dmi_product_name()
        if "dgx_spark" in product_name.lower():
            return True
    except Exception:
        pass

    try:
        smi_out = _probe_nvidia_smi()
        for line in smi_out.splitlines():
            if not line.strip():
                continue
            if "gb10" in line.lower():
                return True
            if re.search(r"\b12\.1\b", line):
                return True
    except Exception:
        pass

    return False


def enabled() -> bool:
    """Combine detect_gb10() with the --spark-defaults kill-switch.

    - "off"  -> False (disables ALL Spark auto-defaults, regardless of detection)
    - "on"   -> True (force on, skip detection)
    - "auto" (default) -> follow detect_gb10()
    """
    from comfy.cli_args import args

    mode = getattr(args, "spark_defaults", "auto")
    if mode == "off":
        return False
    if mode == "on":
        return True
    return detect_gb10()


def get_summary() -> str:
    """Return the human-readable summary stashed by apply_early(), if any."""
    return _summary


def apply_early(args) -> None:
    """Apply the earliest (pre-torch, pre-logging) GB10 auto-default.

    Only side effect: os.environ.setdefault("CUDA_CACHE_MAXSIZE", ...) when
    enabled() -- setdefault so a user's own export always wins. Never touches
    CUDA_CACHE_DISABLE. Logging is not configured yet at this point in main.py, so
    this function must not log; it only stashes a summary string for log_summary()
    to emit later.
    """
    global _summary
    if not enabled():
        return

    os.environ.setdefault("CUDA_CACHE_MAXSIZE", "4294967296")

    # Descriptive summary of every GB10 auto-default this feature applies across
    # all tasks (A: CUDA_CACHE_MAXSIZE; B/C/D: dynamic-vram/reserve-vram/pinned-
    # memory/bf16-text-enc/cache-aggressiveness), so a user reading the startup
    # log knows what changed.
    _summary = (
        "CUDA_CACHE_MAXSIZE=4294967296, dynamic-vram off, reserve-vram 1GB, "
        "pinned-memory off, bf16-text-enc, cache-aggressiveness high"
    )


def log_summary() -> None:
    """Emit one INFO line naming the applied GB10 auto-defaults, if enabled().

    Called from main.py after setup_logger(...) so logging is guaranteed to be
    configured. Emits nothing when not enabled().
    """
    if not enabled():
        return
    log.info(
        "DGX Spark (GB10) auto-defaults active: %s. If you rely on full fp32 "
        "text-encoder precision, pass --fp32-text-enc to override.",
        get_summary(),
    )
