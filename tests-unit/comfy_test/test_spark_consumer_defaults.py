"""Unit tests for Task C's GB10 unified-memory consumer-side auto-defaults, wired
into comfy/model_management.py (+ comfy/pinned_memory.py's two read sites).

Task C scope: reserve-vram default (1GB), pinned-memory-off (via the shared
`pinned_memory_disabled()` helper), bf16 text-encoder default, and the probe-
agreement warning. All four are gated on `UNIFIED_MEMORY and
comfy.spark_defaults.enabled()`, and NEVER mutate `args` -- they change what the
code *reads*, consumer-side (matching the --drop-page-cache precedent).

Per CLAUDE.md testing notes: `import comfy.model_management` (namespace package,
not auto-resolved) BEFORE `monkeypatch.setattr(comfy.model_management, ...)`.

Args-object gotcha: patch attributes on `comfy.model_management.args`, NOT
`comfy.cli_args.args`. `model_management.py` does `from comfy.cli_args import
args`, binding a name to whatever object `comfy.cli_args.args` pointed to at
model_management's own import time. `tests-unit/comfy_test/folder_path_test.py`
does `reload(comfy.cli_args)` in its fixtures, which rebinds `comfy.cli_args.args`
to a brand-new object -- `comfy.model_management`'s already-bound `args` name does
NOT follow that rebind. Patching via `comfy.cli_args.args` is therefore order-
dependent (passes in isolation, fails after folder_path_test.py runs first in the
same session); patching via `comfy.model_management.args` targets the exact
object `text_encoder_dtype()`/`pinned_memory_disabled()` actually read, regardless
of what `comfy.cli_args.args` currently points to.

Per the task brief, the module-body constants (EXTRA_RESERVED_VRAM,
MAX_PINNED_MEMORY) are set once at import time from real machine state and are
awkward to unit-test directly without contorting the module to make them
injectable -- that's explicitly out of scope here. This file instead covers the
LOGIC that drives them: `text_encoder_dtype()` (a pure function reading the same
gate the reserve-vram elif uses) and `pinned_memory_disabled()` (the shared helper
that both the module-body init and pinned_memory.py's two read sites call). The
reserve-vram module-body `elif` and MAX_PINNED_MEMORY module-body init are
E2E-verified in Wave 3.
"""

import comfy.model_management  # noqa: F401 -- namespace pkg; import before monkeypatch
import comfy.spark_defaults
import torch


def _set_te_flags(monkeypatch, *, fp8_e4m3fn=False, fp8_e5m2=False, fp16=False, bf16=False, fp32=False):
    monkeypatch.setattr(comfy.model_management.args, "fp8_e4m3fn_text_enc", fp8_e4m3fn, raising=False)
    monkeypatch.setattr(comfy.model_management.args, "fp8_e5m2_text_enc", fp8_e5m2, raising=False)
    monkeypatch.setattr(comfy.model_management.args, "fp16_text_enc", fp16, raising=False)
    monkeypatch.setattr(comfy.model_management.args, "bf16_text_enc", bf16, raising=False)
    monkeypatch.setattr(comfy.model_management.args, "fp32_text_enc", fp32, raising=False)


class TestTextEncoderDtypeGate:
    def test_gate_on_no_explicit_flag_default_device_returns_bf16(self, monkeypatch):
        _set_te_flags(monkeypatch)
        monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", True)
        monkeypatch.setattr(comfy.spark_defaults, "enabled", lambda: True)
        assert comfy.model_management.text_encoder_dtype(device=None) == torch.bfloat16

    def test_gate_on_no_explicit_flag_cpu_device_returns_bf16(self, monkeypatch):
        _set_te_flags(monkeypatch)
        monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", True)
        monkeypatch.setattr(comfy.spark_defaults, "enabled", lambda: True)
        assert comfy.model_management.text_encoder_dtype(device=torch.device("cpu")) == torch.bfloat16

    def test_gate_on_explicit_fp16_flag_still_wins(self, monkeypatch):
        _set_te_flags(monkeypatch, fp16=True)
        monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", True)
        monkeypatch.setattr(comfy.spark_defaults, "enabled", lambda: True)
        assert comfy.model_management.text_encoder_dtype(device=None) == torch.float16

    def test_gate_on_explicit_fp32_flag_still_wins(self, monkeypatch):
        _set_te_flags(monkeypatch, fp32=True)
        monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", True)
        monkeypatch.setattr(comfy.spark_defaults, "enabled", lambda: True)
        assert comfy.model_management.text_encoder_dtype(device=None) == torch.float32

    def test_gate_off_unchanged_default_falls_through_to_fp16(self, monkeypatch):
        _set_te_flags(monkeypatch)
        monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", False)
        monkeypatch.setattr(comfy.spark_defaults, "enabled", lambda: True)
        assert comfy.model_management.text_encoder_dtype(device=None) == torch.float16

    def test_unified_true_but_spark_defaults_disabled_falls_through_to_fp16(self, monkeypatch):
        _set_te_flags(monkeypatch)
        monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", True)
        monkeypatch.setattr(comfy.spark_defaults, "enabled", lambda: False)
        assert comfy.model_management.text_encoder_dtype(device=None) == torch.float16


class TestPinnedMemoryDisabled:
    def test_true_when_explicit_flag_set_gate_off(self, monkeypatch):
        monkeypatch.setattr(comfy.model_management.args, "disable_pinned_memory", True, raising=False)
        monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", False)
        monkeypatch.setattr(comfy.spark_defaults, "enabled", lambda: False)
        assert comfy.model_management.pinned_memory_disabled() is True

    def test_true_when_gate_on_flag_unset(self, monkeypatch):
        monkeypatch.setattr(comfy.model_management.args, "disable_pinned_memory", False, raising=False)
        monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", True)
        monkeypatch.setattr(comfy.spark_defaults, "enabled", lambda: True)
        assert comfy.model_management.pinned_memory_disabled() is True

    def test_false_when_neither(self, monkeypatch):
        monkeypatch.setattr(comfy.model_management.args, "disable_pinned_memory", False, raising=False)
        monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", False)
        monkeypatch.setattr(comfy.spark_defaults, "enabled", lambda: False)
        assert comfy.model_management.pinned_memory_disabled() is False

    def test_true_when_both_idempotent(self, monkeypatch):
        monkeypatch.setattr(comfy.model_management.args, "disable_pinned_memory", True, raising=False)
        monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", True)
        monkeypatch.setattr(comfy.spark_defaults, "enabled", lambda: True)
        assert comfy.model_management.pinned_memory_disabled() is True

    def test_unified_true_but_spark_defaults_disabled_flag_unset_returns_false(self, monkeypatch):
        monkeypatch.setattr(comfy.model_management.args, "disable_pinned_memory", False, raising=False)
        monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", True)
        monkeypatch.setattr(comfy.spark_defaults, "enabled", lambda: False)
        assert comfy.model_management.pinned_memory_disabled() is False
