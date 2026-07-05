"""Unit tests for comfy.cli_args.enables_dynamic_vram()'s GB10 auto-defaults gate.

Task B scope: enables_dynamic_vram() must return False when spark_defaults.enabled()
is True (GB10 auto-default: keep the fork's single-copy loader engaged without
needing --disable-dynamic-vram), while --enable-dynamic-vram keeps winning first as
the explicit escape hatch. Non-GB10 / --spark-defaults off behavior is unchanged.

Per CLAUDE.md testing notes: import comfy.cli_args / comfy.spark_defaults before
monkeypatch.setattr (namespace packages aren't auto-resolved otherwise).
"""

import comfy.cli_args
import comfy.spark_defaults


def _set_args(monkeypatch, **kwargs):
    for name, value in kwargs.items():
        monkeypatch.setattr(comfy.cli_args.args, name, value, raising=False)


class TestEnablesDynamicVramSparkGate:
    def test_bare_gb10_disables_dynamic_vram(self, monkeypatch):
        """spark_defaults.enabled() True, no explicit flags -> False (GB10 auto-default)."""
        monkeypatch.setattr(comfy.spark_defaults, "enabled", lambda: True)
        _set_args(
            monkeypatch,
            enable_dynamic_vram=False,
            disable_dynamic_vram=False,
            highvram=False,
            gpu_only=False,
            novram=False,
            cpu=False,
        )
        assert comfy.cli_args.enables_dynamic_vram() is False

    def test_explicit_enable_wins_over_spark_defaults(self, monkeypatch):
        """--enable-dynamic-vram is the escape hatch: wins even when GB10 detected."""
        monkeypatch.setattr(comfy.spark_defaults, "enabled", lambda: True)
        _set_args(
            monkeypatch,
            enable_dynamic_vram=True,
            disable_dynamic_vram=False,
            highvram=False,
            gpu_only=False,
            novram=False,
            cpu=False,
        )
        assert comfy.cli_args.enables_dynamic_vram() is True

    def test_spark_defaults_off_falls_through_to_legacy_expression(self, monkeypatch):
        """--spark-defaults off (simulated) -> legacy upstream behavior restored."""
        monkeypatch.setattr(comfy.spark_defaults, "enabled", lambda: False)
        _set_args(
            monkeypatch,
            enable_dynamic_vram=False,
            disable_dynamic_vram=False,
            highvram=False,
            gpu_only=False,
            novram=False,
            cpu=False,
        )
        assert comfy.cli_args.enables_dynamic_vram() is True

    def test_non_gb10_disable_dynamic_vram_flag_unchanged(self, monkeypatch):
        """Non-GB10 machine, --disable-dynamic-vram set -> False (legacy behavior unchanged)."""
        monkeypatch.setattr(comfy.spark_defaults, "enabled", lambda: False)
        _set_args(
            monkeypatch,
            enable_dynamic_vram=False,
            disable_dynamic_vram=True,
            highvram=False,
            gpu_only=False,
            novram=False,
            cpu=False,
        )
        assert comfy.cli_args.enables_dynamic_vram() is False
