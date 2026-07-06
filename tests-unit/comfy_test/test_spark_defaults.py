"""Unit tests for comfy.spark_defaults (GB10 auto-defaults foundation).

Task A scope: GB10 detection (detect_gb10), the enabled()/kill-switch interface that
Tasks B/C/D will consume, and apply_early()'s single side effect
(CUDA_CACHE_MAXSIZE setdefault). Later tasks (dynamic-vram gate, reserve-vram,
pinned-memory, bf16-text-enc, cache-aggressiveness) are NOT covered here.

Per CLAUDE.md testing notes: import comfy.cli_args / comfy.spark_defaults before
monkeypatch.setattr (namespace packages aren't auto-resolved otherwise). detect_gb10
is lru_cache'd -- clear it between cases so fixtures don't bleed across tests.
"""

import os
import subprocess
import logging

import pytest

import comfy.cli_args
import comfy.spark_defaults


@pytest.fixture(autouse=True)
def clear_detect_cache():
    """Ensure detect_gb10's memo doesn't leak between test cases."""
    comfy.spark_defaults.detect_gb10.cache_clear()
    yield
    comfy.spark_defaults.detect_gb10.cache_clear()


def _set_args(monkeypatch, *, unified_memory=None, spark_defaults=None, delete_spark_defaults=False):
    if unified_memory is not None:
        monkeypatch.setattr(comfy.cli_args.args, "unified_memory", unified_memory, raising=False)
    if delete_spark_defaults:
        monkeypatch.delattr(comfy.cli_args.args, "spark_defaults", raising=False)
    elif spark_defaults is not None:
        monkeypatch.setattr(comfy.cli_args.args, "spark_defaults", spark_defaults, raising=False)


class TestDetectGB10:
    def test_manual_force_via_unified_memory_flag(self, monkeypatch):
        _set_args(monkeypatch, unified_memory=True)
        # Even if the DMI/nvidia-smi probes would say "no", the manual flag wins.
        monkeypatch.setattr(comfy.spark_defaults, "_read_dmi_product_name", lambda: "")
        monkeypatch.setattr(comfy.spark_defaults, "_probe_nvidia_smi", lambda: "")
        assert comfy.spark_defaults.detect_gb10() is True

    def test_dmi_product_name_dgx_spark_matches(self, monkeypatch):
        _set_args(monkeypatch, unified_memory=False)
        monkeypatch.setattr(
            comfy.spark_defaults, "_read_dmi_product_name", lambda: "NVIDIA_DGX_Spark"
        )
        monkeypatch.setattr(comfy.spark_defaults, "_probe_nvidia_smi", lambda: "")
        assert comfy.spark_defaults.detect_gb10() is True

    def test_dmi_product_name_case_insensitive_substring(self, monkeypatch):
        _set_args(monkeypatch, unified_memory=False)
        monkeypatch.setattr(
            comfy.spark_defaults, "_read_dmi_product_name", lambda: "some_dgx_spark_variant"
        )
        monkeypatch.setattr(comfy.spark_defaults, "_probe_nvidia_smi", lambda: "")
        assert comfy.spark_defaults.detect_gb10() is True

    def test_nvidia_smi_gb10_fixture_matches(self, monkeypatch):
        _set_args(monkeypatch, unified_memory=False)
        monkeypatch.setattr(comfy.spark_defaults, "_read_dmi_product_name", lambda: "")
        monkeypatch.setattr(
            comfy.spark_defaults, "_probe_nvidia_smi", lambda: "NVIDIA GB10, 12.1\n"
        )
        assert comfy.spark_defaults.detect_gb10() is True

    def test_nvidia_smi_compute_cap_12_1_matches_without_gb10_string(self, monkeypatch):
        _set_args(monkeypatch, unified_memory=False)
        monkeypatch.setattr(comfy.spark_defaults, "_read_dmi_product_name", lambda: "")
        monkeypatch.setattr(
            comfy.spark_defaults, "_probe_nvidia_smi", lambda: "Some Card, 12.1\n"
        )
        assert comfy.spark_defaults.detect_gb10() is True

    def test_non_gb10_fixture_returns_false(self, monkeypatch):
        _set_args(monkeypatch, unified_memory=False)
        monkeypatch.setattr(comfy.spark_defaults, "_read_dmi_product_name", lambda: "")
        monkeypatch.setattr(
            comfy.spark_defaults,
            "_probe_nvidia_smi",
            lambda: "NVIDIA GeForce RTX 4090, 8.9\n",
        )
        assert comfy.spark_defaults.detect_gb10() is False

    def test_dmi_probe_raising_falls_through_to_nvidia_smi(self, monkeypatch):
        _set_args(monkeypatch, unified_memory=False)

        def _raise():
            raise FileNotFoundError("no such file")

        monkeypatch.setattr(comfy.spark_defaults, "_read_dmi_product_name", _raise)
        monkeypatch.setattr(
            comfy.spark_defaults, "_probe_nvidia_smi", lambda: "NVIDIA GB10, 12.1\n"
        )
        assert comfy.spark_defaults.detect_gb10() is True

    def test_nvidia_smi_raising_returns_false(self, monkeypatch):
        _set_args(monkeypatch, unified_memory=False)
        monkeypatch.setattr(comfy.spark_defaults, "_read_dmi_product_name", lambda: "")

        def _raise():
            raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5)

        monkeypatch.setattr(comfy.spark_defaults, "_probe_nvidia_smi", _raise)
        assert comfy.spark_defaults.detect_gb10() is False

    def test_detection_is_cached_single_probe(self, monkeypatch):
        _set_args(monkeypatch, unified_memory=False)
        monkeypatch.setattr(comfy.spark_defaults, "_read_dmi_product_name", lambda: "")

        calls = {"n": 0}

        def _counting_probe():
            calls["n"] += 1
            return "NVIDIA GB10, 12.1\n"

        monkeypatch.setattr(comfy.spark_defaults, "_probe_nvidia_smi", _counting_probe)

        assert comfy.spark_defaults.detect_gb10() is True
        assert comfy.spark_defaults.detect_gb10() is True
        assert comfy.spark_defaults.detect_gb10() is True
        assert calls["n"] == 1


class TestEnabled:
    def test_kill_switch_off_beats_detection_true(self, monkeypatch):
        _set_args(monkeypatch, spark_defaults="off")
        monkeypatch.setattr(comfy.spark_defaults, "detect_gb10", lambda: True)
        assert comfy.spark_defaults.enabled() is False

    def test_force_on_beats_detection_false(self, monkeypatch):
        _set_args(monkeypatch, spark_defaults="on")
        monkeypatch.setattr(comfy.spark_defaults, "detect_gb10", lambda: False)
        assert comfy.spark_defaults.enabled() is True

    def test_auto_follows_detection_true(self, monkeypatch):
        _set_args(monkeypatch, spark_defaults="auto")
        monkeypatch.setattr(comfy.spark_defaults, "detect_gb10", lambda: True)
        assert comfy.spark_defaults.enabled() is True

    def test_auto_follows_detection_false(self, monkeypatch):
        _set_args(monkeypatch, spark_defaults="auto")
        monkeypatch.setattr(comfy.spark_defaults, "detect_gb10", lambda: False)
        assert comfy.spark_defaults.enabled() is False

    def test_missing_spark_defaults_attr_defaults_to_auto(self, monkeypatch):
        _set_args(monkeypatch, delete_spark_defaults=True)
        monkeypatch.setattr(comfy.spark_defaults, "detect_gb10", lambda: True)
        assert comfy.spark_defaults.enabled() is True


class TestApplyEarly:
    def test_sets_cuda_cache_maxsize_when_unset(self, monkeypatch):
        monkeypatch.delenv("CUDA_CACHE_MAXSIZE", raising=False)
        monkeypatch.setattr(comfy.spark_defaults, "enabled", lambda: True)
        comfy.spark_defaults.apply_early(comfy.cli_args.args)
        assert os.environ["CUDA_CACHE_MAXSIZE"] == "4294967296"

    def test_does_not_overwrite_existing_cuda_cache_maxsize(self, monkeypatch):
        monkeypatch.setenv("CUDA_CACHE_MAXSIZE", "1234")
        monkeypatch.setattr(comfy.spark_defaults, "enabled", lambda: True)
        comfy.spark_defaults.apply_early(comfy.cli_args.args)
        assert os.environ["CUDA_CACHE_MAXSIZE"] == "1234"

    def test_does_nothing_when_not_enabled(self, monkeypatch):
        monkeypatch.delenv("CUDA_CACHE_MAXSIZE", raising=False)
        monkeypatch.setattr(comfy.spark_defaults, "enabled", lambda: False)
        comfy.spark_defaults.apply_early(comfy.cli_args.args)
        assert "CUDA_CACHE_MAXSIZE" not in os.environ

    def test_stashes_summary_string_when_enabled(self, monkeypatch):
        monkeypatch.delenv("CUDA_CACHE_MAXSIZE", raising=False)
        monkeypatch.setattr(comfy.spark_defaults, "enabled", lambda: True)
        comfy.spark_defaults.apply_early(comfy.cli_args.args)
        assert comfy.spark_defaults.get_summary()
        assert "CUDA_CACHE_MAXSIZE" in comfy.spark_defaults.get_summary()


class TestLogSummary:
    def test_logs_one_info_line_when_enabled(self, monkeypatch, caplog):
        monkeypatch.setattr(comfy.spark_defaults, "enabled", lambda: True)
        monkeypatch.setattr(
            comfy.spark_defaults, "get_summary", lambda: "CUDA_CACHE_MAXSIZE=4294967296"
        )
        with caplog.at_level(logging.INFO):
            comfy.spark_defaults.log_summary()
        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(info_records) == 1
        assert "CUDA_CACHE_MAXSIZE=4294967296" in info_records[0].message
        assert "fp32-text-enc" in info_records[0].message

    def test_logs_nothing_when_disabled(self, monkeypatch, caplog):
        monkeypatch.setattr(comfy.spark_defaults, "enabled", lambda: False)
        with caplog.at_level(logging.INFO):
            comfy.spark_defaults.log_summary()
        assert len(caplog.records) == 0


class TestNoTorchImport:
    def test_module_source_has_no_torch_or_model_management_import_statements(self):
        """spark_defaults must stay import-cheap: no real import statement may
        pull in torch, comfy.model_management, or comfy.cache_policy (mentions in
        prose/comments are fine and expected -- this module documents why it
        avoids them).

        Generalized to reject ANY real import line containing the "torch" token
        (e.g. "import torch as t", "from torch import nn"), not just an exact
        "import torch" match -- an exact-match check would miss those variants.
        Docstrings/comments are excluded by only inspecting lines that start
        with "import " or "from ".
        """
        src_path = comfy.spark_defaults.__file__
        with open(src_path) as f:
            lines = f.readlines()
        for line in lines:
            stripped = line.strip()
            if not (stripped.startswith("import ") or stripped.startswith("from ")):
                continue
            assert "torch" not in stripped, stripped
            assert "comfy.model_management" not in stripped, stripped
            assert "comfy.cache_policy" not in stripped, stripped
