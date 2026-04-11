"""Unit tests for comfy.cache_policy preset mapping, phase parsing, and watermark trigger."""

import pytest
from unittest.mock import patch, MagicMock

from comfy.cache_policy import (
    CachePhase,
    _PRESET_PHASES,
    _PRESET_GC_PHASES,
    _parse_phase_override,
    _watermark_triggered,
)


class TestPresetPhaseMapping:
    """Table-driven tests for the preset -> active-phases mapping."""

    @pytest.mark.parametrize("preset,expected_phases", [
        ("off", frozenset()),
        ("low", frozenset({CachePhase.POST_CHECKPOINT_LOAD})),
        ("normal", frozenset({CachePhase.POST_CHECKPOINT_LOAD, CachePhase.PRE_INFERENCE})),
        ("high", frozenset({
            CachePhase.POST_FILE_LOAD, CachePhase.POST_MODEL_INIT,
            CachePhase.POST_CHECKPOINT_LOAD, CachePhase.PRE_INFERENCE,
        })),
        ("paranoid", frozenset(CachePhase)),
    ])
    def test_preset_phases(self, preset, expected_phases):
        assert _PRESET_PHASES[preset] == expected_phases

    @pytest.mark.parametrize("preset", list(_PRESET_PHASES.keys()))
    def test_gc_phases_subset_of_active_phases(self, preset):
        """GC phases must be a subset of the active phases for every preset."""
        assert _PRESET_GC_PHASES[preset].issubset(_PRESET_PHASES[preset])

    def test_paranoid_includes_all_phases(self):
        """paranoid must include every CachePhase member."""
        assert _PRESET_PHASES["paranoid"] == frozenset(CachePhase)
        assert _PRESET_GC_PHASES["paranoid"] == frozenset(CachePhase)


class TestParsePhaseOverride:
    def test_single_phase(self):
        result = _parse_phase_override("pre_inference")
        assert result == frozenset({CachePhase.PRE_INFERENCE})

    def test_multiple_phases(self):
        result = _parse_phase_override("post_file_load,pre_inference")
        assert result == frozenset({CachePhase.POST_FILE_LOAD, CachePhase.PRE_INFERENCE})

    def test_whitespace_tolerance(self):
        result = _parse_phase_override(" post_file_load , pre_inference ")
        assert result == frozenset({CachePhase.POST_FILE_LOAD, CachePhase.PRE_INFERENCE})

    def test_unknown_phase_raises(self):
        with pytest.raises(ValueError, match="Unknown cache phase"):
            _parse_phase_override("not_a_phase")

    def test_empty_string(self):
        result = _parse_phase_override("")
        assert result == frozenset()


class TestWatermarkTrigger:
    def test_watermark_triggers_when_below_threshold(self, monkeypatch):
        """Watermark should fire when available memory < threshold."""
        mock_args = MagicMock()
        mock_args.cache_drop_threshold_gb = 20.0
        mock_args.cache_aggressiveness = "normal"
        mock_args.cache_drop_at = None
        monkeypatch.setattr("comfy.cache_policy.args", mock_args)

        mock_vm = MagicMock()
        mock_vm.available = 10 * 1024**3  # 10GB
        with patch("comfy.cache_policy.psutil.virtual_memory", return_value=mock_vm):
            assert _watermark_triggered() is True

    def test_watermark_does_not_trigger_when_above_threshold(self, monkeypatch):
        mock_args = MagicMock()
        mock_args.cache_drop_threshold_gb = 20.0
        mock_args.cache_aggressiveness = "normal"
        mock_args.cache_drop_at = None
        monkeypatch.setattr("comfy.cache_policy.args", mock_args)

        mock_vm = MagicMock()
        mock_vm.available = 30 * 1024**3  # 30GB (above 20GB)
        with patch("comfy.cache_policy.psutil.virtual_memory", return_value=mock_vm):
            assert _watermark_triggered() is False

    def test_watermark_inactive_when_threshold_not_set(self, monkeypatch):
        mock_args = MagicMock()
        mock_args.cache_drop_threshold_gb = None
        mock_args.cache_aggressiveness = "normal"
        mock_args.cache_drop_at = None
        monkeypatch.setattr("comfy.cache_policy.args", mock_args)

        assert _watermark_triggered() is False

    def test_watermark_returns_false_on_psutil_error(self, monkeypatch):
        mock_args = MagicMock()
        mock_args.cache_drop_threshold_gb = 20.0
        mock_args.cache_aggressiveness = "normal"
        mock_args.cache_drop_at = None
        monkeypatch.setattr("comfy.cache_policy.args", mock_args)

        with patch("comfy.cache_policy.psutil.virtual_memory", side_effect=OSError("psutil broken")):
            assert _watermark_triggered() is False
