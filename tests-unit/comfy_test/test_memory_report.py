"""Unit tests for comfy.model_management.memory_report() and memory_delta()."""

from unittest.mock import MagicMock

import comfy.model_management as mm


class TestMemoryReport:
    def test_format_with_cuda_available(self, monkeypatch):
        """When CUDA is available, format should be 'alloc X.XG res X.XG free X.XG | sys avail X.XG used X.XG'."""
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        monkeypatch.setattr("torch.cuda.current_device", lambda: 0)
        monkeypatch.setattr("torch.cuda.memory_allocated", lambda device=None: int(45.2 * 1024**3))
        monkeypatch.setattr("torch.cuda.memory_reserved", lambda device=None: int(46.1 * 1024**3))
        monkeypatch.setattr("torch.cuda.mem_get_info", lambda device=None: (int(71.3 * 1024**3), int(128 * 1024**3)))

        mock_vm = MagicMock()
        mock_vm.available = int(74.8 * 1024**3)
        mock_vm.used = int(52.9 * 1024**3)
        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_vm)

        result = mm.memory_report()
        assert result == "alloc 45.2G res 46.1G free 71.3G | sys avail 74.8G used 52.9G"

    def test_format_without_cuda(self, monkeypatch):
        """When CUDA unavailable, format should be 'sys avail X.XG used X.XG'."""
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)

        mock_vm = MagicMock()
        mock_vm.available = int(118.2 * 1024**3)
        mock_vm.used = int(9.3 * 1024**3)
        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_vm)

        result = mm.memory_report()
        assert result == "sys avail 118.2G used 9.3G"

    def test_memory_delta_format(self):
        """Delta format should be 'delta-torch +X.XG delta-avail +X.XG'."""
        before = "alloc 10.0G res 11.0G free 100.0G | sys avail 105.0G used 20.0G"
        after = "alloc 34.1G res 35.0G free 76.0G | sys avail 80.9G used 44.1G"
        result = mm.memory_delta(before, after)
        assert result == "\u0394torch +24.1G \u0394avail -24.1G"

    def test_memory_delta_malformed_input(self):
        result = mm.memory_delta("garbage", "also garbage")
        assert result == "\u0394torch ?G \u0394avail ?G"
