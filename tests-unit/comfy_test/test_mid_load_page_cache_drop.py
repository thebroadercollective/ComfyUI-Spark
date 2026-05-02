"""Unit tests for mid-load page cache drop during tensor loading."""

import os
from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture
def mock_comfy_env(monkeypatch):
    """Set up mocked comfy environment for load_torch_file tests."""
    # Ensure comfy.model_management and comfy.memory_management are importable
    import comfy.model_management
    import comfy.cache_policy

    fake_size = 6 * 1024 ** 3  # 6GB — above the 5GB threshold

    mock_args = MagicMock()
    mock_args.drop_page_cache = False
    monkeypatch.setattr("comfy.utils.args", mock_args)
    monkeypatch.setattr("comfy.utils.DISABLE_MMAP", False)
    monkeypatch.setattr("comfy.utils.MMAP_TORCH_FILES", False)

    monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", True)
    monkeypatch.setattr(comfy.model_management, "memory_report",
                        MagicMock(return_value="alloc 0.0G res 0.0G free 128.0G | sys avail 128.0G used 0.0G"))
    monkeypatch.setattr(comfy.model_management, "memory_delta",
                        MagicMock(return_value="delta mock"))
    monkeypatch.setattr(comfy.model_management, "drop_file_page_cache",
                        MagicMock(return_value=True))

    # Mock aimdo_enabled
    import comfy.memory_management
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", False)

    # Mock cache_policy.maybe_drop
    monkeypatch.setattr(comfy.cache_policy, "maybe_drop", MagicMock())

    # Mock psutil
    mock_vm = MagicMock()
    mock_vm.available = 80 * 1024 ** 3
    monkeypatch.setattr("comfy.utils.psutil", MagicMock(
        virtual_memory=MagicMock(return_value=mock_vm)
    ))

    return mock_args, fake_size


def _make_mock_safe_open(keys, get_tensor_fn=None):
    """Create a mock safetensors.safe_open context manager."""
    mock_f = MagicMock()
    mock_f.keys.return_value = keys
    if get_tensor_fn:
        mock_f.get_tensor.side_effect = get_tensor_fn
    else:
        mock_f.get_tensor.return_value = MagicMock(
            element_size=MagicMock(return_value=4),
            numel=MagicMock(return_value=1),
        )
    mock_f.metadata.return_value = {}
    mock_f.__enter__ = MagicMock(return_value=mock_f)
    mock_f.__exit__ = MagicMock(return_value=False)
    return mock_f


class TestMidLoadPageCacheDrop:
    """Tests for posix_fadvise calls during tensor loading."""

    def test_fd_opened_and_closed(self, mock_comfy_env, tmp_path):
        """fd should be opened before the loop and closed after."""
        import comfy.utils

        fake_file = tmp_path / "model.safetensors"
        fake_file.write_bytes(b"fake")
        _, fake_size = mock_comfy_env

        opened_fds = []
        closed_fds = []
        real_open = os.open
        real_close = os.close

        def track_open(path, flags, *a, **kw):
            fd = real_open(path, flags, *a, **kw)
            if str(fake_file) == str(path):
                opened_fds.append(fd)
            return fd

        def track_close(fd):
            if fd in opened_fds:
                closed_fds.append(fd)
            return real_close(fd)

        mock_f = _make_mock_safe_open([])

        with patch("comfy.utils.safetensors.safe_open", return_value=mock_f), \
             patch("os.path.getsize", return_value=fake_size), \
             patch("os.open", side_effect=track_open), \
             patch("os.close", side_effect=track_close):
            comfy.utils.load_torch_file(str(fake_file))

        assert len(opened_fds) == 1, "Should open exactly one fd for page cache drop"
        assert opened_fds == closed_fds, "Every opened fd should be closed"

    def test_fadvise_called_on_tick(self, mock_comfy_env, tmp_path, caplog):
        """posix_fadvise should fire when the byte threshold triggers a tick."""
        import logging
        import comfy.utils

        fake_file = tmp_path / "model.safetensors"
        fake_file.write_bytes(b"fake")
        _, fake_size = mock_comfy_env

        # Each tensor ~2.5GB; after 2 tensors (5GB) the tick fires
        big_tensor = MagicMock()
        big_tensor.element_size.return_value = 4
        big_tensor.numel.return_value = 625_000_000  # 2.5GB

        mock_f = _make_mock_safe_open(["t1", "t2", "t3"], get_tensor_fn=lambda k: big_tensor)

        fadvise_calls = []

        def mock_fadvise(fd, offset, length, advice):
            fadvise_calls.append((fd, offset, length, advice))

        # Enable INFO logging so tick_enabled is True (required for tick to fire)
        with caplog.at_level(logging.INFO), \
             patch("comfy.utils.safetensors.safe_open", return_value=mock_f), \
             patch("os.path.getsize", return_value=fake_size), \
             patch("os.posix_fadvise", side_effect=mock_fadvise):
            comfy.utils.load_torch_file(str(fake_file))

        assert len(fadvise_calls) >= 1, "posix_fadvise should be called at least once"
        for _, offset, length, advice in fadvise_calls:
            assert offset == 0
            assert length == 0
            assert advice == os.POSIX_FADV_DONTNEED

    def test_no_drop_for_small_files(self, mock_comfy_env, tmp_path):
        """Files < 5GB should not open a drop fd."""
        import comfy.utils

        fake_file = tmp_path / "small.safetensors"
        fake_file.write_bytes(b"fake")

        small_size = 1 * 1024 ** 3  # 1GB

        fadvise_calls = []
        mock_f = _make_mock_safe_open([])

        with patch("comfy.utils.safetensors.safe_open", return_value=mock_f), \
             patch("os.path.getsize", return_value=small_size), \
             patch("os.posix_fadvise", side_effect=lambda *a: fadvise_calls.append(a)):
            comfy.utils.load_torch_file(str(fake_file))

        assert len(fadvise_calls) == 0

    def test_no_drop_when_not_unified_and_no_flag(self, mock_comfy_env, tmp_path, monkeypatch):
        """No mid-load drop when neither unified memory nor --drop-page-cache."""
        import comfy.utils
        import comfy.model_management

        mock_args, fake_size = mock_comfy_env
        monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", False)
        mock_args.drop_page_cache = False

        fake_file = tmp_path / "model.safetensors"
        fake_file.write_bytes(b"fake")

        fadvise_calls = []
        mock_f = _make_mock_safe_open([])

        with patch("comfy.utils.safetensors.safe_open", return_value=mock_f), \
             patch("os.path.getsize", return_value=fake_size), \
             patch("os.posix_fadvise", side_effect=lambda *a: fadvise_calls.append(a)):
            comfy.utils.load_torch_file(str(fake_file))

        assert len(fadvise_calls) == 0

    def test_fd_closed_on_exception(self, mock_comfy_env, tmp_path):
        """fd should be closed even if tensor loading raises."""
        import comfy.utils

        fake_file = tmp_path / "model.safetensors"
        fake_file.write_bytes(b"fake")
        _, fake_size = mock_comfy_env

        closed_fds = []
        real_close = os.close

        def track_close(fd):
            closed_fds.append(fd)
            return real_close(fd)

        mock_f = _make_mock_safe_open(["t1"])
        mock_f.get_tensor.side_effect = RuntimeError("tensor load failed")

        with patch("comfy.utils.safetensors.safe_open", return_value=mock_f), \
             patch("os.path.getsize", return_value=fake_size), \
             patch("os.close", side_effect=track_close):
            with pytest.raises(RuntimeError, match="tensor load failed"):
                comfy.utils.load_torch_file(str(fake_file))

        assert len(closed_fds) >= 1, "fd should be closed even on exception"

    def test_graceful_on_open_failure(self, mock_comfy_env, tmp_path):
        """If os.open fails for the drop fd, loading should continue."""
        import comfy.utils

        fake_file = tmp_path / "model.safetensors"
        fake_file.write_bytes(b"fake")
        _, fake_size = mock_comfy_env

        mock_f = _make_mock_safe_open([])

        def failing_open(path, flags, *a, **kw):
            raise OSError("permission denied")

        with patch("comfy.utils.safetensors.safe_open", return_value=mock_f), \
             patch("os.path.getsize", return_value=fake_size), \
             patch("os.open", side_effect=failing_open):
            # Should not raise
            comfy.utils.load_torch_file(str(fake_file))

    def test_drop_page_cache_flag_enables_on_non_unified(self, mock_comfy_env, tmp_path, monkeypatch):
        """--drop-page-cache flag should enable mid-load drops even without unified memory."""
        import comfy.utils
        import comfy.model_management

        mock_args, fake_size = mock_comfy_env
        monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", False)
        mock_args.drop_page_cache = True

        fake_file = tmp_path / "model.safetensors"
        fake_file.write_bytes(b"fake")

        opened_fds = []
        real_open = os.open

        def track_open(path, flags, *a, **kw):
            fd = real_open(path, flags, *a, **kw)
            if str(fake_file) == str(path):
                opened_fds.append(fd)
            return fd

        mock_f = _make_mock_safe_open([])

        with patch("comfy.utils.safetensors.safe_open", return_value=mock_f), \
             patch("os.path.getsize", return_value=fake_size), \
             patch("os.open", side_effect=track_open):
            comfy.utils.load_torch_file(str(fake_file))

        assert len(opened_fds) == 1, "--drop-page-cache should enable mid-load drop fd"

    def test_post_load_drop_file_page_cache_called(self, mock_comfy_env, tmp_path):
        """drop_file_page_cache should be called after loading completes."""
        import comfy.utils
        import comfy.model_management

        fake_file = tmp_path / "model.safetensors"
        fake_file.write_bytes(b"fake")
        _, fake_size = mock_comfy_env

        mock_f = _make_mock_safe_open([])

        with patch("comfy.utils.safetensors.safe_open", return_value=mock_f), \
             patch("os.path.getsize", return_value=fake_size):
            comfy.utils.load_torch_file(str(fake_file))

        comfy.model_management.drop_file_page_cache.assert_called_once_with(str(fake_file))
