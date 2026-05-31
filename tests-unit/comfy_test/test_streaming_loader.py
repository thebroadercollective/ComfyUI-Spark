"""Unit tests for the unified-memory streaming safetensors loader.

The streaming loader (_stream_safetensors_to_cuda in comfy/utils.py) replaces
safetensors.safe_open for large unified-memory loads so that posix_fadvise(DONTNEED) can
actually evict OS page cache mid-load (safe_open holds the whole file mmap'd, which pins it).
These tests verify it is a byte-exact drop-in for safe_open and that gating/fallback behave.

The loader targets CUDA, so the round-trip tests are skipped when CUDA is unavailable.
"""

import os
import json
import struct
import time

import pytest
import torch

CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not CUDA, reason="streaming loader targets CUDA")


def _save(tmp_path, name, tensors, metadata=None):
    import safetensors.torch
    p = str(tmp_path / name)
    safetensors.torch.save_file(tensors, p, metadata=metadata or {})
    return p


def _bytes_eq(a, b):
    if a.numel() == 0:
        return b.numel() == 0 and tuple(a.shape) == tuple(b.shape)
    return torch.equal(
        a.reshape(-1).contiguous().view(torch.uint8).cpu(),
        b.reshape(-1).contiguous().view(torch.uint8).cpu(),
    )


def _safe_open_ref(path):
    import safetensors
    ref = {}
    with safetensors.safe_open(path, framework="pt", device="cuda") as f:
        keys = list(f.keys())
        for k in keys:
            ref[k] = f.get_tensor(k)
    return keys, ref


@cuda_only
class TestStreamingReconstruction:
    """The streamed sd must be byte-identical to safe_open(device='cuda')."""

    def _mixed_dtype_model(self):
        m = {
            "w_bf16": torch.randn(64, 128).to(torch.bfloat16),
            "b_f32": torch.randn(128),
            "h_f16": torch.randn(32, 16, dtype=torch.float16),
            "i_i64": torch.randint(-5, 5, (10,), dtype=torch.int64),
            "u_u8": torch.randint(0, 255, (7,), dtype=torch.uint8),
            "z_zero": torch.empty((0,), dtype=torch.float32),
            "z_zerodim": torch.empty((3, 0, 4), dtype=torch.float32),
            "scalar": torch.tensor(3.14159),
        }
        # fp8 is the kind of dtype the fork-local _TYPES omits; include it if supported.
        if hasattr(torch, "float8_e4m3fn"):
            m["f8_e4m3"] = torch.randn(16, 16).to(torch.float8_e4m3fn)
        return m

    def test_byte_identical_to_safe_open(self, tmp_path):
        import comfy.utils as U
        path = _save(tmp_path, "mixed.safetensors", self._mixed_dtype_model())
        ref_keys, ref = _safe_open_ref(path)

        sd, meta, n = U._stream_safetensors_to_cuda(
            path, "mixed.safetensors", os.path.getsize(path), time.perf_counter(), False)

        assert set(sd) == set(ref)
        assert n == len(ref)
        # key order must match safe_open exactly (safe_open yields sorted keys)
        assert list(sd.keys()) == ref_keys
        for k in ref:
            assert sd[k].dtype == ref[k].dtype, k
            assert tuple(sd[k].shape) == tuple(ref[k].shape), k
            assert sd[k].device.type == "cuda", k
            assert _bytes_eq(ref[k], sd[k]), k

    def test_metadata_handling(self, tmp_path):
        import comfy.utils as U
        path = _save(tmp_path, "meta.safetensors", {"x": torch.randn(4)}, metadata={"foo": "bar"})
        sz = os.path.getsize(path)
        _, m_none, _ = U._stream_safetensors_to_cuda(path, "m", sz, time.perf_counter(), False)
        _, m_yes, _ = U._stream_safetensors_to_cuda(path, "m", sz, time.perf_counter(), True)
        assert m_none is None
        assert m_yes == {"foo": "bar"}

    def test_end_to_end_load_torch_file_uses_stream(self, tmp_path, monkeypatch, caplog):
        """load_torch_file should select the stream path for a gated unified load and match safe_open."""
        import logging
        import comfy.utils as U
        import comfy.model_management
        import comfy.memory_management

        monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", True)
        monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", False)
        # Lower the threshold so a small test file streams.
        monkeypatch.setattr(U, "STREAMING_LOAD_THRESHOLD", 1024)

        path = _save(tmp_path, "e2e.safetensors", self._mixed_dtype_model())
        ref_keys, ref = _safe_open_ref(path)

        with caplog.at_level(logging.INFO):
            got = U.load_torch_file(path, return_metadata=False)

        assert any("method=unified-cuda-stream" in r.getMessage() for r in caplog.records)
        assert list(got.keys()) == ref_keys
        for k in ref:
            assert _bytes_eq(ref[k], got[k]), k
            assert got[k].device.type == "cuda", k


@cuda_only
class TestStreamingFallback:
    """Errors in the streaming path must fall back to safe_open, not fail the load."""

    def test_unknown_dtype_preflight_raises(self, tmp_path):
        """A dtype safetensors itself can't resolve must raise (so the caller falls back)."""
        import comfy.utils as U
        raw = os.urandom(16)
        hdr = {"bad": {"dtype": "F8_E8M0", "shape": [16], "data_offsets": [0, 16]}}
        hb = json.dumps(hdr).encode()
        p = str(tmp_path / "bad.safetensors")
        with open(p, "wb") as fh:
            fh.write(struct.pack("<Q", len(hb)) + hb + raw)
        with pytest.raises(Exception):
            U._stream_safetensors_to_cuda(p, "bad", os.path.getsize(p), time.perf_counter(), False)

    def test_load_torch_file_falls_back_on_bad_header(self, tmp_path, monkeypatch, caplog):
        """If the stream path raises, load_torch_file must fall back to safe_open and still load."""
        import logging
        import comfy.utils as U
        import comfy.model_management
        import comfy.memory_management

        monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", True)
        monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", False)
        monkeypatch.setattr(U, "STREAMING_LOAD_THRESHOLD", 1024)

        # Build a VALID safetensors file (so safe_open works), then force the stream path to fail
        # by making the header parse raise via a monkeypatched preadv that truncates.
        path = _save(tmp_path, "valid.safetensors", {"w": torch.randn(32, 32)})
        ref_keys, ref = _safe_open_ref(path)

        real_pread = os.pread

        def broken_pread(fd, n, off):
            # Corrupt the very first header-length read to trigger a stream-path failure.
            if off == 0 and n == 8:
                return b"\x00\x00"  # too short -> struct.unpack raises
            return real_pread(fd, n, off)

        monkeypatch.setattr(os, "pread", broken_pread)

        with caplog.at_level(logging.WARNING):
            got = U.load_torch_file(path, return_metadata=False)

        assert any("falling back to safe_open" in r.getMessage() for r in caplog.records)
        assert list(got.keys()) == ref_keys
        for k in ref:
            assert _bytes_eq(ref[k], got[k]), k


class TestStreamingGate:
    """Gating decisions don't require CUDA (no actual load performed)."""

    def test_disabled_when_streaming_dtypes_unavailable(self, tmp_path, monkeypatch):
        """When safetensors._TYPES is unavailable, the stream path is disabled (no preadv used)."""
        import comfy.utils as U
        import comfy.model_management
        import comfy.memory_management

        monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", True)
        monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", False)
        monkeypatch.setattr(U, "_SAFETENSORS_DTYPES", None)
        monkeypatch.setattr(U, "STREAMING_LOAD_THRESHOLD", 1024)

        path = _save(tmp_path, "small.safetensors", {"w": torch.randn(8)})

        called = {"preadv": False}
        if hasattr(os, "preadv"):
            real = os.preadv
            monkeypatch.setattr(os, "preadv", lambda *a, **k: called.__setitem__("preadv", True) or real(*a, **k))

        # Loads via safe_open path (CPU-safe: explicit cpu device avoids the cuda default).
        U.load_torch_file(path, device=torch.device("cpu"))
        assert called["preadv"] is False, "stream path should not run when _SAFETENSORS_DTYPES is None"
