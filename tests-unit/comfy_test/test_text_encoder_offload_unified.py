"""text_encoder_offload_device() on unified memory.

On unified memory, "offloading" a CUDA-resident text encoder to cpu duplicates
its weights inside the same physical pool (observed: sampler-time free_memory
copying the 33G mistral TE to cpu, driving swap/OOM). The offload device must
follow the load device — same pattern as unet_offload_device()/vae_offload_device().
Explicit-CPU paths (--cpu-text-enc, node device="cpu") must keep cpu.
"""
import pytest
import torch

import comfy.model_management as mm
from comfy.cli_args import args

CUDA = torch.device("cuda", 0)
CPU = torch.device("cpu")


@pytest.fixture
def gpu_env(monkeypatch):
    monkeypatch.setattr(mm, "get_torch_device", lambda: CUDA)
    monkeypatch.setattr(args, "gpu_only", False)
    monkeypatch.setattr(args, "cpu_text_enc", False)


def test_unified_gpu_te_offloads_to_gpu(monkeypatch, gpu_env):
    monkeypatch.setattr(mm, "UNIFIED_MEMORY", True)
    monkeypatch.setattr(mm, "text_encoder_device", lambda: CUDA)
    assert mm.text_encoder_offload_device() == CUDA


def test_unified_cpu_te_keeps_cpu_offload(monkeypatch, gpu_env):
    # --cpu-text-enc / lowvram-style cpu placement: load device is cpu, so the
    # offload device must stay cpu (a cuda offload target would be nonsense).
    monkeypatch.setattr(mm, "UNIFIED_MEMORY", True)
    monkeypatch.setattr(mm, "text_encoder_device", lambda: CPU)
    assert mm.text_encoder_offload_device() == CPU


def test_unified_cpu_text_enc_flag_keeps_cpu_offload(monkeypatch, gpu_env):
    # Same as above but through the real text_encoder_device() chain: with
    # --cpu-text-enc it resolves cpu before any vram-state logic runs.
    monkeypatch.setattr(mm, "UNIFIED_MEMORY", True)
    monkeypatch.setattr(args, "cpu_text_enc", True)
    assert mm.text_encoder_offload_device() == CPU


def test_non_unified_keeps_cpu_offload(monkeypatch, gpu_env):
    monkeypatch.setattr(mm, "UNIFIED_MEMORY", False)
    monkeypatch.setattr(mm, "text_encoder_device", lambda: CUDA)
    assert mm.text_encoder_offload_device() == CPU


def test_gpu_only_still_wins(monkeypatch, gpu_env):
    monkeypatch.setattr(mm, "UNIFIED_MEMORY", False)
    monkeypatch.setattr(args, "gpu_only", True)
    assert mm.text_encoder_offload_device() == CUDA
