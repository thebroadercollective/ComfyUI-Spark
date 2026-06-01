"""Regression tests for RAMPressureCache.ram_release evicting the resident model.

Bug: on DGX Spark unified memory, the post-node RAM-pressure handler escalates to
ram_release(free_active=True) (execution.py) because free_pins() is a no-op with
dynamic VRAM disabled and psutil.available underreports free memory (freed weights
linger in torch's CUDA reserved cache). free_active=True then evicted the *current*
generation's cached CheckpointLoader output -- the ModelPatcher -- dropping the last
reference to the resident diffusion model and freeing tens of GB mid-workflow, forcing
a multi-minute reload on the next generation.

The 60GB-aware 1e30 score in scan_list_for_ram_usage only tags OLD-generation
ModelPatchers, so the active patcher scored as a tiny default entry and got swept up.

Fix: on unified memory, ram_release must never evict a current-generation entry that
holds a ModelPatcher, while still evicting CPU-tensor outputs (the real relief valve).
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import comfy.model_management  # noqa: F401 (namespace pkg; import before monkeypatch)
import comfy_execution.caching as caching
from comfy.model_patcher import ModelPatcher


class _FakeCacheEntry:
    """Mirrors the real CacheEntry: only .outputs is read by ram_release."""

    def __init__(self, outputs):
        self.outputs = outputs


def _make_cache(generation=5):
    # Bypass __init__/set_prompt: populate the internal dicts directly.
    cache = caching.RAMPressureCache.__new__(caching.RAMPressureCache)
    cache.cache = {}
    cache.used_generation = {}
    cache.timestamps = {}
    cache.children = {}
    cache.generation = generation
    return cache


def _add_entry(cache, key, outputs, generation, ts):
    cache.cache[key] = _FakeCacheEntry(outputs)
    cache.used_generation[key] = generation
    cache.timestamps[key] = ts


@pytest.fixture
def model_patcher_mock():
    # MagicMock(spec=ModelPatcher) sets __class__, so isinstance(...) passes.
    return MagicMock(spec=ModelPatcher)


def _force_unrelieved_pressure(monkeypatch):
    # Simulate the unified false-pressure signal: psutil.available never rises even as
    # entries are evicted (freed weights sit in torch's CUDA reserved cache). This is
    # what drove the eviction loop past the CPU tensors and into the resident model.
    monkeypatch.setattr(
        caching.psutil, "virtual_memory", lambda: SimpleNamespace(available=0)
    )


def test_ram_release_protects_current_gen_model_on_unified(monkeypatch, model_patcher_mock):
    monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", True)
    _force_unrelieved_pressure(monkeypatch)

    cache = _make_cache(generation=5)
    # current-gen entry holding the resident diffusion model (CheckpointLoader output)
    _add_entry(cache, "model", [[model_patcher_mock]], generation=5, ts=200.0)
    # current-gen entry holding a CPU-tensor output (latent/image) -- legit relief target
    _add_entry(cache, "tensor", [[torch.zeros(1024, 1024)]], generation=5, ts=100.0)

    cache.ram_release(target=10 ** 9, free_active=True)

    assert "model" in cache.cache, "resident model must NOT be evicted on unified memory"
    assert "tensor" not in cache.cache, "CPU-tensor relief valve must still evict"


def test_ram_release_evicts_model_when_not_unified(monkeypatch, model_patcher_mock):
    # Non-unified hardware keeps upstream behavior: free_active may evict current-gen
    # models as a last resort. The unified guard must not change that path.
    monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", False)
    _force_unrelieved_pressure(monkeypatch)

    cache = _make_cache(generation=5)
    _add_entry(cache, "model", [[model_patcher_mock]], generation=5, ts=200.0)
    _add_entry(cache, "tensor", [[torch.zeros(1024, 1024)]], generation=5, ts=100.0)

    cache.ram_release(target=10 ** 9, free_active=True)

    assert "model" not in cache.cache, "non-unified preserves upstream free_active eviction"


def test_ram_release_skips_old_gen_unaffected_by_guard(monkeypatch, model_patcher_mock):
    # The guard only protects CURRENT-gen patchers. An OLD-gen patcher (stale model from a
    # previous workflow) must remain evictable on unified -- that is the correct relief.
    monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", True)
    _force_unrelieved_pressure(monkeypatch)

    cache = _make_cache(generation=5)
    _add_entry(cache, "old_model", [[model_patcher_mock]], generation=3, ts=50.0)

    cache.ram_release(target=10 ** 9, free_active=True)

    assert "old_model" not in cache.cache, "stale old-gen model must stay evictable"
