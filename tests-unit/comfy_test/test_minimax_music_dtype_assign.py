"""Regression tests: MiniMax Music TE assign=True dtype normalization.

Upstream's MiniMax Music support (efd4e951, v0.33.0) added a new text-encoder
load path — MiniMaxMusic3TEModel.load_sd -> load_state_dict(assign=can_assign_sd)
(comfy/text_encoders/minimax_music.py) — which on DGX Spark unified memory is the
assign=True path (the fork sets can_assign_sd from should_assign_weights()). Like
every other encoder family's load_sd, it must pre-cast mixed checkpoint dtypes to
the constructed model dtype, or --bf16-text-enc is silently defeated and an fp32
checkpoint stays fp32 (2x TE residency + a per-op cast tax).

The wrinkle specific to this call site: load_state_dict *deletes* the unused
pruned/unpruned embedding and lm_head submodules before delegating to
nn.Module.load_state_dict. The normalization must therefore run AFTER those
deletions, so it only ever sees live parameters.
"""

import torch

import comfy.model_management  # noqa: F401 -- namespace pkg; import before monkeypatch
import comfy.memory_management  # noqa: F401 -- namespace pkg; import before monkeypatch
import comfy.ops
from comfy.text_encoders.minimax_music import MiniMaxMusic3TEModel


class _StubInner(torch.nn.Module):
    """Stand-in for MiniMaxMusic3AR's .model: carries the pruned-variant flags and
    both halves of each prunable pair, so load_state_dict's deletion branches run
    for real."""

    def __init__(self, dtype):
        super().__init__()
        self.embed_tokens = torch.nn.Linear(4, 4, bias=False, dtype=dtype)
        self.embed_tokens_prefill = torch.nn.Linear(4, 4, bias=False, dtype=dtype)
        self.embed_tokens_audio = torch.nn.Linear(4, 4, bias=False, dtype=dtype)
        self.lm_head = torch.nn.Linear(4, 4, bias=False, dtype=dtype)
        self.lm_head_pruned = torch.nn.Linear(4, 4, bias=False, dtype=dtype)
        self.pruned_embedding = None
        self.pruned_lm_head = None


class _StubMiniMaxTE(MiniMaxMusic3TEModel):
    """Subclasses the real TE so the zero-arg super() in load_state_dict binds,
    but skips MiniMaxMusic3AR's 36-layer construction."""

    def __init__(self, dtype=torch.bfloat16, operations=comfy.ops.manual_cast):
        torch.nn.Module.__init__(self)
        self.model = _StubInner(dtype)
        self.operations = operations


def _fp32_sd():
    """Checkpoint for the UNPRUNED variant (no *_prefill / *_pruned keys), plus one
    stale key naming a submodule load_state_dict deletes."""
    return {
        "model.embed_tokens.weight": torch.randn(4, 4, dtype=torch.float32),
        "model.lm_head.weight": torch.randn(4, 4, dtype=torch.float32),
        "model.embed_tokens_audio.weight": torch.randn(4, 4, dtype=torch.float32),
    }


def test_load_sd_normalizes_mixed_dtype_to_model_dtype(monkeypatch):
    monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", True)
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", False)

    model = _StubMiniMaxTE()
    assert model.operations is comfy.ops.manual_cast  # identity, not issubclass
    model.can_assign_sd = True

    model.load_sd(_fp32_sd())

    assert model.model.embed_tokens.weight.dtype == torch.bfloat16
    assert model.model.lm_head.weight.dtype == torch.bfloat16


def test_normalization_runs_after_pruned_submodule_deletion(monkeypatch):
    """The stale key for a deleted submodule must pass through untouched: it has no
    matching live param, which is only true if normalization runs post-deletion."""
    monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", True)
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", False)

    model = _StubMiniMaxTE()
    model.can_assign_sd = True
    sd = _fp32_sd()

    model.load_sd(sd)

    assert not hasattr(model.model, "embed_tokens_audio")
    assert sd["model.embed_tokens_audio.weight"].dtype == torch.float32


def test_gate_skips_when_not_manual_cast(monkeypatch):
    """mixed_precision_ops returns a *subclass* of manual_cast for quantized TEs;
    the identity check must exclude it, leaving checkpoint dtypes preserved."""
    monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", True)
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", False)

    model = _StubMiniMaxTE(operations=object())  # sentinel: not comfy.ops.manual_cast
    model.can_assign_sd = True

    model.load_sd(_fp32_sd())

    assert model.model.embed_tokens.weight.dtype == torch.float32
    assert model.model.lm_head.weight.dtype == torch.float32


def test_gate_skips_when_not_assigning(monkeypatch):
    """assign=False (non-unified / aimdo) keeps upstream's implicit copy-cast; the
    helper must not run, and load_state_dict's own copy handles the dtype."""
    monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", True)
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", False)

    model = _StubMiniMaxTE()
    model.can_assign_sd = False
    sd = _fp32_sd()

    model.load_sd(sd)

    # copied into the pre-allocated bf16 buffers, and the state dict is unmutated
    assert model.model.embed_tokens.weight.dtype == torch.bfloat16
    assert sd["model.embed_tokens.weight"].dtype == torch.float32
