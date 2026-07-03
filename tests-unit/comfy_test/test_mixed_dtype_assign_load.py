"""Failing tests (TDD RED) for the mixed-dtype assign=True load bug.

Bug: on DGX Spark unified memory, BaseModel.load_model_weights (comfy/model_base.py)
calls diffusion_model.load_state_dict(..., assign=True), which PRESERVES checkpoint
tensor dtypes instead of casting them into the pre-allocated (model-dtype) parameter
buffers the way assign=False / upstream's copy semantics do. A mixed-dtype checkpoint
(e.g. bf16 weights + a stray fp32 tensor -- seen on Krea2) therefore produces fp32
params inside a bf16 model built with comfy.ops.disable_weight_init ops, which do NOT
cast at runtime (comfy_cast_weights=False) -- this blows up as a dtype-mismatch
RuntimeError in F.linear at inference.

Task 2 (not this task) will add a pre-cast in load_model_weights: before
load_state_dict, checkpoint tensors whose dtype mismatches the corresponding named
param/buffer get .to(param.dtype), gated on ALL of:
  - assign is True
  - comfy.model_management.UNIFIED_MEMORY is True
  - comfy.memory_management.aimdo_enabled is False
  - self.model_config.quant_config is None
  - self.model_config.custom_operations is None
  - both dtypes are in the allowlist {fp64, fp32, fp16, bf16}

These tests encode that contract. Expected RED state (before Task 2's fix):
  - Test 1 (repro) and Test 2 (parity) FAIL against current code.
  - Test 3 (quant gate) and Test 4 (gate matrix) PASS already -- they pin CURRENT
    behavior (dtype preservation / pass-through) that must survive the fix.
"""

import torch

import comfy.model_management  # noqa: F401 -- namespace pkg; import before monkeypatch
import comfy.memory_management  # noqa: F401 -- namespace pkg; import before monkeypatch
import comfy.ops
from comfy.model_base import BaseModel
from comfy.supported_models_base import BASE


class _FakeDiffusionModel(torch.nn.Module):
    """Mirrors Krea2's minimal shape: a single disable_weight_init.Linear named `first`."""

    def __init__(self):
        super().__init__()
        self.first = comfy.ops.disable_weight_init.Linear(
            8, 4, bias=True, dtype=torch.bfloat16, device="cpu"
        )


class _FakeModelConfig(BASE):
    unet_config = {"disable_unet_model_creation": True}
    quant_config = None
    custom_operations = None


def _make_model(model_config=None):
    if model_config is None:
        model_config = _FakeModelConfig(_FakeModelConfig.unet_config)
    model = BaseModel(model_config, device="cpu")
    # disable_unet_model_creation=True means BaseModel.__init__ never sets
    # self.diffusion_model -- attach the tiny fake module ourselves.
    model.diffusion_model = _FakeDiffusionModel()
    return model


def _fp32_state_dict():
    return {
        "first.weight": torch.randn(4, 8, dtype=torch.float32),
        "first.bias": torch.randn(4, dtype=torch.float32),
    }


def test_mixed_dtype_assign_true_downcasts_and_forward_succeeds(monkeypatch):
    """Repro (RED): assign=True on unified memory must still land bf16 params."""
    monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", True)
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", False)

    model = _make_model()
    sd = _fp32_state_dict()
    expected_weight = sd["first.weight"].clone().to(torch.bfloat16)
    expected_bias = sd["first.bias"].clone().to(torch.bfloat16)

    model.load_model_weights(sd, unet_prefix="", assign=True)

    assert model.diffusion_model.first.weight.dtype == torch.bfloat16
    assert model.diffusion_model.first.bias.dtype == torch.bfloat16
    assert torch.equal(model.diffusion_model.first.weight, expected_weight)
    assert torch.equal(model.diffusion_model.first.bias, expected_bias)

    # Forward must succeed with a bf16 input -- no dtype-mismatch RuntimeError.
    x = torch.randn(2, 8, dtype=torch.bfloat16)
    out = model.diffusion_model.first(x)
    assert out.dtype == torch.bfloat16


def test_mixed_dtype_assign_true_matches_assign_false_parity(monkeypatch):
    """Parity (RED): pre-cast + assign=True must match assign=False's resulting dtypes/values."""
    monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", True)
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", False)

    model_assign_true = _make_model()
    model_assign_false = _make_model()

    sd_common = _fp32_state_dict()
    sd_true = {k: v.clone() for k, v in sd_common.items()}
    sd_false = {k: v.clone() for k, v in sd_common.items()}

    model_assign_true.load_model_weights(sd_true, unet_prefix="", assign=True)
    model_assign_false.load_model_weights(sd_false, unet_prefix="", assign=False)

    w_true = model_assign_true.diffusion_model.first.weight
    w_false = model_assign_false.diffusion_model.first.weight
    b_true = model_assign_true.diffusion_model.first.bias
    b_false = model_assign_false.diffusion_model.first.bias

    assert w_true.dtype == w_false.dtype
    assert b_true.dtype == b_false.dtype
    assert torch.equal(w_true, w_false)
    assert torch.equal(b_true, b_false)


def test_quant_config_gate_skips_precast(monkeypatch):
    """Gate (must already PASS): a quant_config model must NOT be pre-cast -- params stay fp32."""
    monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", True)
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", False)

    class _QuantModelConfig(_FakeModelConfig):
        quant_config = {"sentinel": True}

    model = _make_model(_QuantModelConfig(_QuantModelConfig.unet_config))
    sd = _fp32_state_dict()

    model.load_model_weights(sd, unet_prefix="", assign=True)

    # to_load is deleted inside load_model_weights, so the dtype surviving on the
    # params themselves is the only observable evidence of pass-through.
    assert model.diffusion_model.first.weight.dtype == torch.float32
    assert model.diffusion_model.first.bias.dtype == torch.float32


def test_gate_matrix_non_unified_skips_precast(monkeypatch):
    """Gate (must already PASS): UNIFIED_MEMORY=False -- no pre-cast, params stay fp32."""
    monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", False)
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", False)

    model = _make_model()
    sd = _fp32_state_dict()

    model.load_model_weights(sd, unet_prefix="", assign=True)

    assert model.diffusion_model.first.weight.dtype == torch.float32
    assert model.diffusion_model.first.bias.dtype == torch.float32


def test_gate_matrix_aimdo_enabled_skips_precast(monkeypatch):
    """Gate (must already PASS): UNIFIED_MEMORY=True but aimdo_enabled=True -- no pre-cast."""
    monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", True)
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", True)

    model = _make_model()
    sd = _fp32_state_dict()

    model.load_model_weights(sd, unet_prefix="", assign=True)

    assert model.diffusion_model.first.weight.dtype == torch.float32
    assert model.diffusion_model.first.bias.dtype == torch.float32


def test_gate_matrix_int_tensor_passes_through_untouched(monkeypatch):
    """Gate (must already PASS): a non-float (uint8) tensor is outside the allowlist and
    must pass through untouched even when every other gate condition is satisfied."""
    monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", True)
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", False)

    class _IntBufferDiffusionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = comfy.ops.disable_weight_init.Linear(
                8, 4, bias=True, dtype=torch.bfloat16, device="cpu"
            )
            self.register_buffer("counter", torch.zeros(4, dtype=torch.uint8))

    model_config = _FakeModelConfig(_FakeModelConfig.unet_config)
    model = BaseModel(model_config, device="cpu")
    model.diffusion_model = _IntBufferDiffusionModel()

    sd = _fp32_state_dict()
    sd["counter"] = torch.tensor([1, 2, 3, 4], dtype=torch.uint8)

    model.load_model_weights(sd, unet_prefix="", assign=True)

    assert model.diffusion_model.counter.dtype == torch.uint8
    assert torch.equal(model.diffusion_model.counter, torch.tensor([1, 2, 3, 4], dtype=torch.uint8))
