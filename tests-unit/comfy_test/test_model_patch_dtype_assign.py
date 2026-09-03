"""Mixed-dtype assign=True normalization for the shared MODEL_PATCH loader.

`comfy_extras/nodes_model_patch.py::ModelPatchLoader.load_model_patch` funnels every
model-patch family through one `model.load_state_dict(sd, assign=...)`, and the fork
substitutes `should_assign_weights()` there -- so on unified memory it is an
assign=True site with the usual dtype-preservation hazard.

Every branch of that dispatcher historically hardcoded `comfy.ops.manual_cast`
(runtime cast -> crash-safe regardless of checkpoint dtypes). Upstream `d3eaf6ad`
("Minimax h3 controlnet as a model patch instead of a controlnet") added the first
branch that resolves ops via `pick_operations(dtype, manual_cast_dtype)`; on GB10 a
bf16 checkpoint gives `manual_cast_dtype is None`, so `pick_operations` returns
`disable_weight_init` -- NO runtime cast. A mixed-dtype checkpoint there reproduces
the Krea 2 turbo failure (fp32 weight vs bf16 activation RuntimeError in F.linear).

These tests pin the same contract the UNET / encoder sites carry:
  - assign=True + unified + non-quantized -> mismatched tensors pre-cast to param dtype
  - quantized ops (mixed_precision_ops) -> pass through untouched
  - non-unified -> pass through untouched
"""

import torch

import comfy.model_management  # noqa: F401 -- namespace pkg; import before monkeypatch
import comfy.memory_management  # noqa: F401 -- namespace pkg; import before monkeypatch
import comfy.ops


class _FakePatchModel(torch.nn.Module):
    """Minimal stand-in for a MiniMaxH3FunControl built with disable_weight_init."""

    def __init__(self, ops=comfy.ops.disable_weight_init):
        super().__init__()
        self.control_proj_in = ops.Linear(
            8, 4, bias=True, dtype=torch.bfloat16, device="cpu"
        )

    def forward(self, x):
        return self.control_proj_in(x)


def _fp32_sd():
    return {
        "control_proj_in.weight": torch.randn(4, 8, dtype=torch.float32),
        "control_proj_in.bias": torch.randn(4, dtype=torch.float32),
    }


def _normalize(model, sd):
    """Call exactly what the loader call site is expected to call."""
    return comfy.model_management.normalize_assign_state_dict_dtypes(
        model, sd, log_tag="MODEL_PATCH_DTYPE_NORMALIZE"
    )


def test_model_patch_loader_normalizes_mixed_dtypes(monkeypatch):
    """The loader must pre-cast before assign=True, and forward() must survive."""
    monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", True)
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", False)

    import comfy_extras.nodes_model_patch as nmp

    model = _FakePatchModel()
    sd = _fp32_sd()
    expected_w = sd["control_proj_in.weight"].clone().to(torch.bfloat16)

    nmp.normalize_model_patch_state_dict(model, sd, assign=True, quantized=False)
    model.load_state_dict(sd, assign=True)

    assert model.control_proj_in.weight.dtype == torch.bfloat16
    assert model.control_proj_in.bias.dtype == torch.bfloat16
    assert torch.equal(model.control_proj_in.weight, expected_w)
    # disable_weight_init does not cast at runtime -- this is the crash the fix prevents.
    out = model(torch.randn(2, 8, dtype=torch.bfloat16))
    assert out.dtype == torch.bfloat16


def test_model_patch_loader_skips_when_quantized(monkeypatch):
    """Quantized ops own their dequant path -- never touch their tensors."""
    monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", True)
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", False)

    import comfy_extras.nodes_model_patch as nmp

    model = _FakePatchModel()
    sd = _fp32_sd()
    nmp.normalize_model_patch_state_dict(model, sd, assign=True, quantized=True)
    assert sd["control_proj_in.weight"].dtype == torch.float32


def test_model_patch_loader_skips_when_not_assigning(monkeypatch):
    """assign=False keeps upstream copy semantics -- nothing to pre-cast."""
    monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", True)
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", False)

    import comfy_extras.nodes_model_patch as nmp

    model = _FakePatchModel()
    sd = _fp32_sd()
    nmp.normalize_model_patch_state_dict(model, sd, assign=False, quantized=False)
    assert sd["control_proj_in.weight"].dtype == torch.float32


def test_model_patch_loader_inert_off_unified(monkeypatch):
    """Off unified memory the helper is a no-op (gate lives inside the helper)."""
    monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", False)
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", False)

    import comfy_extras.nodes_model_patch as nmp

    model = _FakePatchModel()
    sd = _fp32_sd()
    nmp.normalize_model_patch_state_dict(model, sd, assign=True, quantized=False)
    assert sd["control_proj_in.weight"].dtype == torch.float32
