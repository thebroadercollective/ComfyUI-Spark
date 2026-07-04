"""Failing tests (TDD RED) for text-encoder assign=True dtype normalization.

Bug: on DGX Spark unified memory, SDClipModel.load_sd (comfy/sd1_clip.py) calls
self.transformer.load_state_dict(sd, assign=can_assign_sd). assign=True PRESERVES
checkpoint tensor dtypes instead of casting them into the pre-allocated
(model-dtype) parameter buffers the way assign=False / upstream's copy semantics
do. For text encoders this silently defeats --bf16-text-enc: an fp32 checkpoint
loaded onto a bf16-constructed transformer stays fp32, doubling TE memory (this
is the same class of bug already fixed for the diffusion-model path in
comfy/model_base.py::load_model_weights -- see
tests-unit/comfy_test/test_mixed_dtype_assign_load.py for the reference pattern).

This task (Task 1) is RED-only. It encodes the contract of a shared helper that
DOES NOT EXIST YET (Task 2 adds it) plus a call-site wiring that DOES NOT EXIST
YET (Task 3 adds it):

  comfy.model_management.normalize_assign_state_dict_dtypes(
      module, state_dict, log_tag="DTYPE_NORMALIZE") -> int

  - Pre-casts state-dict tensors whose dtype mismatches the matching named
    param/buffer of `module` to that param/buffer's dtype (reproduces
    assign=False copy semantics).
  - Returns 0 immediately (no mutation) when comfy.model_management.UNIFIED_MEMORY
    is False OR comfy.memory_management.aimdo_enabled is True.
  - Allowlist BOTH sides: (fp64, fp32, fp16, bf16). Non-tensor entries, keys with
    no matching named param/buffer, and dtypes outside the allowlist pass through
    untouched.
  - Mutates state_dict in place (value replacement only); returns count of
    tensors cast.
  - When count > 0, logs at INFO: "%s %d mixed-dtype tensors cast to model dtype
    (assign=True)" % (log_tag, normalized).

Call-site contract (Task 3): SDClipModel.load_sd (comfy/sd1_clip.py, ~line 309)
will call the helper on self.transformer before load_state_dict when
can_assign_sd is truthy AND self.operations is comfy.ops.manual_cast (identity,
not issubclass).

RED mechanics: the helper is referenced via direct attribute access INSIDE each
test body (not at module import level), so a missing symbol produces a clean
per-test AttributeError failure rather than a collection error for the whole
file.

Expected RED split at this commit (recorded in .superpowers/sdd/task-1-report.md):
  - Tests 1-6 (helper contract) FAIL: AttributeError, the helper does not exist.
  - Test 7 (SDClipModel integration, happy path) FAILS: assign=True currently
    preserves the fp32 checkpoint dtype, so transformer params come out fp32,
    not bf16.
  - Test 8 (SDClipModel gate) currently PASSES trivially: it pins CURRENT
    behavior (fp32 survives) which must also hold once the gate is wired up,
    since load_sd has no normalization call at all yet.
"""

import logging

import torch

import comfy.model_management  # noqa: F401 -- namespace pkg; import before monkeypatch
import comfy.memory_management  # noqa: F401 -- namespace pkg; import before monkeypatch
import comfy.ops
from comfy.sd1_clip import SDClipModel


# ---------------------------------------------------------------------------
# Helper unit tests (1-6): comfy.model_management.normalize_assign_state_dict_dtypes
# ---------------------------------------------------------------------------


class _LinearModule(torch.nn.Module):
    """A single comfy.ops.manual_cast.Linear layer -- mirrors a real TE building block."""

    def __init__(self, dtype=torch.bfloat16):
        super().__init__()
        self.fc = comfy.ops.manual_cast.Linear(4, 4, bias=True, dtype=dtype, device="cpu")


def test_downcast_fp32_sd_tensor_to_bf16_param(monkeypatch):
    """Test 1 (downcast direction): fp32 sd tensor -> bf16 param slot comes out bf16
    with values equal to the fp32 source downcast."""
    monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", True)
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", False)

    module = _LinearModule(dtype=torch.bfloat16)
    fp32_weight = torch.randn(4, 4, dtype=torch.float32)
    sd = {"fc.weight": fp32_weight.clone()}
    expected = fp32_weight.to(torch.bfloat16)

    count = comfy.model_management.normalize_assign_state_dict_dtypes(module, sd)

    assert count == 1
    assert sd["fc.weight"].dtype == torch.bfloat16
    assert torch.equal(sd["fc.weight"], expected)


def test_upcast_fp16_sd_tensor_to_fp32_param(monkeypatch):
    """Test 1 (upcast direction): fp16 sd tensor -> fp32 param slot comes out fp32.
    Upstream copy semantics are bidirectional -- this is not just a downcast helper."""
    monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", True)
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", False)

    class _Fp32Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(4, dtype=torch.float32), requires_grad=False)

    module = _Fp32Module()
    fp16_val = torch.randn(4, dtype=torch.float16)
    sd = {"p": fp16_val.clone()}
    expected = fp16_val.to(torch.float32)

    count = comfy.model_management.normalize_assign_state_dict_dtypes(module, sd)

    assert count == 1
    assert sd["p"].dtype == torch.float32
    assert torch.equal(sd["p"], expected)


def test_buffer_mismatch_is_cast(monkeypatch):
    """Test 2: a registered float buffer with mismatched sd dtype gets cast too
    (buffers are named_buffers, not just named_parameters)."""
    monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", True)
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", False)

    class _BufferModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("running_stat", torch.zeros(4, dtype=torch.bfloat16))

    module = _BufferModule()
    fp32_val = torch.randn(4, dtype=torch.float32)
    sd = {"running_stat": fp32_val.clone()}
    expected = fp32_val.to(torch.bfloat16)

    count = comfy.model_management.normalize_assign_state_dict_dtypes(module, sd)

    assert count == 1
    assert sd["running_stat"].dtype == torch.bfloat16
    assert torch.equal(sd["running_stat"], expected)


def test_allowlist_passthrough_int_sd_tensor(monkeypatch):
    """Test 3a: an int64 sd tensor is outside the allowlist and must pass through
    untouched even though the matching param dtype (bf16) IS in the allowlist."""
    monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", True)
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", False)

    class _BfModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16), requires_grad=False)

    module = _BfModule()
    original = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    sd = {"p": original.clone()}

    count = comfy.model_management.normalize_assign_state_dict_dtypes(module, sd)

    assert count == 0
    assert sd["p"].dtype == torch.int64
    assert torch.equal(sd["p"], original)


def test_allowlist_passthrough_param_dtype_outside_allowlist(monkeypatch):
    """Test 3b: the matching param's dtype (float8_e4m3fn) is outside the allowlist,
    so an in-allowlist fp32 sd tensor must still pass through untouched (both
    sides must be in the allowlist)."""
    monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", True)
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", False)

    class _Fp8Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # torch.float8_e4m3fn is constructible on CPU on this torch version.
            self.p = torch.nn.Parameter(
                torch.zeros(4, dtype=torch.float8_e4m3fn), requires_grad=False
            )

    module = _Fp8Module()
    original = torch.randn(4, dtype=torch.float32)
    sd = {"p": original.clone()}

    count = comfy.model_management.normalize_assign_state_dict_dtypes(module, sd)

    assert count == 0
    assert sd["p"].dtype == torch.float32
    assert torch.equal(sd["p"], original)


def test_allowlist_passthrough_non_tensor_entry(monkeypatch):
    """Test 3c: a non-tensor sd entry (e.g. metadata string) passes through
    untouched even when its key matches a real param name."""
    monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", True)
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", False)

    module = _LinearModule(dtype=torch.bfloat16)
    sd = {"fc.weight": "not-a-tensor"}

    count = comfy.model_management.normalize_assign_state_dict_dtypes(module, sd)

    assert count == 0
    assert sd["fc.weight"] == "not-a-tensor"


def test_allowlist_passthrough_no_matching_param(monkeypatch):
    """Test 3d: a key with no matching named param/buffer passes through untouched."""
    monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", True)
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", False)

    module = _LinearModule(dtype=torch.bfloat16)
    original = torch.randn(4, dtype=torch.float32)
    sd = {"does.not.exist": original.clone()}

    count = comfy.model_management.normalize_assign_state_dict_dtypes(module, sd)

    assert count == 0
    assert torch.equal(sd["does.not.exist"], original)


def test_unified_memory_false_returns_zero_unmutated(monkeypatch):
    """Test 4: UNIFIED_MEMORY=False -> returns 0, sd unmutated regardless of
    otherwise-castable dtypes."""
    monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", False)
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", False)

    module = _LinearModule(dtype=torch.bfloat16)
    original = torch.randn(4, 4, dtype=torch.float32)
    sd = {"fc.weight": original.clone()}

    count = comfy.model_management.normalize_assign_state_dict_dtypes(module, sd)

    assert count == 0
    assert sd["fc.weight"].dtype == torch.float32
    assert torch.equal(sd["fc.weight"], original)


def test_unified_memory_true_aimdo_enabled_returns_zero_unmutated(monkeypatch):
    """Test 5: UNIFIED_MEMORY=True but aimdo_enabled=True -> returns 0, sd unmutated."""
    monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", True)
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", True)

    module = _LinearModule(dtype=torch.bfloat16)
    original = torch.randn(4, 4, dtype=torch.float32)
    sd = {"fc.weight": original.clone()}

    count = comfy.model_management.normalize_assign_state_dict_dtypes(module, sd)

    assert count == 0
    assert sd["fc.weight"].dtype == torch.float32
    assert torch.equal(sd["fc.weight"], original)


def test_inplace_mutation_count_and_log(monkeypatch, caplog):
    """Test 6: in-place mutation (same dict object), returned count matches the
    number of tensors cast, and the caller-supplied log_tag appears in an INFO
    log line."""
    monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", True)
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", False)

    module = _LinearModule(dtype=torch.bfloat16)
    sd = {
        "fc.weight": torch.randn(4, 4, dtype=torch.float32),
        "fc.bias": torch.randn(4, dtype=torch.float32),
    }
    sd_identity = id(sd)

    with caplog.at_level(logging.INFO):
        count = comfy.model_management.normalize_assign_state_dict_dtypes(
            module, sd, log_tag="CUSTOM_TAG_FOR_TEST"
        )

    assert count == 2
    assert id(sd) == sd_identity  # same dict object, mutated in place
    assert sd["fc.weight"].dtype == torch.bfloat16
    assert sd["fc.bias"].dtype == torch.bfloat16
    assert any("CUSTOM_TAG_FOR_TEST" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# SDClipModel integration tests (7-8)
# ---------------------------------------------------------------------------


class _StubTransformer(torch.nn.Module):
    """Minimal stand-in for comfy.clip_model.CLIPTextModel: exposes .num_layers
    and holds a couple of operations.Linear layers so load_sd has real params
    to normalize."""

    def __init__(self, config, dtype, device, operations):
        super().__init__()
        self.num_layers = 1
        self.fc1 = operations.Linear(4, 4, bias=True, dtype=dtype, device=device)
        self.fc2 = operations.Linear(4, 4, bias=True, dtype=dtype, device=device)


def _make_sdclip_model():
    # textmodel_json_config as a dict (not None) takes the isinstance(dict)
    # branch in SDClipModel.__init__ directly -- no file I/O. Default
    # layer="last" avoids the layer=="hidden" asserts. model_options left at
    # default {} => operations resolves to comfy.ops.manual_cast (no
    # custom_operations / quantization_metadata set).
    return SDClipModel(
        device="cpu",
        dtype=torch.bfloat16,
        textmodel_json_config={},
        model_class=_StubTransformer,
    )


def _fp32_transformer_sd():
    return {
        "fc1.weight": torch.randn(4, 4, dtype=torch.float32),
        "fc1.bias": torch.randn(4, dtype=torch.float32),
        "fc2.weight": torch.randn(4, 4, dtype=torch.float32),
        "fc2.bias": torch.randn(4, dtype=torch.float32),
    }


def test_sdclipmodel_load_sd_normalizes_mixed_dtype_to_bf16(monkeypatch):
    """Test 7: can_assign_sd truthy + self.operations is comfy.ops.manual_cast ->
    load_sd normalizes the fp32 checkpoint tensors to the bf16-constructed
    transformer's dtype before load_state_dict(assign=True)."""
    monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", True)
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", False)

    model = _make_sdclip_model()
    assert model.operations is comfy.ops.manual_cast  # sanity: identity, not issubclass
    model.can_assign_sd = True

    model.load_sd(_fp32_transformer_sd())

    assert model.transformer.fc1.weight.dtype == torch.bfloat16
    assert model.transformer.fc1.bias.dtype == torch.bfloat16
    assert model.transformer.fc2.weight.dtype == torch.bfloat16
    assert model.transformer.fc2.bias.dtype == torch.bfloat16


def test_sdclipmodel_load_sd_gate_skips_when_not_manual_cast(monkeypatch):
    """Test 8 (gate): with self.operations swapped to a non-manual_cast sentinel,
    the normalization must be skipped -- params come out fp32 (checkpoint dtype
    preserved by assign=True), same as current (pre-fix) behavior."""
    monkeypatch.setattr(comfy.model_management, "UNIFIED_MEMORY", True)
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", False)

    model = _make_sdclip_model()
    model.can_assign_sd = True
    model.operations = object()  # sentinel: not comfy.ops.manual_cast

    model.load_sd(_fp32_transformer_sd())

    assert model.transformer.fc1.weight.dtype == torch.float32
    assert model.transformer.fc1.bias.dtype == torch.float32
    assert model.transformer.fc2.weight.dtype == torch.float32
    assert model.transformer.fc2.bias.dtype == torch.float32
