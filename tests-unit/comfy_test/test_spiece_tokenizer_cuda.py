"""Regression test for SPieceTokenizer.__init__ on a CUDA tensor.

Bug: comfy/text_encoders/spiece_tokenizer.py:15 called bare
`tokenizer_path.numpy()` on the sentencepiece vocab tensor. On DGX Spark
unified memory, load_clip loads the whole text-encoder state dict to CUDA
(text_encoder_device()) without --cpu-text-enc, so the checkpoint-embedded
tokenizer_data["spiece_model"] byte tensor arrives at SPieceTokenizer.__init__
as a CUDA tensor and bare .numpy() raises
`TypeError: can't convert cuda:0 device type tensor to numpy` (hit loading
Wan 2.2's umt5_xxl_fp8_e4m3fn_scaled.safetensors on a bare GB10 launch).

Fix (matches the established fork pattern at comfy/text_encoders/flux.py:77
and comfy/ops.py:1062,1446): tokenizer_path.cpu().numpy().tobytes(). .cpu() is
a no-op copy for CPU tensors and essentially free on unified memory.

sentencepiece is imported lazily inside SPieceTokenizer.__init__, so the
monkeypatch must target the sentencepiece module attribute directly (import
sentencepiece here first, then monkeypatch sentencepiece.SentencePieceProcessor
-- patching comfy.text_encoders.spiece_tokenizer.sentencepiece would not exist
as a module-level name to patch).
"""

import pytest
import sentencepiece
import torch

from comfy.text_encoders.spiece_tokenizer import SPieceTokenizer

CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not CUDA, reason="pins the CUDA-tensor regression")


class _FakeSentencePieceProcessor:
    """Captures the model_proto bytes it was constructed with instead of
    actually parsing a sentencepiece model."""

    last_model_proto = None

    def __init__(self, model_proto=None, model_file=None, add_bos=False, add_eos=True):
        _FakeSentencePieceProcessor.last_model_proto = model_proto


def _patch_sentencepiece(monkeypatch):
    _FakeSentencePieceProcessor.last_model_proto = None
    monkeypatch.setattr(sentencepiece, "SentencePieceProcessor", _FakeSentencePieceProcessor)


def test_spiece_tokenizer_init_cpu_tensor(monkeypatch):
    """CPU tensor: bare .numpy() already works; pins the passthrough contract
    that the fake receives the exact source bytes (unaffected by the fix)."""
    _patch_sentencepiece(monkeypatch)
    data = bytes([1, 2, 3, 4, 5, 250])
    tensor = torch.ByteTensor(list(data))

    SPieceTokenizer(tensor)

    assert _FakeSentencePieceProcessor.last_model_proto == data


@cuda_only
def test_spiece_tokenizer_init_cuda_tensor(monkeypatch):
    """CUDA tensor: this is the case that raises TypeError before the fix
    (bare .numpy() on a CUDA tensor). After the fix (.cpu().numpy()), the
    constructor succeeds and the fake receives identical bytes."""
    _patch_sentencepiece(monkeypatch)
    data = bytes([10, 20, 30, 40, 50, 251])
    tensor = torch.ByteTensor(list(data)).cuda()

    SPieceTokenizer(tensor)

    assert _FakeSentencePieceProcessor.last_model_proto == data
