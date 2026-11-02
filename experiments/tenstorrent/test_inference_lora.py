"""Validate nonzero learned LoRA candidates against the original attention."""
import copy
import os

os.environ["ATTN_BACKEND"] = "sdpa"

import pytest
import torch
from genrecon.modules.attention import MultiHeadAttention
from genrecon.modules.lora import apply_lora_to_attention
from experiments.tenstorrent.inference_lora import prepare_lora


@pytest.mark.parametrize("kind", ["self", "cross"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mode", ["merged", "lowrank", "frozen_delta"])
def test_learned_attention(kind, dtype, mode):
    torch.manual_seed(19)
    model = MultiHeadAttention(48, 4, ctx_channels=24, type=kind).to(dtype).eval()
    apply_lora_to_attention(model, 4, 8)
    # Injection creates FP32 adapters; match the real pipeline's conversion.
    model.to(dtype)
    with torch.no_grad():
        for name, value in model.named_parameters():
            if "lora_B" in name:
                value.normal_(std=.05)
        x = torch.randn(2, 17, 48, dtype=dtype)
        context = torch.randn(2, 9, 24, dtype=dtype) if kind == "cross" else None
        expected = model(x, context)
        candidate = copy.deepcopy(model)
        assert prepare_lora(candidate, mode) == (2 if kind == "self" else 3)
        actual = candidate(x, context)
        assert torch.isfinite(actual).all()
        error = (actual.float() - expected.float()).norm() / expected.float().norm()
        assert error < (0.02 if dtype == torch.bfloat16 else 2e-6)
        if mode == "frozen_delta":
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # The reference is still stock, and preparation did not mutate it.
        torch.testing.assert_close(model(x, context), expected, rtol=0, atol=0)
        candidate.train()
        with pytest.raises(RuntimeError, match="inference-only"):
            candidate(x, context)


def test_reject_training_or_absent_lora():
    model = torch.nn.Linear(8, 8)
    with pytest.raises(ValueError, match="eval"):
        prepare_lora(model, "merged")
    model.eval()
    with pytest.raises(ValueError, match="No stock LoRA"):
        prepare_lora(model, "merged")
