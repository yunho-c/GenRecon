import os
os.environ['ATTN_BACKEND'] = 'sdpa'
os.environ['SPARSE_CONV_BACKEND'] = 'none'

import pytest
import torch
from torch import nn

from experiments.tenstorrent.inference_precision import prepare_float32_device
from experiments.tenstorrent.reference import float32_reference
from genrecon.modules.attention import MultiHeadAttention, RotaryPositionEmbedder
from genrecon.modules.lora import apply_lora_to_attention
from genrecon.modules.utils import convert_module_to


def test_matches_original_fp32_reference_preserving_lora_aliases_and_complex_buffers():
    torch.set_num_threads(4)
    torch.manual_seed(58)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.dtype = torch.bfloat16
            self.attention = MultiHeadAttention(48, 4, use_rope=True, qk_rms_norm=True)
            apply_lora_to_attention(self.attention, 4, 8)
            for name, value in self.attention.named_parameters():
                if 'lora_B' in name:
                    nn.init.normal_(value, std=.05)
            self.apply(lambda module: convert_module_to(module, self.dtype))
            self.register_buffer('phases', RotaryPositionEmbedder(12, 3)(torch.randint(0, 32, (17, 3))))
            self.register_buffer('indices', torch.arange(17))
            self.register_buffer('real_buffer', torch.randn(3).bfloat16())
            self.shared_weight = self.attention.to_qkv.weight

        def forward(self, x):
            return self.attention(x.to(self.dtype), phases=self.phases)

    model = Model().eval()
    ids = {id(p) for p in model.parameters()}
    original = {name: p.float().clone() for name, p in model.named_parameters()}
    phases, indices = model.phases.clone(), model.indices.clone()
    closure = model.attention.to_qkv.forward
    x = torch.randn(2, 17, 48)
    with torch.no_grad(), float32_reference(model):
        expected = model(x)
    result = prepare_float32_device(model)
    with torch.no_grad():
        actual = model(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert ids == {id(p) for p in model.parameters()}
    assert model.shared_weight is model.attention.to_qkv.weight
    assert model.attention.to_qkv.forward is closure
    for name, p in model.named_parameters():
        assert p.dtype == torch.float32
        torch.testing.assert_close(p, original[name], rtol=0, atol=0)
    assert model.dtype == torch.float32
    assert model.phases.dtype == torch.complex64
    assert model.indices.dtype == torch.int64
    torch.testing.assert_close(model.phases, phases, rtol=0, atol=0)
    torch.testing.assert_close(model.indices, indices, rtol=0, atol=0)
    assert model.real_buffer.dtype == torch.float32
    assert result['changed_parameters'] > 0 and result['changed_buffers'] == 1
    assert result['parameter_bytes_after'] > result['parameter_bytes_before']


def test_training_and_non_cpu_preparation_rejected_without_mutation():
    model = nn.Linear(4, 4).bfloat16()
    original = model.weight.clone()
    with pytest.raises(ValueError, match='eval'):
        prepare_float32_device(model)
    torch.testing.assert_close(model.weight, original, rtol=0, atol=0)
    model = nn.Linear(4, 4, device='meta').eval()
    with pytest.raises(ValueError, match='CPU'):
        prepare_float32_device(model)
    assert model.weight.device.type == 'meta'
