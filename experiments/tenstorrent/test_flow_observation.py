import os
os.environ.update(ATTN_BACKEND='sdpa',SPARSE_CONV_BACKEND='none')

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from genrecon.modules.attention import RotaryPositionEmbedder
from genrecon.modules.transformer import ModulatedTransformerCrossBlock
from genrecon.modules.lora import apply_lora_to_attention
from experiments.tenstorrent.flow_observation import ShapeFlowObservation
from experiments.tenstorrent.inference_rope import prepare_real_rope
from experiments.tenstorrent.inference_attention import prepare_attention


class TinyFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.dtype = torch.float32
        self.input_layer = nn.Linear(8,48)
        self.out_layer = nn.Linear(48,8)
        self.t_embedder = nn.Sequential(nn.Unflatten(0,(1,1)),nn.Linear(1,48))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(),nn.Linear(48,288))
        self.blocks = nn.ModuleList([ModulatedTransformerCrossBlock(48,24,4,
            use_rope=True,share_mod=True,qk_rms_norm=True,qk_rms_norm_cross=True) for _ in range(2)])
        self.proj_linears = nn.ModuleList([nn.Linear(24,48) for _ in self.blocks])
        for block in self.blocks:
            for attention in (block.self_attn,block.cross_attn):
                apply_lora_to_attention(attention,4,8)
                for name,param in attention.named_parameters():
                    if 'lora_B' in name:
                        nn.init.normal_(param,std=.05)
        self.register_buffer('phases',RotaryPositionEmbedder(12,3)(torch.randint(0,32,(17,3))))

    def forward(self,x,t,context,projected):
        h = self.input_layer(x).unsqueeze(0).to(self.dtype)
        mod = self.adaLN_modulation(self.t_embedder(t)).to(self.dtype)
        for block,projection in zip(self.blocks,self.proj_linears):
            h = h + projection(projected.unsqueeze(0))
            h = block(h,mod,context,self.phases)
        return self.out_layer(F.layer_norm(h,h.shape[-1:]))[0]


@pytest.mark.parametrize('first_block',[False,True])
@pytest.mark.parametrize('prepared',[False,True])
def test_capture_preserves_complete_forward_and_learned_values(first_block,prepared):
    torch.set_num_threads(4)
    torch.manual_seed(59)
    flow = TinyFlow().eval()
    if prepared:
        prepare_real_rope(flow,head_first=True,flatten=True)
        prepare_attention(flow,'fp32_reciprocal')
    identities = {name:id(value) for name,value in flow.named_parameters()}
    values = {name:value.clone() for name,value in flow.named_parameters()}
    inputs = (torch.randn(17,8),torch.tensor([.6]),torch.randn(1,9,24),torch.randn(17,24))
    capture = ShapeFlowObservation(flow,first_block=first_block).eval()
    with torch.no_grad():
        actual = dict(zip(capture.names,capture(*inputs),strict=True))
        torch.testing.assert_close(actual['final'],flow(*inputs),rtol=0,atol=0)
        if first_block:
            shift,scale,gate,shift_mlp,scale_mlp,gate_mlp = (
                flow.blocks[0].modulation+actual['modulation']).chunk(6,dim=1)
            expected = actual['first_after_cross']+actual['first_mlp_output']*gate_mlp.unsqueeze(1)
            torch.testing.assert_close(actual['block_00'],expected,rtol=0,atol=0)
            torch.testing.assert_close(actual['first_norm1'],flow.blocks[0].norm1(actual['first_input']),rtol=0,atol=0)
    assert identities == {name:id(value) for name,value in flow.named_parameters()}
    for name,value in flow.named_parameters():
        torch.testing.assert_close(value,values[name],rtol=0,atol=0)


def test_unsupported_modulation_is_rejected():
    flow = TinyFlow().eval()
    flow.blocks[0].share_mod = False
    with pytest.raises(ValueError,match='shared-modulation'):
        ShapeFlowObservation(flow,first_block=True)
