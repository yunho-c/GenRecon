"""Observe fixed-sequence shape flows; native equality must be checked separately."""
import torch
from torch import nn
from torch.nn import functional as F


FIRST_BLOCK_NAMES = (
    'modulation', 'first_projected', 'first_input', 'first_norm1',
    'first_self_input', 'first_self_output', 'first_after_self',
    'first_cross_input', 'first_cross_output', 'first_after_cross',
    'first_norm3', 'first_mlp_input', 'first_mlp_output',
)


def observe_block(block, x, mod, context, phases):
    """Original shared-modulation arithmetic, retaining whole attention/MLP calls.

    Do not expose MLP preactivation here: retaining it can prevent linear/GELU
    fusion. Even these outputs require an ordinary native-forward equality check.
    """
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        (block.modulation + mod).type(mod.dtype).chunk(6, dim=1)
    )
    norm1 = block.norm1(x)
    self_input = norm1 * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
    self_output = block.self_attn(self_input, phases=phases)
    after_self = x + self_output * gate_msa.unsqueeze(1)
    cross_input = block.norm2(after_self)
    cross_output = block.cross_attn(cross_input, context)
    after_cross = after_self + cross_output
    norm3 = block.norm3(after_cross)
    mlp_input = norm3 * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
    mlp_output = block.mlp(mlp_input)
    output = after_cross + mlp_output * gate_mlp.unsqueeze(1)
    return output, (x, norm1, self_input, self_output, after_self, cross_input,
                    cross_output, after_cross, norm3, mlp_input, mlp_output)


class ShapeFlowObservation(nn.Module):
    def __init__(self, flow, first_block=False):
        super().__init__()
        if not len(flow.blocks) or any(not block.share_mod for block in flow.blocks):
            raise ValueError('Observation requires nonempty shared-modulation blocks')
        self.flow = flow
        self.first_block = first_block

    @property
    def names(self):
        return (['final', 'input_layer'] +
                [f'block_{i:02d}' for i in range(len(self.flow.blocks))] +
                (list(FIRST_BLOCK_NAMES) if self.first_block else []))

    def forward(self, x, t, context, projected):
        flow = self.flow
        h = flow.input_layer(x).unsqueeze(0).to(flow.dtype)
        states = [h]
        mod = flow.adaLN_modulation(flow.t_embedder(t)).to(flow.dtype)
        context = context.to(flow.dtype)
        projected = projected.unsqueeze(0).to(flow.dtype)
        details = ()
        for index, (block, projection) in enumerate(zip(flow.blocks, flow.proj_linears)):
            projected_block = projection(projected)
            h = h + projected_block
            if self.first_block and index == 0:
                h, values = observe_block(block, h, mod, context, flow.phases)
                details = (mod, projected_block, *values)
            else:
                h = block(h, mod, context, flow.phases)
            states.append(h)
        output = flow.out_layer(F.layer_norm(h.to(x.dtype), h.shape[-1:]))[0]
        return (output, *states, *details)
