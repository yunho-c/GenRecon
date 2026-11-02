"""Bound compiler liveness without changing native sparse transformer blocks."""
import torch
from torch import nn
from torch.nn import functional as F

from experiments.tenstorrent.flow_probes import NativeSparseFlowProbe


class SparseFlowGroup(NativeSparseFlowProbe):
    """A contiguous block range, retaining every token and original attention.

    Share original modules, never deepcopy them (LoRA uses closures). Reuse the
    native probe's fixed coordinates/layout/RoPE. Only the first group embeds
    latent features, only the last emits output. Timestep embedding is repeated
    per group with unchanged arithmetic.
    """
    def __init__(self, model, coords, start, stop):
        if model.pe_mode != "rope" or not model.share_mod or model.proj_linears is None:
            raise ValueError("grouped probe requires shared-modulation RoPE flows")
        if not 0 <= start < stop <= len(model.blocks):
            raise ValueError("invalid transformer block range")
        super().__init__(model, coords, precompute_rope=True)
        del self.model
        self.first = start == 0
        self.last = stop == len(model.blocks)
        self.dtype = model.dtype
        self.blocks = nn.ModuleList(model.blocks[start:stop])
        self.proj_linears = nn.ModuleList(model.proj_linears[start:stop])
        self.t_embedder = model.t_embedder
        self.adaLN_modulation = model.adaLN_modulation
        if self.first:
            self.input_layer = model.input_layer
        if self.last:
            self.out_layer = model.out_layer

    def forward(self, values, t, context, projected):
        h = self.sparse(values)
        if self.first:
            h = self.input_layer(h)
        h = h.replace(h.feats.to(self.dtype))
        mod = self.adaLN_modulation(self.t_embedder(t)).to(self.dtype)
        context, projected = context.to(self.dtype), projected.to(self.dtype)
        for block, projection in zip(self.blocks, self.proj_linears):
            h = h.replace(h.feats + projection(projected))
            h = block(h, mod, context)
        if self.last:
            # The scene sampler's latent/noise dtype is explicitly FP32.
            values = h.feats.float()
            h = self.out_layer(h.replace(F.layer_norm(values, values.shape[-1:])))
        return h.feats


def flow_groups(model, coords, size):
    if size < 1:
        raise ValueError("group size must be positive")
    return nn.ModuleList(SparseFlowGroup(model, coords, start, min(start + size, len(model.blocks)))
                         for start in range(0, len(model.blocks), size))
