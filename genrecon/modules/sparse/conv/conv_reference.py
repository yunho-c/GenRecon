"""Portable inference reference using the checkpoint's FlexGEMM weight layout."""

import math
import torch
from torch import nn


def sparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1,
                       dilation=1, padding=None, bias=True, indice_key=None):
    self.kernel_size = tuple(kernel_size) if isinstance(kernel_size, (tuple, list)) else (kernel_size,) * 3
    self.stride = tuple(stride) if isinstance(stride, (tuple, list)) else (stride,) * 3
    self.dilation = tuple(dilation) if isinstance(dilation, (tuple, list)) else (dilation,) * 3
    if self.stride != (1, 1, 1) or padding is not None:
        raise ValueError("reference backend implements submanifold convolution only")
    self.in_channels, self.out_channels = in_channels, out_channels
    weight = torch.empty(out_channels, in_channels, *self.kernel_size)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    self.weight = nn.Parameter(weight.permute(0, 2, 3, 4, 1).contiguous())
    if bias:
        self.bias = nn.Parameter(torch.empty(out_channels))
        bound = 1 / math.sqrt(in_channels * math.prod(self.kernel_size))
        nn.init.uniform_(self.bias, -bound, bound)
    else:
        self.register_parameter("bias", None)


def sparse_conv3d_forward(self, x):
    from flex_gemm.tenstorrent import build_subm_neighbor_map, sparse_conv_reference
    if x.feats.device.type != "cpu":
        raise ValueError("reference sparse convolution requires CPU tensors")
    if torch.is_grad_enabled() and (x.feats.requires_grad or self.weight.requires_grad):
        raise RuntimeError("reference sparse convolution is inference-only; use torch.no_grad()")
    if x.feats.shape[0] == 0:
        return x.replace(x.feats.new_empty((0, self.out_channels)))
    key = ("reference_subm", self.kernel_size, self.dilation)
    mapping = x.get_spatial_cache(key)
    if mapping is None:
        mapping = build_subm_neighbor_map(x.coords, (x.shape[0], *x.spatial_shape),
                                          self.kernel_size, self.dilation)
        x.register_spatial_cache(key, mapping)
    result = sparse_conv_reference(x.feats.detach(), self.weight.detach(), mapping,
                                    None if self.bias is None else self.bias.detach())
    return x.replace(result.to(x.dtype))


def sparse_inverse_conv3d_init(self, *args, **kwargs):
    raise NotImplementedError("reference inverse convolution is not implemented")


def sparse_inverse_conv3d_forward(self, x):
    raise NotImplementedError("reference inverse convolution is not implemented")
