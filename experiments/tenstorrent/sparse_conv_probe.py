#!/usr/bin/env python3
"""Research probe: pretrained submanifold-conv weights, host map, TT gather/GEMM.

This tests explicit im2col and per-offset products, not a sparse decoder backend.
It deliberately uses the existing isolated-run measurement/comparison harness.
"""

import argparse
import itertools
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from safetensors import safe_open


def fixture(n, channels, seed=123):
    generator = torch.Generator().manual_seed(seed)
    side = math.ceil((n / 0.4 / 2) ** (1 / 3))
    flat = torch.randperm(2 * side**3, generator=generator)[:n]
    coords = torch.stack((flat // side**3, flat // side**2 % side,
                          flat // side % side, flat % side), dim=1)
    lookup = {tuple(c): i for i, c in enumerate(coords.tolist())}
    offsets = list(itertools.product(range(-1, 2), repeat=3))
    neighbor = torch.tensor([
        [lookup.get((b, x + dx, y + dy, z + dz), n)
         for dx, dy, dz in offsets]
        for b, x, y, z in coords.tolist()
    ], dtype=torch.int64)
    features = torch.randn(n, channels, generator=generator)
    return features, neighbor, coords, side


class ExplicitSubmanifoldConv(nn.Module):
    def __init__(self, weight, bias, lowering='im2col'):
        super().__init__()
        self.lowering = lowering
        self.weight = nn.Parameter(weight.reshape(weight.shape[0], -1).T.contiguous())
        self.bias = nn.Parameter(bias.clone())

    def forward(self, features, neighbor):
        # Missing neighbors point to the appended zero row. Only active output
        # coordinates are evaluated; no dense spatial feature volume is created.
        padded = torch.cat((features, torch.zeros_like(features[:1])), dim=0)
        if self.lowering == 'offset_sum':
            channels = features.shape[1]
            result = torch.zeros((features.shape[0], self.bias.shape[0]),
                                 device=features.device, dtype=torch.float32)
            for k in range(27):
                contribution = padded[neighbor[:, k]] @ self.weight[k * channels:(k + 1) * channels]
                result = result + contribution.float()
            return (result + self.bias.float()).to(features.dtype)
        columns = padded[neighbor.reshape(-1)].reshape(neighbor.shape[0], -1)
        return columns @ self.weight + self.bias


def check_cpu_algebra():
    """Independent dense Conv3d check, including boundaries and batch isolation."""
    features, neighbor, coords, side = fixture(65, 5)
    generator = torch.Generator().manual_seed(456)
    weight = torch.randn(7, 3, 3, 3, 5, generator=generator)
    bias = torch.randn(7, generator=generator)
    model = ExplicitSubmanifoldConv(weight, bias)
    dense = torch.zeros(2, 5, side, side, side)
    b, x, y, z = coords.unbind(1)
    dense[b, :, x, y, z] = features
    expected = torch.nn.functional.conv3d(
        dense, weight.permute(0, 4, 1, 2, 3), bias, padding=1
    )[b, :, x, y, z]
    actual = model(features, neighbor)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    offset_model = ExplicitSubmanifoldConv(weight, bias, lowering='offset_sum')
    torch.testing.assert_close(offset_model(features, neighbor), expected, atol=2e-5, rtol=2e-5)
    return float((actual - expected).abs().max().detach())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layer', default='blocks.3.4.conv2')
    parser.add_argument('--n', type=int, default=257)
    parser.add_argument('--dtype', choices=('float32', 'bfloat16'), default='bfloat16')
    parser.add_argument('--math-fidelity', choices=('hifi3', 'hifi4'))
    parser.add_argument('--lowering', choices=('im2col', 'offset_sum'), default='im2col')
    parser.add_argument('--checkpoint', type=Path, default=Path.home() /
                        'models/genrecon/trellis2/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(16)
    algebra_error = check_cpu_algebra()

    def build_case(_name, _checkpoints):
        if args.math_fidelity:
            import torch_xla
            torch_xla.set_custom_compile_options({'math_fidelity': args.math_fidelity})
        with safe_open(args.checkpoint, framework='pt', device='cpu') as checkpoint:
            weight = checkpoint.get_tensor(args.layer + '.weight')
            bias = checkpoint.get_tensor(args.layer + '.bias')
        dtype = getattr(torch, args.dtype)
        features, neighbor, coords, side = fixture(args.n, weight.shape[-1])
        model = ExplicitSubmanifoldConv(weight.to(dtype), bias.to(dtype), args.lowering)
        metadata = {
            'scope': 'One pretrained convolution; synthetic features/coordinates; CPU neighbor map',
            'checkpoint': str(args.checkpoint), 'checkpoint_layer': args.layer,
            'original_weight_shape': list(weight.shape),
            'math_fidelity_override': args.math_fidelity,
            'lowering': args.lowering,
            'reference_detail': 'FP32 arithmetic after casting checkpoint weights and synthetic inputs to the requested probe dtype',
            'cpu_dense_algebra_max_abs_error': algebra_error,
            'active_voxels': args.n, 'batch_size': 2, 'spatial_side': side,
            'mean_present_neighbors': float((neighbor != args.n).sum(1).float().mean()),
            'full_im2col_bytes': args.n * 27 * weight.shape[-1] * torch.empty((), dtype=dtype).element_size(),
            'limitations': 'No pruning/C2S/whole decoder or realistic scene benchmark; compiler peak memory not measured',
        }
        return model, (features.to(dtype), neighbor), metadata

    # Reuse the established transfer, comparison, fallback and timing checks.
    import probes
    import run
    probes.build_case = build_case
    return run.worker(SimpleNamespace(
        worker='sparse_conv_explicit', output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint.parent, cpu_threads=16,
        reference_precision='float32', seed=123, runs=3,
    ))


if __name__ == '__main__':
    raise SystemExit(main())
