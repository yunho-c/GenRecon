"""Full ConvNeXt/C2S sparse VAE inference with TTNN feature residency."""

import time
import torch
from torch import nn

from ...modules.sparse import SparseTensor


def child_plan(coords, selection):
    """Preserve upstream parent-major, x-fastest child ordering exactly."""
    if (coords.device.type != "cpu" or selection.device.type != "cpu"
            or tuple(selection.shape) != (len(coords), 8)):
        raise ValueError("child selection must be CPU [N,8] for CPU [N,4] coordinates")
    parents, children = (selection > 0).nonzero(as_tuple=True)
    output = coords[parents].clone()
    for axis in range(3):
        output[:, axis + 1] = output[:, axis + 1] * 2 + ((children >> axis) & 1)
    return output, parents * 8 + children


class TopologyCache:
    """One layout per stage/kernel; exact coordinate equality prevents stale reuse.

    A shape and texture decoder can share this cache on the same device. It
    retains only the last layout at each stage, bounding cache growth over scenes.
    """

    def __init__(self):
        self.entries = {}
        self.device = None
        self.builds = 0
        self.hits = 0

    def get(self, coords, batch, spatial, kernel, dilation, stage, device, masked):
        from flex_gemm.tenstorrent import build_subm_neighbor_map, prepare_sparse_topology
        if self.device is not device:
            self.entries.clear()
            self.device = device
        key = (stage, tuple(kernel), tuple(dilation), masked)
        entry = self.entries.get(key)
        signature = (batch, tuple(spatial), device.id())
        if entry is not None and entry[0] == signature and torch.equal(entry[1], coords):
            self.hits += 1
            return entry[2]
        mapping = build_subm_neighbor_map(coords, (batch, *spatial), kernel, dilation)
        top = prepare_sparse_topology(mapping, len(coords), device=device, masked=masked)
        self.entries[key] = (signature, coords.clone(), top)
        self.builds += 1
        return top


class TenstorrentSparseDecoder:
    """Inference adapter for the original decoder's learned architecture.

    Convolutions use FlexGEMM; normalization, MLPs, residuals and C2S feature
    gathering use TTNN. Only eight subdivision logits per parent cross to CPU
    at refinement boundaries. Final outputs return to CPU for pipeline consumers.
    """

    def __init__(self, model, device, *, algorithm="explicit_gemm", row_chunk=256,
                 split_k=4, precision="accurate", topology_cache=None, observer=None):
        import ttnn
        from .precision import split_weight_digits
        from flex_gemm.tenstorrent import prepare_sparse_weights
        from flex_gemm.tenstorrent.ops import _compute_config
        from ...modules.sparse.conv.conv import SparseConv3d
        if model.training:
            raise ValueError("Tenstorrent decoder requires model.eval()")
        choices = {"explicit_gemm", "implicit_gemm", "masked_implicit_gemm",
                   "implicit_gemm_splitk", "masked_implicit_gemm_splitk"}
        if algorithm not in choices:
            raise ValueError(f"unknown sparse convolution algorithm: {algorithm}")
        if precision not in ("bf16", "accurate"):
            raise ValueError("precision must be bf16 or accurate")
        if any(c % 32 for c in model.model_channels):
            raise ValueError("TT decoder stage widths must be multiples of 32")
        for stage in model.blocks:
            for block in stage:
                if type(block).__name__ not in ("SparseConvNeXtBlock3d", "SparseResBlockC2S3d"):
                    raise ValueError(f"unsupported decoder block: {type(block).__name__}")
        if any(p.device.type != "cpu" for p in model.parameters()):
            raise ValueError("prepare the decoder from CPU parameters")
        self.model, self.device = model, device
        self.precision = precision
        self.dtype = ttnn.float32 if precision == "accurate" else ttnn.bfloat16
        self.algorithm, self.row_chunk, self.split_k = algorithm, row_chunk, split_k
        self.topology_cache = topology_cache if topology_cache is not None else TopologyCache()
        self.observer = observer
        self.config = _compute_config(device)
        self.weights = {}
        self.stats = {}
        start = time.perf_counter()
        for name, module in model.named_modules():
            if isinstance(module, SparseConv3d):
                if tuple(module.weight.shape) != (module.out_channels, *module.kernel_size, module.in_channels):
                    raise ValueError("TT decoder requires FlexGEMM checkpoint weight layout")
                if precision == "accurate":
                    digits, scale = split_weight_digits(module.weight)
                    self.weights[name] = ([prepare_sparse_weights(w, device=device) for w in digits],
                        self.upload(scale), None if module.bias is None else
                        self.upload(module.bias.detach().reshape(1, -1)))
                else:
                    self.weights[name] = prepare_sparse_weights(module.weight.detach(), device=device,
                        bias=None if module.bias is None else module.bias.detach())
            elif isinstance(module, nn.Linear):
                bias = None if module.bias is None else self.upload(module.bias.detach().reshape(1, -1))
                if precision == "accurate":
                    digits, scale = split_weight_digits(module.weight)
                    self.weights[name] = ([self.upload(w.T, ttnn.bfloat16) for w in digits],
                        self.upload(scale), bias)
                else:
                    self.weights[name] = (self.upload(module.weight.detach().T), bias)
            elif isinstance(module, nn.LayerNorm):
                self.weights[name] = (None if module.weight is None else self.upload(module.weight.detach().reshape(1, -1)),
                    None if module.bias is None else self.upload(module.bias.detach().reshape(1, -1)))
        ttnn.synchronize_device(device)
        self.prepare_seconds = time.perf_counter() - start
        self.parameter_signature = self._parameter_signature()

    def _parameter_signature(self):
        return tuple((id(p), p.data_ptr(), p._version, p.dtype) for p in self.model.parameters())

    def upload(self, values, dtype=None):
        import ttnn
        dtype = self.dtype if dtype is None else dtype
        values = values.float() if dtype == ttnn.float32 else values.bfloat16()
        return ttnn.from_torch(values.contiguous(), device=self.device,
                                dtype=dtype, layout=ttnn.TILE_LAYOUT)

    def linear(self, name, x):
        import ttnn
        if self.precision == "accurate":
            from .precision import split_feature_digits, digit_product
            weights, weight_scale, bias = self.weights[name]
            features, feature_scale = split_feature_digits(x)
            def product(a, b):
                return ttnn.matmul(a, b, dtype=ttnn.float32, compute_kernel_config=self.config)
            result = digit_product(features, weights, feature_scale, weight_scale, product)
            return result if bias is None else ttnn.add(result, bias)
        weight, bias = self.weights[name]
        return ttnn.linear(x, weight, bias=bias, compute_kernel_config=self.config)

    def norm(self, name, x, epsilon):
        import ttnn
        weight, bias = self.weights[name] if name is not None else (None, None)
        if self.precision == "accurate":
            from .precision import layer_norm_fp32
            return layer_norm_fp32(x, weight, bias, epsilon=epsilon)
        return ttnn.layer_norm(x, epsilon=epsilon, weight=weight, bias=bias,
                                compute_kernel_config=self.config)

    def conv(self, name, module, x, coords, batch, spatial, stage):
        from flex_gemm.tenstorrent import sparse_conv_device
        import ttnn
        start = time.perf_counter()
        top = self.topology_cache.get(coords, batch, spatial, module.kernel_size,
                module.dilation, stage, self.device, self.algorithm.startswith("masked"))
        self.stats["topology_seconds"] += time.perf_counter() - start
        if self.precision == "accurate":
            from .precision import split_feature_digits, digit_product
            weights, weight_scale, bias = self.weights[name]
            features, feature_scale = split_feature_digits(x)
            def product(a, b):
                return sparse_conv_device(a, b, top, algorithm=self.algorithm,
                    split_k=self.split_k, row_chunk=self.row_chunk, fp32_output=True)
            result = digit_product(features, weights, feature_scale, weight_scale, product)
            return result if bias is None else ttnn.add(result, bias)
        return sparse_conv_device(x, self.weights[name], top, algorithm=self.algorithm,
                                    split_k=self.split_k, row_chunk=self.row_chunk)

    def c2s(self, x, indices, channels):
        import ttnn
        ids = ttnn.from_torch(indices.reshape(1, -1).to(torch.uint32),
            dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        def gather(values):
            rows = ttnn.reshape(ttnn.to_layout(values, ttnn.ROW_MAJOR_LAYOUT), [-1, channels // 8])
            gathered = ttnn.embedding(ids, rows, layout=ttnn.TILE_LAYOUT)
            return ttnn.reshape(gathered, [-1, channels // 8])
        if self.precision == "accurate":
            # The pinned SDK's FP32 gather loses mantissa bits in its transpose
            # path. Three BF16 residual parts preserve FP32 values through the
            # supported embedding path, with no feature transfer to the host.
            residual, result = x, None
            for _ in range(3):
                part = ttnn.typecast(residual, ttnn.bfloat16)
                residual = ttnn.subtract(residual, ttnn.typecast(part, ttnn.float32))
                value = ttnn.typecast(gather(part), ttnn.float32)
                result = value if result is None else ttnn.add(result, value)
            return result
        return gather(x)

    def observe(self, name, values, coords):
        if self.observer is not None:
            import ttnn
            self.observer(name, ttnn.to_torch(values).float(), coords.clone())

    @torch.no_grad()
    def run(self, x, *, guide_subs=None, return_subs=False, stop_at=None):
        import ttnn
        if self.model.training or x.feats.requires_grad:
            raise ValueError("Tenstorrent decoder supports inference only")
        if self._parameter_signature() != self.parameter_signature:
            raise ValueError("decoder parameters changed; disable and re-enable Tenstorrent")
        if x.feats.device.type != "cpu" or x.coords.device.type != "cpu":
            raise ValueError("Tenstorrent decoder inputs must be CPU SparseTensor boundaries")
        if not torch.isfinite(x.feats).all():
            raise ValueError("Tenstorrent decoder features must be finite")
        stages = len(self.model.blocks)
        if guide_subs is not None and len(guide_subs) != stages - 1:
            raise ValueError("one subdivision guide is required per upsampling stage")
        if not self.model.pred_subdiv and guide_subs is None:
            raise ValueError("texture decoding requires shape subdivision guides")
        if stop_at is not None and not 0 <= stop_at < stages:
            raise ValueError("upsample_times is outside the decoder stage range")
        if return_subs and not self.model.pred_subdiv:
            raise ValueError("only shape decoders predict subdivision logits")
        start = time.perf_counter()
        coords = x.coords.clone()
        batch = x.shape[0]
        spatial = tuple(x.spatial_shape) if len(coords) else (1, 1, 1)
        self.stats = {"topology_seconds": 0.0, "stage_voxels": [len(coords)],
                      "feature_uploads": int(bool(len(coords))), "subdivision_downloads": 0}
        h = self.linear("from_latent", self.upload(x.feats)) if len(coords) else None
        subs = []
        for stage, blocks in enumerate(self.model.blocks):
            if stop_at == stage:
                return coords
            for index, block in enumerate(blocks):
                name = f"blocks.{stage}.{index}"
                if type(block).__name__ == "SparseConvNeXtBlock3d":
                    if h is None:
                        continue
                    residual = h
                    h = self.conv(name + ".conv", block.conv, h, coords, batch, spatial, stage)
                    h = self.norm(name + ".norm", h, block.norm.eps)
                    h = self.linear(name + ".mlp.2", ttnn.silu(self.linear(name + ".mlp.0", h)))
                    h = ttnn.add(h, residual)
                else:
                    logits = None
                    if self.model.pred_subdiv:
                        logits = (ttnn.to_torch(self.linear(name + ".to_subdiv", h)).float()
                                  if h is not None else torch.empty(0, 8))
                        self.stats["subdivision_downloads"] += int(h is not None)
                        sub = SparseTensor(logits, coords, torch.Size([batch, 8]))
                        sub.register_spatial_cache("shape", torch.Size(spatial))
                        subs.append(sub)
                    if guide_subs is not None:
                        guide = guide_subs[stage]
                        if not torch.equal(guide.coords, coords):
                            raise ValueError("subdivision guide coordinates/order do not match decoder parents")
                        selection = guide.feats
                    else:
                        selection = logits
                    next_coords, gather = child_plan(coords, selection)
                    if len(next_coords):
                        residual = h
                        h = ttnn.silu(self.norm(name + ".norm1", h, block.norm1.eps))
                        h = self.conv(name + ".conv1", block.conv1, h, coords, batch, spatial, stage)
                        h = self.c2s(h, gather, block.out_channels * 8)
                        skip = self.c2s(residual, gather, block.channels)
                        del residual
                        from .precision import repeat_channels
                        skip = repeat_channels(skip, block.out_channels // (block.channels // 8))
                        h = ttnn.silu(self.norm(name + ".norm2", h, block.norm2.eps))
                        h = self.conv(name + ".conv2", block.conv2, h, next_coords, batch,
                                      tuple(s * 2 for s in spatial), stage + 1)
                        h = ttnn.add(h, skip)
                        del skip
                    else:
                        h = None
                    coords, spatial = next_coords, tuple(s * 2 for s in spatial)
                    self.stats["stage_voxels"].append(len(coords))
                if h is not None:
                    self.observe(name, h, coords)
        if h is not None:
            h = self.linear("output_layer", self.norm(None, h, 1e-5))
            output = ttnn.to_torch(h).to(x.dtype)
        else:
            output = torch.empty(0, self.model.out_channels, dtype=x.dtype)
        result = SparseTensor(output, coords, torch.Size([batch, self.model.out_channels]))
        result.register_spatial_cache("shape", torch.Size(spatial))
        ttnn.synchronize_device(self.device)
        self.stats["total_seconds"] = time.perf_counter() - start
        return (result, subs) if return_subs else result
