"""Module-sized conversion boundaries. Synthetic input is always reported.

Wrappers use the repository's implementations, not stand-ins for their math.
Random-weight probes establish operator coverage only, never model quality.
"""

import json
from pathlib import Path
import torch
from torch import nn


class ProjectionProbe(nn.Module):
    def __init__(self):
        super().__init__()
        from genrecon.modules.cond_3D.projection import Projection
        self.projection = Projection(16, 32)

    def forward(self, points, features, extrinsics, intrinsics):
        from genrecon.modules.cond_3D.projection import project_features_on_points
        projected, valid, _ = project_features_on_points(self.projection, points, features, extrinsics, intrinsics)
        return projected, valid


class SparseAggregatorProbe(nn.Module):
    def __init__(self, channels, voxels):
        super().__init__()
        from genrecon.modules.cond_3D.sparse_aggregation_net import MultiViewFeatAggregator
        self.aggregator = MultiViewFeatAggregator(channels)
        coords = torch.zeros(voxels, 4, dtype=torch.int32)
        coords[:, 1] = torch.arange(voxels)
        self.register_buffer("coords", coords)

    def forward(self, features, valid):
        from genrecon.modules.sparse import SparseTensor
        x = SparseTensor(features, self.coords, shape=torch.Size([1, *features.shape[1:]]))
        return self.aggregator(x, valid).feats


class DiTBlockProbe(nn.Module):
    def __init__(self):
        super().__init__()
        from genrecon.modules.transformer import ModulatedTransformerCrossBlock
        from genrecon.modules.attention import RotaryPositionEmbedder
        self.block = ModulatedTransformerCrossBlock(1536, 1024, 12, mlp_ratio=5.3334,
            use_rope=True, share_mod=True, qk_rms_norm=True, qk_rms_norm_cross=True)
        positions = torch.stack(torch.meshgrid(*[torch.arange(4)] * 3, indexing="ij"), dim=-1).reshape(-1, 3)
        self.register_buffer("phases", RotaryPositionEmbedder(128, 3)(positions))

    def forward(self, x, mod, context):
        return self.block(x, mod, context, self.phases)


class ProjectionIndexProbe(nn.Module):
    def forward(self, points, extrinsics, intrinsics):
        from genrecon.modules.cond_3D.projection import project_points_to_patches
        ones = torch.ones_like(points[:, :1])
        return project_points_to_patches(torch.cat([points, ones], dim=-1),
            extrinsics.reshape(8, 1, 4, 4), intrinsics.reshape(8, 1, 3, 3), 32, 5)


class ProjectionElementwiseProbe(nn.Module):
    """Small camera transforms expressed as elementwise FP32 reductions.

    Experiment only: test whether matmul precision causes discrete index errors.
    CPU equivalence to the original function is checked before device execution.
    """
    def forward(self, points, features, extrinsics, intrinsics):
        b, views, tokens, channels = features.shape
        hom = torch.cat([points, torch.ones_like(points[:, :1])], dim=-1)
        ext = extrinsics.reshape(views, 4, 4)
        intr = intrinsics.reshape(views, 3, 3)
        cam = (hom[None, :, None, :] * ext[:, None, :3, :]).sum(-1)
        depth = cam[..., 2]
        valid = depth > 1e-6
        xyz = cam / torch.where(valid, depth, torch.ones_like(depth)).unsqueeze(-1)
        uv = (xyz[:, :, None, :] * intr[:, None, :, :]).sum(-1)
        u, v = uv[..., 0], uv[..., 1]
        valid = valid & (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
        ids = torch.floor(v*32).long().clamp(0,31)*32 + torch.floor(u*32).long().clamp(0,31) + 5
        out = features.reshape(views,tokens,channels).gather(1, ids.unsqueeze(-1).expand(-1,-1,channels))
        return (out*valid.unsqueeze(-1)).reshape(b,views,-1,channels), valid.reshape(b,views,-1)


class ProjectionGatherProbe(nn.Module):
    def forward(self, features, ids, valid):
        b, views, tokens, channels = features.shape
        projected = features.reshape(views,tokens,channels).gather(1, ids.unsqueeze(-1).expand(-1,-1,channels))
        return (projected*valid.unsqueeze(-1)).reshape(b,views,-1,channels)


class SparseAttentionProbe(nn.Module):
    def forward(self, features):
        from genrecon.modules.sparse.basic import VarLenTensor
        from genrecon.modules.sparse.attention.full_attn import sparse_scaled_dot_product_attention
        return sparse_scaled_dot_product_attention(VarLenTensor(features, [slice(0, features.shape[0])])).feats


class SparseDiTBlockProbe(nn.Module):
    def __init__(self, precompute_rope=False):
        super().__init__()
        from genrecon.modules.sparse.transformer import ModulatedSparseTransformerCrossBlock
        self.block = ModulatedSparseTransformerCrossBlock(1536, 1024, 12, mlp_ratio=5.3334,
            use_rope=True, share_mod=True, qk_rms_norm=True, qk_rms_norm_cross=True)
        xyz = torch.stack(torch.meshgrid(*[torch.arange(4)]*3, indexing="ij"), dim=-1).reshape(-1,3)
        self.register_buffer("coords", torch.cat([torch.zeros(64,1,dtype=torch.int64), xyz], dim=-1).int())
        self.register_buffer("batch_map", torch.zeros(64,dtype=torch.int64))
        self.precompute_rope = precompute_rope
        if precompute_rope:
            from genrecon.modules.attention import RotaryPositionEmbedder
            rope = self.block.self_attn.rope
            self.rope_key = f"rope_phase_{rope.dim}d_freq{rope.rope_freq[0]}-{rope.rope_freq[1]}_hd{rope.head_dim}"
            self.register_buffer("phases", RotaryPositionEmbedder(rope.head_dim,rope.dim,rope.rope_freq)(xyz))

    def forward(self, x, mod, context):
        from genrecon.modules.sparse import SparseTensor
        sparse = SparseTensor(x, self.coords, shape=torch.Size([1, x.shape[-1]]), scale=(1,1,1))
        sparse.register_spatial_cache("layout", [slice(0, x.shape[0])])
        sparse.register_spatial_cache("batch_boardcast_map", self.batch_map)
        if self.precompute_rope:
            sparse.register_spatial_cache(self.rope_key,self.phases)
        return self.block(sparse, mod, context).feats


class EulerCfgProbe(nn.Module):
    """Exercise the real sampler with supplied positive/negative velocities.

    The prediction provider is deliberately not a neural denoiser. Its only role
    is to isolate CFG, optional rescaling and the Euler update from model cost.
    """
    def __init__(self):
        super().__init__()
        from genrecon.pipelines.samplers.flow_euler import FlowEulerCfgSampler
        self.sampler = FlowEulerCfgSampler(sigma_min=1e-5)

    @staticmethod
    def prediction(x, t, cond):
        return cond

    def forward(self, x, positive, negative):
        out = self.sampler.sample_once(self.prediction, x, 0.6, 0.5,
            cond=positive, neg_cond=negative, guidance_strength=3., guidance_rescale=0.5)
        return out.pred_x_prev, out.pred_x_0


class EulerCfgTensorProbe(EulerCfgProbe):
    def forward(self, x, positive, negative):
        # Same source numerical functions; keep EasyDict construction on host.
        x0, _, velocity = self.sampler._get_model_prediction(self.prediction, x, 0.6,
            cond=positive, neg_cond=negative, guidance_strength=3., guidance_rescale=0.5)
        return x - (0.6 - 0.5) * velocity, x0


class DenseOverlapProbe(nn.Module):
    def forward(self, a, b, offsets):
        from genrecon.pipelines.samplers.multi_diff_orchestrator import MultiDiffusionOrchestrator
        return MultiDiffusionOrchestrator(None)._aggregate_overlaps_dense([a,b], list(offsets.unbind()), 16)


class OverlapReductionProbe(nn.Module):
    """Only the scatter/average/gather; coordinate lookup is a host operation."""
    def __init__(self, inverse, counts):
        super().__init__()
        self.register_buffer("inverse", inverse)
        self.register_buffer("counts", counts)

    def forward(self, features):
        acc = torch.zeros(self.counts.shape[0], features.shape[1], device=features.device, dtype=features.dtype)
        acc = acc.scatter_add(0, self.inverse[:,None].expand_as(features), features)
        return (acc / self.counts[:,None])[self.inverse]


def build_case(name, checkpoints):
    if name in ("ss_flow", "shape_flow_stock", "tex_flow_stock", "shape_flow_tensor", "tex_flow_tensor",
                "ss_aggregator", "shape_aggregator", "tex_aggregator", "shape_flow_cached", "tex_flow_cached"):
        from flow_probes import build_flow_case
        return build_flow_case(name, checkpoints)
    meta = {"weights": "random", "input_kind": "seeded synthetic", "dtype": "float32"}
    if name in ("projection", "projection_indices", "projection_elementwise", "projection_cpu_indices"):
        points = torch.rand(256, 3) - 0.5
        ext = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 8, 1, 1)
        ext[:, :, 2, 3] = 1.0
        ext[:, 0, 2, 3] = -1.0  # invalid depth
        intr = torch.tensor([[1., 0., .5], [0., 1., .5], [0., 0., 1.]]).reshape(1, 1, 3, 3).repeat(1, 8, 1, 1)
        meta["weights"] = "none"
        if name == "projection_indices":
            return ProjectionIndexProbe(), (points, ext, intr), meta
        inputs = (points, torch.randn(1, 8, 1029, 1024), ext, intr)
        if name == "projection_cpu_indices":
            _, valid, ids = ProjectionIndexProbe()(points, ext, intr)
            meta["host_work"] = "camera projection, visibility and patch indices computed on CPU before the timed TT graph"
            model = ProjectionGatherProbe()
            torch.testing.assert_close(model(inputs[1], ids, valid), ProjectionProbe()(*inputs)[0])
            meta["cpu_original_equivalence"] = "torch.testing.assert_close passed"
            return model, (inputs[1], ids, valid), meta
        if name == "projection_elementwise":
            model = ProjectionElementwiseProbe()
            with torch.no_grad():
                for orig, adapted in zip(ProjectionProbe()(*inputs), model(*inputs)):
                    torch.testing.assert_close(adapted, orig)
            meta["cpu_original_equivalence"] = "torch.testing.assert_close passed for projected features and mask"
            return model, inputs, meta
        return ProjectionProbe(), inputs, meta
    if name in ("dense_aggregator", "sparse_aggregator"):
        from genrecon.modules.cond_3D.aggregation_net import MultiViewFeatAggregator
        feats = torch.randn(1, 8, 256, 1024)
        mask = torch.rand(1, 8, 256) > 0.25
        mask[:, :, 0] = False  # all-invalid voxel must remain zero
        model = MultiViewFeatAggregator(1024, 16) if name == "dense_aggregator" else SparseAggregatorProbe(1024, 256)
        aggr = model if name == "dense_aggregator" else model.aggregator
        # The source zero-initializes the last feature MLP. Exercise its entire
        # learned residual path instead of accidentally testing only a mean.
        nn.init.normal_(aggr.feature_mlp[-1].weight, std=0.02)
        nn.init.normal_(aggr.feature_mlp[-1].bias, std=0.02)
        if name == "sparse_aggregator":
            feats, mask = feats[0].permute(1, 0, 2).contiguous(), mask[0].T.contiguous()
        return model, (feats, mask), meta
    if name == "ss_decoder":
        from genrecon.models.sparse_structure_vae import SparseStructureDecoder
        from safetensors.torch import load_file
        stem = checkpoints / "trellis-image-large/ckpts/ss_dec_conv3d_16l8_fp16"
        config = json.loads(stem.with_suffix(".json").read_text())
        config["args"]["use_fp16"] = False
        model = SparseStructureDecoder(**config["args"])
        model.load_state_dict(load_file(str(stem.with_suffix(".safetensors"))), strict=True)
        meta.update(weights="pretrained", checkpoint=str(stem), revision="25e0d31ffbebe4b5a97464dd851910efc3002d96")
        return model, (torch.randn(1, 8, 16, 16, 16),), meta
    if name == "dit_block":
        return DiTBlockProbe(), (torch.randn(1, 64, 1536), torch.randn(1, 9216), torch.randn(1, 1029, 1024)), meta
    if name in ("sparse_attention_stock", "sparse_attention_sdpa"):
        from genrecon.modules.sparse import config
        config.ATTN = "flash_attn" if name.endswith("stock") else "sdpa"
        meta["weights"] = "none"
        return SparseAttentionProbe(), (torch.randn(64, 3, 12, 128),), meta
    if name == "sparse_dit_block":
        return SparseDiTBlockProbe(), (torch.randn(64,1536), torch.randn(1,9216), torch.randn(1,1029,1024)), meta
    if name == "sparse_dit_cached":
        inputs = (torch.randn(64,1536), torch.randn(1,9216), torch.randn(1,1029,1024))
        native = SparseDiTBlockProbe().eval()
        cached = SparseDiTBlockProbe(precompute_rope=True).eval()
        cached.block.load_state_dict(native.block.state_dict(),strict=True)
        with torch.no_grad():
            torch.testing.assert_close(cached(*inputs),native(*inputs),rtol=0,atol=0)
        meta.update(host_work="fixed coordinates/layout and RoPE phases prepared on CPU",
            cpu_original_equivalence="exact equality against native block with uncached RoPE")
        return cached, inputs, meta
    if name == "sparse_dit_tensor":
        native = SparseDiTBlockProbe().eval()
        model = DiTBlockProbe().eval()
        model.block.load_state_dict(native.block.state_dict(), strict=True)
        inputs = (torch.randn(1,64,1536), torch.randn(1,9216), torch.randn(1,1029,1024))
        with torch.no_grad():
            expected = native(inputs[0][0], *inputs[1:])
            torch.testing.assert_close(model(*inputs)[0], expected)
        meta.update(cpu_original_equivalence="torch.testing.assert_close passed against native sparse block",
            adaptation="single chunk active tokens as a dense sequence; identical state dict; no voxel densification")
        return model, inputs, meta
    if name in ("euler_cfg", "euler_cfg_tensor"):
        meta.update(weights="none", scope="one Euler/CFG step with supplied velocities, including guidance rescale")
        inputs = tuple(torch.randn(1,8,16,16,16) for _ in range(3))
        model = EulerCfgProbe() if name == "euler_cfg" else EulerCfgTensorProbe()
        if name.endswith("tensor"):
            torch.testing.assert_close(model(*inputs), EulerCfgProbe()(*inputs))
            meta["cpu_original_equivalence"] = "torch.testing.assert_close passed against sample_once"
        return model, inputs, meta
    if name == "overlap_dense_stock":
        meta["weights"] = "none"
        return DenseOverlapProbe(), (torch.randn(1,8,16,16,16), torch.randn(1,8,16,16,16),
            torch.tensor([[0.,0.,0.], [.5,0.,0.]])), meta
    if name == "overlap_cpu_indices":
        from genrecon.modules.sparse import SparseTensor
        from genrecon.pipelines.samplers.multi_diff_orchestrator import MultiDiffusionOrchestrator
        xyz = torch.stack(torch.meshgrid(*[torch.arange(4)]*3, indexing="ij"), dim=-1).reshape(-1,3)
        coords = torch.cat([torch.zeros(64,1,dtype=torch.int64),xyz], dim=-1).int()
        offsets = [torch.zeros(3), torch.tensor([.5,0.,0.])]
        global_xyz = torch.cat([xyz, xyz+torch.tensor([2,0,0])])
        _, inverse, counts = torch.unique(global_xyz, dim=0, return_inverse=True, return_counts=True)
        model = OverlapReductionProbe(inverse, counts.float())
        features = torch.randn(128,32)
        samples = [SparseTensor(feats=feats,coords=coords) for feats in features.split(64)]
        expected = MultiDiffusionOrchestrator(None)._aggregate_overlaps_sparse(samples,offsets,4)
        torch.testing.assert_close(model(features), torch.cat([s.feats for s in expected]))
        meta.update(weights="none", cpu_original_equivalence="torch.testing.assert_close passed against sparse overlap aggregation",
            host_work="global coordinate uniqueness, inverse indices and counts computed before timed TT graph")
        return model, (features,), meta
    if name in ("shape_decoder_stock", "tex_decoder_stock"):
        # Exercise the actual default dependency path, not the 'none' storage backend.
        from genrecon.modules.sparse import config
        config.CONV = "flex_gemm"
        if name == "shape_decoder_stock":
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "o-voxel"))
            from genrecon.models.sc_vaes.fdg_vae import FlexiDualGridVaeDecoder as Decoder
            filename = "shape_dec_next_dc_f16c32_fp16.json"
        else:
            from genrecon.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder as Decoder
            filename = "tex_dec_next_dc_f16c32_fp16.json"
        config = json.loads((checkpoints / "trellis2/ckpts" / filename).read_text())
        Decoder(**config["args"])
        raise RuntimeError("Decoder construction succeeded; add sparse tensor inputs before continuing")
    if name == "mesh_export_stock":
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "o-voxel"))
        import o_voxel.postprocess  # noqa: F401
        raise RuntimeError("Exporter import succeeded; add mesh/voxel fixture before continuing")
    raise ValueError(f"Unknown case: {name}")
