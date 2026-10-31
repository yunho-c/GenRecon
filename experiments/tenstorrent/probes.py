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


class SparseAttentionProbe(nn.Module):
    def forward(self, features):
        from genrecon.modules.sparse.basic import VarLenTensor
        from genrecon.modules.sparse.attention.full_attn import sparse_scaled_dot_product_attention
        return sparse_scaled_dot_product_attention(VarLenTensor(features, [slice(0, features.shape[0])])).feats


def build_case(name, checkpoints):
    meta = {"weights": "random", "input_kind": "seeded synthetic", "dtype": "float32"}
    if name in ("projection", "projection_indices"):
        points = torch.rand(256, 3) - 0.5
        ext = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 8, 1, 1)
        ext[:, :, 2, 3] = 1.0
        ext[:, 0, 2, 3] = -1.0  # invalid depth
        intr = torch.tensor([[1., 0., .5], [0., 1., .5], [0., 0., 1.]]).reshape(1, 1, 3, 3).repeat(1, 8, 1, 1)
        meta["weights"] = "none"
        if name == "projection_indices":
            return ProjectionIndexProbe(), (points, ext, intr), meta
        return ProjectionProbe(), (points, torch.randn(1, 8, 1029, 1024), ext, intr), meta
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
    if name == "sparse_attention_stock":
        meta["weights"] = "none"
        return SparseAttentionProbe(), (torch.randn(64, 3, 12, 128),), meta
    if name in ("shape_decoder_stock", "tex_decoder_stock"):
        # Exercise the actual default dependency path, not the 'none' storage backend.
        from genrecon.modules.sparse import config
        config.CONV = "flex_gemm"
        if name == "shape_decoder_stock":
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
