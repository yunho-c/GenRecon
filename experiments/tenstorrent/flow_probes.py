"""Pretrained flow probes, with strict loading and explicit compilation boundaries."""
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = {
    "ss": ("sparse_structure", "configs/gen/ss_flow_img/genrecon.json"),
    "shape": ("shape_slat", "configs/gen/slat_flow_img2shape/genrecon_512.json"),
    "tex": ("texture_slat", "configs/gen/slat_flow_imgshape2tex/genrecon_512.json"),
}


def load_model(stage, checkpoints):
    from easydict import EasyDict
    from genrecon.utils.model_wrapper_utils import build_single_model
    from genrecon.modules.utils import convert_module_to
    stem, config_path = CONFIGS[stage]
    path = checkpoints / (stem + ".pt")
    raw = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    config = EasyDict(json.loads((ROOT / config_path).read_text())["models"]["denoiser"])
    model = build_single_model(config)
    expected = set(model.state_dict())
    candidates = []

    def visit(value, label="root"):
        if isinstance(value, dict):
            tensors = {k: v for k,v in value.items() if isinstance(v,torch.Tensor)}
            if tensors:
                candidates.append((len(expected & tensors.keys()), label, tensors))
            for key, child in value.items():
                if isinstance(child,dict):
                    visit(child, label + "." + key)

    visit(raw)
    if not candidates:
        raise ValueError("Checkpoint contains no tensor state dictionary")
    _, label, state = max(candidates, key=lambda c: c[0])
    model.load_state_dict(state, strict=True)
    # Match the pipeline's coercion: linears in torso/aggregator/projection are
    # BF16; embeddings, output and normalization parameters retain their types.
    model.convert_to(torch.bfloat16)
    for attr in ("projection", "aggregator", "proj_linears"):
        getattr(model, attr).apply(lambda m: convert_module_to(m, dtype=model.dtype))
    model.eval()
    provenance = json.loads(path.with_suffix(".source.json").read_text())
    return model, config, dict(weights="pretrained", checkpoint=provenance,
        checkpoint_state=label, strict_loading=True, config=config_path,
        dtype="pipeline mixed precision: BF16 torso, FP32 embedding/output/norm",
        input_kind="seeded synthetic", blocks=len(model.blocks))


class DenseFlowProbe(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, t, context, projected):
        return self.model(x, t, {"cond_2D": context, "cond_3D": projected})


class NativeSparseFlowProbe(nn.Module):
    def __init__(self, model, coords):
        super().__init__()
        self.model = model
        self.register_buffer("coords", coords)
        self.register_buffer("batch_map", torch.zeros(len(coords),dtype=torch.int64))

    def sparse(self, features):
        from genrecon.modules.sparse import SparseTensor
        x = SparseTensor(features, self.coords, shape=torch.Size([1,features.shape[-1]]), scale=(1,1,1))
        x.register_spatial_cache("layout", [slice(0,features.shape[0])])
        x.register_spatial_cache("batch_boardcast_map", self.batch_map)
        return x

    def forward(self, x, t, context, projected):
        return self.model(self.sparse(x), t,
            {"cond_2D": context, "cond_3D": self.sparse(projected)}).feats


class SingleChunkSLatProbe(nn.Module):
    """Equivalent active-token sequence for one fixed-layout sparse chunk.

    Uses native dense blocks with strictly copied sparse block state, including
    LoRA. No grid densification, weight merging, retraining or attention changes.
    Coordinates/RoPE are prepared on CPU. This is not a ragged scene runtime.
    """
    def __init__(self, model, config, coords):
        super().__init__()
        from genrecon.modules.transformer import ModulatedTransformerCrossBlock
        from genrecon.modules.attention import RotaryPositionEmbedder
        from genrecon.modules.lora import apply_lora_to_attention
        self.dtype = model.dtype
        self.input_layer = nn.Linear(model.in_channels, model.model_channels)
        self.input_layer.load_state_dict(model.input_layer.state_dict(), strict=True, assign=True)
        self.out_layer = nn.Linear(model.model_channels, model.out_channels)
        self.out_layer.load_state_dict(model.out_layer.state_dict(), strict=True, assign=True)
        self.t_embedder = model.t_embedder
        self.adaLN_modulation = model.adaLN_modulation
        self.proj_linears = model.proj_linears
        self.blocks = nn.ModuleList()
        for original in model.blocks:
            with torch.device("meta"):
                block = ModulatedTransformerCrossBlock(model.model_channels, model.cond_channels, model.num_heads,
                    mlp_ratio=model.mlp_ratio, use_rope=True, share_mod=True,
                    qk_rms_norm=model.qk_rms_norm, qk_rms_norm_cross=model.qk_rms_norm_cross)
                for attn in (block.self_attn, block.cross_attn):
                    apply_lora_to_attention(attn, config.lora.r, config.lora.alpha)
            block.load_state_dict(original.state_dict(), strict=True, assign=True)
            self.blocks.append(block)
        self.register_buffer("phases", RotaryPositionEmbedder(model.model_channels//model.num_heads,3)(coords[:,1:]))

    def forward(self, x, t, context, projected):
        h = self.input_layer(x).unsqueeze(0).to(self.dtype)
        mod = self.adaLN_modulation(self.t_embedder(t)).to(self.dtype)
        context = context.to(self.dtype)
        projected = projected.unsqueeze(0).to(self.dtype)
        for block, projection in zip(self.blocks,self.proj_linears):
            h = h + projection(projected)
            h = block(h,mod,context,self.phases)
        h = F.layer_norm(h.to(x.dtype), h.shape[-1:])
        return self.out_layer(h)[0]


def build_flow_case(name, checkpoints):
    stage = name.split("_")[0]
    model, config, metadata = load_model(stage, checkpoints)
    if name.endswith("aggregator"):
        from probes import SparseAggregatorProbe
        feats = torch.randn(1,8,256,1024,dtype=torch.bfloat16)
        valid = torch.rand(1,8,256) > .25
        valid[:,:,0] = False
        if stage == "ss":
            return model.aggregator, (feats,valid), metadata
        wrapper = SparseAggregatorProbe(1024,256)
        wrapper.aggregator = model.aggregator
        return wrapper, (feats[0].permute(1,0,2).contiguous(),valid[0].T.contiguous()), metadata
    if stage == "ss":
        inputs = (torch.randn(1,8,16,16,16), torch.tensor([600.]),
            torch.randn(1,1029,1024), torch.randn(1,4096,1024))
        return DenseFlowProbe(model), inputs, metadata
    xyz = torch.stack(torch.meshgrid(*[torch.arange(8)]*3,indexing="ij"),dim=-1).reshape(-1,3)
    coords = torch.cat([torch.zeros(len(xyz),1,dtype=torch.int64),xyz],dim=-1).int()
    inputs = (torch.randn(len(coords), model.in_channels), torch.tensor([600.]),
        torch.randn(1,1029,1024), torch.randn(len(coords),1024))
    native = NativeSparseFlowProbe(model, coords)
    if name.endswith("stock"):
        return native, inputs, metadata
    adapted = SingleChunkSLatProbe(model,config,coords).eval()
    with torch.no_grad():
        expected = native(*inputs)
        actual = adapted(*inputs)
        torch.testing.assert_close(actual,expected)
    metadata.update(cpu_original_equivalence="torch.testing.assert_close passed for all 30 sparse blocks and IO",
        adaptation="single chunk active tokens as a dense sequence; all native sparse weights including LoRA preserved",
        host_work="fixed coordinates and RoPE phases; precomputed image conditions",
        active_voxels=len(coords))
    return adapted, inputs, metadata
