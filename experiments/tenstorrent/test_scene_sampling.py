"""Check that the timing adapter preserves original Euler/CFG semantics."""
import json
import os
from pathlib import Path

os.environ["SPARSE_CONV_BACKEND"] = "none"
os.environ["SPARSE_ATTN_BACKEND"] = "sdpa"
os.environ["ATTN_BACKEND"] = "sdpa"

import pytest
import torch

from experiments.tenstorrent.scene_sampling import TensorFlowAdapter, MeasuredSampler, features
from genrecon.modules.sparse import SparseTensor
from genrecon.pipelines.samplers import FlowEulerGuidanceIntervalSampler


@pytest.mark.parametrize("stage,count", [("ss", 22), ("shape", 21), ("tex", 18)])
def test_original_sampler_equivalence(stage, count):
    torch.manual_seed(7)
    cfg = json.loads((Path(__file__).resolve().parents[2] / "configs/pipelines/texture.json").read_text())["args"]
    family = {"ss": "sparse_structure", "shape": "shape_slat", "tex": "tex_slat"}[stage]
    params = cfg[family + "_sampler"]["params"]
    coords = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 2, 3]], dtype=torch.int32)
    noise = torch.randn(1, 3, 4) if stage == "ss" else SparseTensor(torch.randn(3, 4), coords)
    concat = SparseTensor(torch.randn(3, 4), coords) if stage == "tex" else None
    positive, negative = torch.randn(4), torch.zeros(4)
    def native(x, t, cond, concat_cond=None):
        value = features(x)
        result = value.sin() * .2 + t.reshape(-1)[0] / 1000 * value * .1 + cond
        if concat_cond is not None:
            result = result + concat_cond.feats.cos() * .15
        return result if isinstance(x, torch.Tensor) else x.replace(result)
    def tensor_forward(x, t, cond):
        # An independently expressed native-model boundary: adapter must concat
        # shape features correctly without changing sampler/CFG reductions.
        base = x[..., :4]
        out = base.sin() * .2 + t[0] / 1000 * base * .1 + cond
        return out if stage != "tex" else out + x[..., 4:].cos() * .15
    expected = FlowEulerGuidanceIntervalSampler(1e-5).sample(native, noise,
        cond=positive, neg_cond=negative, concat_cond=concat, verbose=False, **params)
    adapter = TensorFlowAdapter(tensor_forward)
    sampler = MeasuredSampler()
    actual = sampler.sample(adapter, noise, cond=positive, neg_cond=negative,
        concat_cond=concat, verbose=False, **params)
    torch.testing.assert_close(features(actual.samples), features(expected.samples), rtol=0, atol=0)
    for a, b in zip(actual.pred_x_t, expected.pred_x_t):
        torch.testing.assert_close(features(a), features(b), rtol=0, atol=0)
    assert len(adapter.calls) == count
    assert len(sampler.steps) == 12
    assert sum(s["forwards"] for s in sampler.steps) == count


def test_texture_coordinate_mismatch_rejected():
    x = SparseTensor(torch.ones(2, 4), torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.int32))
    condition = SparseTensor(torch.ones(2, 4), x.coords.flip(0))
    adapter = TensorFlowAdapter(lambda *args: torch.zeros(2, 4))
    with pytest.raises(ValueError, match="coordinate order"):
        adapter(x, torch.tensor([1.]), {}, concat_cond=condition)


@pytest.mark.parametrize("channels,group_size", [(32, 1), (32, 2), (64, 2)])
def test_grouped_native_flow_equivalence(channels, group_size):
    from torch import nn
    from experiments.tenstorrent.flow_probes import NativeSparseFlowProbe
    from experiments.tenstorrent.grouped_flow import flow_groups
    from genrecon.models.structured_latent_flow import SLatFlowModel
    from genrecon.modules.lora import apply_lora_to_attention
    torch.manual_seed(9)
    model = SLatFlowModel(resolution=32, in_channels=channels, out_channels=32,
        model_channels=48, cond_channels=24, num_heads=4, num_blocks=3,
        mlp_ratio=2, pe_mode="rope", share_mod=True,
        qk_rms_norm=True, qk_rms_norm_cross=True).eval()
    model.proj_linears = nn.ModuleList(nn.Linear(24, 48) for _ in model.blocks)
    for block in model.blocks:
        for attn in (block.self_attn, block.cross_attn):
            apply_lora_to_attention(attn, 4, 4)
    coords = torch.cat([torch.zeros(17, 1, dtype=torch.int32), torch.randint(0, 32, (17, 3), dtype=torch.int32)], dim=1)
    args = (torch.randn(17, channels), torch.tensor([712.]), torch.randn(1, 13, 24), torch.randn(17, 24))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("lora_B"):
                parameter.normal_(std=.05)
        model.out_layer.weight.normal_(std=.1)  # avoid the default zero head
        expected = NativeSparseFlowProbe(model, coords, precompute_rope=True)(*args)
        actual = args[0]
        for group in flow_groups(model, coords, group_size):
            actual = group(actual, *args[1:])
        assert expected.abs().max() > .1
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
