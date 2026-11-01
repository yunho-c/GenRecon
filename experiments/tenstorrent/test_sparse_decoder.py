"""CPU contract tests and optional TTNN execution of the complete decoder path."""

import pytest
import torch

from genrecon.modules.sparse import config, SparseTensor, SparseChannel2Spatial
from genrecon.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder
from genrecon.backends.tenstorrent.sparse_decoder import child_plan, TopologyCache


@pytest.fixture(autouse=True)
def reference_backend():
    previous = config.CONV
    threads = torch.get_num_threads()
    config.set_conv_backend("reference")
    torch.set_num_threads(8)
    yield
    config.set_conv_backend(previous)
    torch.set_num_threads(threads)


def problem(pred_subdiv=True):
    torch.manual_seed(21)
    model = SparseUnetVaeDecoder(7 if pred_subdiv else 6, [64, 32], 8, [1, 1],
        ["SparseConvNeXtBlock3d"] * 2, ["SparseResBlockC2S3d"], [{}, {}], pred_subdiv=pred_subdiv).eval()
    with torch.no_grad():
        # Exercise learned paths normally zeroed by initialization.
        for name, p in model.named_parameters():
            if p.ndim > 1:
                p.normal_(std=0.5 / (p.numel() / p.shape[0])**0.5)
            elif name.endswith("bias"):
                p.normal_(std=0.1)
    coords = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0],
                           [1, 0, 0, 0], [1, 0, 0, 1]], dtype=torch.int32)
    return model, SparseTensor(torch.randn(5, 8), coords, torch.Size([2, 8]))


@pytest.mark.parametrize("empty", [False, True])
def test_child_order_matches_native(empty):
    model, x = problem()
    values = torch.arange(5 * 64).reshape(5, 64).float()
    x = x.replace(values)
    selected = torch.zeros(5, 8) if empty else torch.randn(5, 8)
    guide = SparseTensor(selected > 0, x.coords, torch.Size([2, 8]))
    native = SparseChannel2Spatial(2).eval()(x, guide)
    coords, indices = child_plan(x.coords, selected)
    torch.testing.assert_close(coords, native.coords)
    torch.testing.assert_close(values.reshape(-1, 8)[indices], native.feats)


def test_shape_import_does_not_require_mesh_extensions():
    from genrecon.models.sc_vaes.fdg_vae import FlexiDualGridVaeDecoder
    assert issubclass(FlexiDualGridVaeDecoder, SparseUnetVaeDecoder)


@pytest.fixture(scope="module")
def device():
    ttnn = pytest.importorskip("ttnn")
    if not ttnn.GetNumAvailableDevices():
        pytest.skip("no Tenstorrent device")
    d = ttnn.open_device(device_id=0)
    yield d
    ttnn.close_device(d)


@pytest.mark.parametrize("algorithm,precision", [("explicit_gemm", "bf16"),
    ("masked_implicit_gemm_splitk", "bf16"), ("explicit_gemm", "accurate"),
    ("masked_implicit_gemm_splitk", "accurate")])
def test_full_decoder_and_texture_guides(device, algorithm, precision):
    model, x = problem()
    with torch.no_grad():
        expected, guides = model.forward_features(x, return_subs=True)
    cache = TopologyCache()
    model.enable_tenstorrent(device, algorithm=algorithm, row_chunk=3, topology_cache=cache, precision=precision)
    actual, predicted = model.forward_features(x, guide_subs=guides, return_subs=True)
    torch.testing.assert_close(actual.coords, expected.coords)
    tolerance = 3e-5 if precision == "accurate" else 0.04
    torch.testing.assert_close(actual.feats, expected.feats, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(predicted[0].feats, guides[0].feats, atol=tolerance, rtol=tolerance)
    if precision == "accurate":
        free = model.forward_features(x)
        torch.testing.assert_close(free.coords, expected.coords)
        torch.testing.assert_close(predicted[0].feats > 0, guides[0].feats > 0)
    builds = cache.builds
    again = model.forward_features(x, guide_subs=guides)
    torch.testing.assert_close(again.feats, actual.feats, atol=0, rtol=0)
    assert cache.builds == builds and cache.hits > 0
    stats = model._tenstorrent_backend.stats
    assert stats["feature_uploads"] == 1 and stats["subdivision_downloads"] == 1
    texture, tx = problem(pred_subdiv=False)
    with torch.no_grad():
        tex_expected = texture(tx, guide_subs=guides)
    texture.enable_tenstorrent(device, algorithm=algorithm, row_chunk=3, topology_cache=cache, precision=precision)
    tex_actual = texture(tx, guide_subs=guides)
    torch.testing.assert_close(tex_actual.coords, expected.coords)
    torch.testing.assert_close(tex_actual.feats, tex_expected.feats, atol=tolerance, rtol=tolerance)
    assert cache.builds == builds
    # Coordinate equality, not just the row count, protects map reuse.
    changed = SparseTensor(x.feats, x.coords + torch.tensor([0, 2, 0, 0]), x.shape)
    model.forward_features(changed)
    assert cache.builds > builds


def test_empty_children_and_parameter_invalidation(device):
    model, x = problem()
    guide = SparseTensor(torch.full((5, 8), -1.), x.coords, torch.Size([2, 8]))
    model.enable_tenstorrent(device, precision="bf16")
    with pytest.raises(ValueError, match="finite"):
        model.forward_features(x.replace(torch.full_like(x.feats, float("nan"))))
    output = model.forward_features(x, guide_subs=[guide])
    assert tuple(output.feats.shape) == (0, 7)
    assert tuple(output.coords.shape) == (0, 4)
    with torch.no_grad():
        model.output_layer.weight.add_(1)
    with pytest.raises(ValueError, match="parameters changed"):
        model.forward_features(x)


def test_accurate_linear_preserves_input_residual(device):
    import ttnn
    model, _ = problem()
    with torch.no_grad():
        model.from_latent.weight.zero_()
        model.from_latent.weight[:, 0] = 1
        model.from_latent.bias.fill_(-256.)
    model.enable_tenstorrent(device, precision="accurate")
    backend = model._tenstorrent_backend
    values = torch.zeros(3, 8)
    values[:, 0] = 257.125
    result = backend.linear("from_latent", backend.upload(values))
    torch.testing.assert_close(ttnn.to_torch(result), torch.full((3, 64), 1.125), atol=0, rtol=0)


def test_accurate_feature_rearrangement_is_lossless(device):
    import ttnn
    model, _ = problem()
    model.enable_tenstorrent(device, precision="accurate")
    backend = model._tenstorrent_backend
    torch.manual_seed(8)
    x = torch.randn(5, 256)
    indices = torch.tensor([0, 3, 8, 10, 24, 33, 39])
    actual = backend.c2s(backend.upload(x), indices, 256)
    torch.testing.assert_close(ttnn.to_torch(actual), x.reshape(-1, 32)[indices], atol=0, rtol=0)


@pytest.mark.parametrize("rows,channels,bf16", [(388659, 32, False), (1711800, 16, False), (388659, 32, True)])
def test_large_channel_repetition_is_lossless(device, rows, channels, bf16):
    import ttnn
    from genrecon.backends.tenstorrent.precision import repeat_channels
    x = torch.randn(rows, channels, generator=torch.Generator().manual_seed(401))
    if bf16:
        x = x.bfloat16()
    xd = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16 if bf16 else ttnn.float32,
                         layout=ttnn.TILE_LAYOUT)
    actual = ttnn.to_torch(repeat_channels(xd, 4))
    torch.testing.assert_close(actual, x.repeat_interleave(4, dim=1), atol=0, rtol=0)


def test_accurate_linear_with_mixed_magnitudes(device):
    import ttnn
    from flex_gemm.tenstorrent.ops import _compute_config
    from genrecon.backends.tenstorrent.precision import (
        split_weight_digits, split_feature_digits, digit_product)
    torch.manual_seed(12)
    x, weight = torch.randn(35, 1024), torch.randn(37, 1024) / 32
    x[0] = 0
    x[1] *= 0.01
    x[2] *= 10
    digits, scale = split_weight_digits(weight)
    def upload(t, dtype=ttnn.float32):
        return ttnn.from_torch(t.contiguous(), device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    xd, xs = split_feature_digits(upload(x))
    wd = [upload(w.T, ttnn.bfloat16) for w in digits]
    result = digit_product(xd, wd, xs, upload(scale),
        lambda a, b: ttnn.matmul(a, b, dtype=ttnn.float32, compute_kernel_config=_compute_config(device)))
    reference = (x.double() @ weight.double().T).float()
    torch.testing.assert_close(ttnn.to_torch(result), reference, atol=1e-5, rtol=3e-6)


@pytest.mark.parametrize("channels", [35, 64, 512, 1024])
def test_fp32_layer_norm(device, channels):
    import ttnn
    from genrecon.backends.tenstorrent.precision import layer_norm_fp32
    torch.manual_seed(121)
    x, w, b = torch.randn(35, channels) * 3, torch.randn(channels), torch.randn(channels)
    x[0] = 0.125  # zero variance exercises epsilon and affine bias
    def upload(t):
        return ttnn.from_torch(t, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    actual = layer_norm_fp32(upload(x), upload(w[None]), upload(b[None]), epsilon=1e-6)
    expected = torch.nn.functional.layer_norm(x, (channels,), w, b, 1e-6)
    torch.testing.assert_close(ttnn.to_torch(actual), expected, atol=3e-6, rtol=3e-6)
