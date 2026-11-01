"""FP32 device arithmetic for threshold-sensitive decoder paths."""


def repeat_channels(x, repeats, row_chunk=16384):
    """Repeat adjacent channels without a transpose page spanning all voxels.

    The SDK implements last-axis repeat_interleave via transpose/concat. Its
    row-major page grows with N and exceeds L1 for large scenes. Bound that
    page while preserving TILE layout; the SDK's row-major transpose loses
    low FP32 mantissa bits.
    """
    import ttnn
    if repeats == 1:
        return x
    rows, channels = x.shape
    outputs = []
    for start in range(0, rows, row_chunk):
        part = ttnn.slice(x, [start, 0], [min(start + row_chunk, rows), channels])
        part = ttnn.repeat_interleave(part, repeats, dim=1)
        outputs.append(part)
    while len(outputs) > 1:
        outputs = [ttnn.concat(outputs[i:i + 4], dim=0) if len(outputs[i:i + 4]) > 1
                   else outputs[i] for i in range(0, len(outputs), 4)]
    return outputs[0]


def split_weight_digits(weight):
    """Four signed base-64 digits, with one power-of-two scale per output.

    Weights use the native [out, ..., in] layout. Small integer operands avoid
    the observed loss in Wormhole's floating-point dot-product accumulation.
    All digits lie in [-32, 32] and are exactly representable in BF16.
    """
    import torch
    weight = weight.detach().float()
    maximum = weight.abs().flatten(1).amax(1)
    exponent = torch.ceil(torch.log2(maximum.clamp_min(1e-30))) - 5
    scale = torch.where(maximum == 0, 1., torch.exp2(exponent))
    residual = weight / scale.reshape(-1, *([1] * (weight.ndim - 1)))
    digits = []
    for _ in range(4):
        digit = residual.round()
        digits.append(digit.bfloat16())
        residual = (residual - digit) * 64
    return digits, scale.reshape(1, -1)


def split_feature_digits(x):
    """Extract four base-64 digits and a global scale without host transfers."""
    import ttnn
    maximum = ttnn.maximum(ttnn.max(ttnn.abs(x), keepdim=True), 1e-30)
    exponent = ttnn.subtract(ttnn.ceil(ttnn.log2(maximum)), 5.)
    scale = ttnn.exp2(exponent)
    inverse = ttnn.exp2(ttnn.neg(exponent))
    residual = ttnn.multiply(x, ttnn.repeat(inverse, list(x.shape)))
    digits = []
    for _ in range(4):
        digit = ttnn.round(residual)
        digits.append(ttnn.typecast(digit, ttnn.bfloat16))
        residual = ttnn.multiply(ttnn.subtract(residual, digit), 64.)
    return digits, scale


def digit_product(features, weights, feature_scale, weight_scale, product):
    """Accumulate 13 integer GEMMs in FP32, then restore operand scales.

    Terms through base-64 order four are retained. The three omitted terms
    have coefficients <= 2**-30; four digits retain about FP32 precision for
    normal decoder activations. This is an accuracy mode, not a fast GEMM.
    """
    import ttnn
    result = None
    for order in range(5):
        subtotal = None
        for i in range(4):
            j = order - i
            if 0 <= j < 4:
                value = product(features[i], weights[j])
                subtotal = value if subtotal is None else ttnn.add(subtotal, value)
                del value
        subtotal = ttnn.multiply(subtotal, 2. ** (-6 * order))
        result = subtotal if result is None else ttnn.add(result, subtotal)
        del subtotal
    rows, channels = result.shape
    # Explicit expansion selects the accurate SFPU binary product path.
    scale = ttnn.multiply(ttnn.repeat(feature_scale, [1, channels]), weight_scale)
    return ttnn.multiply(result, ttnn.repeat(scale, [rows, 1]))


def _sum_channels(x):
    """Pairwise FP32 additions avoid the FPU reduction's TF32 input conversion."""
    import ttnn
    rows, channels = x.shape
    padded = 1 << (channels - 1).bit_length()
    if padded != channels:
        x = ttnn.pad(x, [(0, 0), (0, padded - channels)], 0)
        channels = padded
    while channels > 1:
        half = channels // 2
        left = ttnn.slice(x, [0, 0], [rows, half])
        right = ttnn.slice(x, [0, half], [rows, 2 * half])
        summed = ttnn.add(left, right)
        x = summed
        channels = half
    return x


def layer_norm_fp32(x, weight=None, bias=None, *, epsilon=1e-6):
    """LayerNorm with FP32 sums and SFPU products, entirely on device.

    The pinned SDK's fused LayerNorm loses precision through its matrix-engine
    path even with FP32 input/output. Explicit expansion keeps the elementwise
    multiplies on the SFPU path. This trades launches and temporary buffers for
    accuracy at the decoder's discrete subdivision/edge decisions.
    """
    import ttnn
    if x.dtype != ttnn.float32 or len(x.shape) != 2:
        raise ValueError("layer_norm_fp32 requires a device FP32 [N,C] tensor")
    rows, channels = x.shape
    # A scalar binary operand may be encoded at reduced precision. Materialize
    # non-power-of-two reciprocals in FP32 before multiplying the row sums.
    scale = (1 / channels if channels & (channels - 1) == 0 else
             ttnn.full([rows, 1], 1 / channels, dtype=ttnn.float32,
                       layout=ttnn.TILE_LAYOUT, device=x.device()))
    mean = ttnn.multiply(_sum_channels(x), scale)
    centered = ttnn.subtract(x, ttnn.repeat(mean, [1, channels]))
    del mean
    variance = ttnn.multiply(_sum_channels(ttnn.multiply(centered, centered)), scale)
    inverse_std = ttnn.rsqrt(ttnn.add(variance, epsilon), fast_and_approximate_mode=False)
    del variance
    result = ttnn.multiply(centered, ttnn.repeat(inverse_std, [1, channels]))
    del centered, inverse_std
    if weight is not None:
        result = ttnn.multiply(result, ttnn.repeat(weight, [rows, 1]))
    if bias is not None:
        result = ttnn.add(result, bias)
    return result
