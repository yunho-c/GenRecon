# Sparse shape and texture VAE decoders on Tenstorrent

Research date: 2026-11-01. Target: GenRecon inference on the available Wormhole B0
device, with CPU orchestration and no NVIDIA GPU. SfM, training, and mesh/export
acceleration are outside this study.

## Assessment

**Proceed with a port. No fundamental algorithmic or hardware obstacle was
identified.** Both decoders can be expressed using host-generated neighbor maps,
device gathers, dense matrix multiplication, normalization, activations, and
four sparse upsampling transitions. The same sparse-convolution backend can
serve both decoders without changing their learned architecture or retraining.

This conclusion has three different evidence levels:

1. **Directly demonstrated here:** individual pretrained decoder convolutions,
   including the widest 1024-to-4096 upsampling projection, execute on Wormhole
   through TT-XLA with no reported XLA CPU fallback.
2. **Close external precedent:** public TRELLIS.2 ports implement the same VAE
   architecture on Apple Silicon. Maintainers report successful inference with
   Metal sparse convolutions; the source also provides portable gather/GEMM
   implementations.
3. **Engineering inference:** a complete CPU+TT decoder should be achievable,
   but acceptable reconstruction quality, full-scene memory use, and useful
   performance remain unproven. The current experiments do not establish them.

Confidence is high in functional feasibility, conditional on replacing the
sparse backend and separating host control from numerical graphs. Confidence in
a fast, accurate production port is lower until full decoder tests pass.

## What actually needs porting

The local checkpoint configs and source, rather than class docstrings, define
the workload. These checkpoints use convolutional blocks, not sparse attention.

| Property | Shape decoder | Texture decoder |
| --- | --- | --- |
| Parameters, counted from safetensors headers | 474,231,463 | 474,216,006 |
| Latent channels | 32 | 32 |
| Stage widths | 1024, 512, 256, 128, 64 | Same |
| ConvNeXt blocks by stage | 4, 16, 8, 4, 0 | Same |
| Sparse upsampling transitions | Four, each doubles spatial resolution | Same |
| Child occupancy decisions | Predicts eight logits per parent; keeps logits > 0 | Uses shape decoder's subdivision guides |
| Output channels | 7: vertex offsets, edge flags, quad weights | 6: material attributes |

Each decoder contains **40 full-channel 3x3x3 submanifold convolutions**: 32 in
ConvNeXt blocks and two in each of four upsampling blocks. The convolution named
`SparseConvNeXtBlock3d.conv` is not depthwise. Each ConvNeXt block also has a
channel MLP with expansion ratio four, layer normalization, SiLU, and a residual.

An upsampling block computes a convolution from C to 8*C_next channels, selects
children, rearranges channel groups into child feature rows, then applies a
C_next-to-C_next convolution. Thus the first such projection is 1024-to-4096,
not merely 1024-to-512. This is a material workload for benchmarking.

The required sparse convolution is stride-one submanifold convolution: output
coordinates equal input coordinates. Neither sparse inverse convolution nor
generic arbitrary-stride sparse convolution is needed for these checkpoints.

Source: [decoder blocks and forward](../../genrecon/models/sc_vaes/sparse_unet_vae.py),
[FlexGEMM adapter](../../genrecon/modules/sparse/conv/conv_flex_gemm.py),
[channel-to-space operation](../../genrecon/modules/sparse/spatial/spatial2channel.py),
and [shape output interpretation](../../genrecon/models/sc_vaes/fdg_vae.py).
Checkpoint configs are in `/home/user/models/genrecon/trellis2/ckpts/`.

## Why there is no apparent fundamental obstacle

For an active output voxel i, a convolution is simply:

    Y[i, :] = bias + sum over 27 offsets k of X[neighbor(i, k), :] @ W[k, :, :]

An absent neighbor contributes zero. The neighbor relation is integer metadata;
the expensive arithmetic consists of ordinary dense matrix products. A host
hash map or sorted-coordinate lookup can build the indices. An output-centered
gather implementation also avoids requiring floating-point scatter atomics.

The shape decoder's data-dependent part is concentrated at four refinement
boundaries. The host can compact the selected children and build the next
neighbor map while numerical feature tensors remain on TT. Within each spatial
stage, many convolutions reuse the same map. Texture decoding can reuse the
shape decoder's maps and child selections if coordinate ordering, batching,
kernel, and scale are identical. Its complete subdivision schedule is already
known after shape decoding.

This requires a new backend, not merely installing the original CUDA extension.
The previous stock-decoder failures happened during O-Voxel/FlexGEMM imports,
before TT compilation. They were not rejection of the decoder mathematics.
GenRecon already separates the neural shape forward from mesh extraction in
[joint decoding](../../genrecon/pipelines/joint_decode.py); lazy imports can
remove that dependency coupling for a CPU+TT implementation.

Tenstorrent exposes both [gather](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api/ttnn.gather.html)
and programmable [reader/compute/writer kernels](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/matmul_single_core.html).
The latter provide a credible optimization path: read indexed feature rows,
assemble tiles in local memory, run matrix products, and write output tiles.
That is a proposed design, not an existing validated TT sparse-convolution kernel.

I did not find a ready-made, validated Tenstorrent submanifold-convolution
backend in the public sources searched. **There are directly relevant public
porting attempts**, including Tenstorrent's MapTR work described below; absence
of a ready-made backend does not mean nobody has attempted the operation.
The existing [TTNN sparse_matmul](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api/ttnn.sparse_matmul.html)
operates on sparse matrix groups; it is not a drop-in coordinate-based 3D
convolution. Documentation for current main/latest also does not guarantee that
every API is available in our installed TT-Forge 1.4.0 stack.

## Closest prior art

### Tenstorrent MapTR: a direct sparse-convolution porting attempt

An expanded search on 2026-11-01 found relevant prior work missed by the initial
study. Tenstorrent's [MapTR model source](https://github.com/tenstorrent/tt-forge-models/blob/45ed219bd6d2b22c22cfe0f6473a19392a36e51a/maptr/pytorch/src/maptr.py#L3096)
contains `get_indice_pairs_3d_cpu`, `get_indice_conv_cpu_fp32`, `SparseConvolution`,
and `SubMConv3d`. Its feature computation uses gather, matrix multiplication,
and scatter accumulation, with cached index pairs and host-side loop control.
This is directly related to the operation needed by our decoders.

[TT-XLA issue #4013](https://github.com/tenstorrent/tt-xla/issues/4013), opened
2026-03-31 and still open when checked on 2026-11-01, documents an actual attempt
to run the MapTR Tiny Fusion path on TT. After a tracing fix, Python loops in
neighbor-map construction caused an indefinite hang. A vectorized replacement
then failed TT compilation while legalizing a particular `stablehlo.scatter`
pattern. The issue links logs and a reproduction branch. Its
[isolated neighbor-map test](https://github.com/tenstorrent/tt-xla/blob/sai_arthi_raguram/maptr/tests/torch/ops/test_get_indice_pairs_3d_cpu.py)
is publicly available and explicitly targets 3x3x3 submanifold mapping.

This establishes a real porting attempt, but I found no successful accelerator
benchmark for that sparse path in the reviewed issue and source. Other MapTR
variants and their dense image-backbone results do not establish success for
the sparse LiDAR path. The failed mapping pattern also does not imply that
all scatter operations are unsupported.

For our implementation, these sources offer useful reproduction cases and
index-pair semantics. Keeping coordinate-map construction on the CPU avoids
requiring the compiler to handle the demonstrated failing pattern. The MapTR
feature code still contains per-pair Python loops and `.item()` calls, so it
would need adaptation rather than wholesale reuse as an accelerated backend.

### Related TT projects with different evidence

Tenstorrent's [FusionAD sparse encoder in TT-Polaris](https://github.com/tenstorrent/tt-polaris/blob/2accd8e5f8dc056fe9836be5de257111379efc0b/workloads/FusionAD/projects/mmdet_plugin/models/backbones/sparse_encoder_hd.py)
explicitly approximates sparse 3D convolution with 2D convolution and folds
depth into the batch dimension. [TT-Polaris](https://github.com/tenstorrent/tt-polaris)
is a high-level performance-analysis simulator. This is relevant workload
modeling, not evidence of an exact submanifold-convolution kernel on hardware.

The community [Sparse4D TT port](https://github.com/koreaygj/sparse4D-tt)
reports actual N300 execution and custom projection, sampling, weighted-sum,
and compaction kernels. Those are useful neighboring capabilities, but its
name and detection results alone do not establish a voxel SubMConv3d backend.

### TRELLIS.2 on Apple Silicon: directly relevant

The [MLX sparse convolution implementation](https://github.com/pedronaugusto/trellis2-apple/blob/698fb31113e2f79622c1f6df09cb2a03cf7da953/mlx_backend/sparse_conv.py)
preserves the checkpoint weight layout, caches neighbor indices, gathers a zero
row for missing neighbors, and performs batched matrix multiplication over
kernel offsets. It chunks offsets to limit temporary memory. The accompanying
[VAE decoder](https://github.com/pedronaugusto/trellis2-apple/blob/698fb31113e2f79622c1f6df09cb2a03cf7da953/mlx_backend/vae_decoders.py)
implements the same block families and shape-guided texture subdivision.

This is the most useful high-level reference for a first implementation. One
detail should not be copied for large GenRecon scenes: its neighbor-map builder
allocates a dense spatial lookup table. A sparse CPU hash/sort map avoids that
volume-dependent allocation. Apple unified memory also does not establish the
cost of CPU/TT transfers over PCIe.

The [mtlgemm Metal kernel](https://github.com/pedronaugusto/mtlgemm/blob/c2e5d3fa15152bc4aef65773cab1ded5a1855d38/flex_gemm/kernels/metal/spconv/spconv_gemm.metal)
is the closer reference for a performant custom kernel. It gathers neighbor
features into tiles, accumulates matrix products, and skips offsets that are
empty across an output tile. Its source contains FP32/FP16/BF16 variants and
[parity tests](https://github.com/pedronaugusto/mtlgemm/blob/c2e5d3fa15152bc4aef65773cab1ded5a1855d38/tests/test_masked_vs_dense_parity.py).
These tests are useful starting points; their existence is not independent
verification of the implementation or every claim in its comments.

A downstream maintainer [reports matched decoder measurements](https://github.com/shivampkumar/trellis-mac/issues/1#issuecomment-4291959971)
on an M4 Pro 24 GB, 512-resolution run: shape decoding improved from 17.92 to
7.24 seconds, texture from 19.94 to 6.86 seconds after switching from portable
convolution to the corrected Metal backend. These are third-party self-reported
measurements, not reproduced here, and do not predict TT timings. They are strong
evidence that the actual decoder workload can leave CUDA. The discussion also
documents a useful failure mode: copying every convolution's features through
the CPU had erased the acceleration benefit.

The [pure-PyTorch fallback](https://github.com/shivampkumar/trellis-mac/blob/d58628f4f5b9c3de8274cb110074154f4b31cef2/backends/conv_none.py)
is another useful reference for a CPU correctness implementation. Its nested
Python coordinate loops are unsuitable as a final large-scene map builder.

### FlexGEMM and TorchSparse++: algorithmic references

The original [FlexGEMM implementation](https://github.com/JeffreyXiang/FlexGEMM)
and its author's [technical explanation](https://jeffreyxiang.github.io/en/blogs/flexgemm)
show explicit im2col/GEMM and fused implicit-GEMM alternatives. The latter
avoids writing and rereading the whole expanded feature matrix. The
[TRELLIS.2 paper, Appendix B](https://arxiv.org/html/2512.14692v1#A2)
also describes grouping similar neighbor masks, skipping empty work, and Split-K.
It reports NVIDIA and AMD support; that does not imply a Triton kernel runs
unchanged on Tensix.

[TorchSparse++](https://arxiv.org/abs/2311.12862) is useful for the tradeoff between
simple gather/GEMM/scatter and fused, pipelined kernels. Its GPU performance
results are not portable TT performance predictions. These references guide
algorithm design; Metal/Triton/CUDA kernel code requires a different TT backend.

## New measurements on our Wormhole device

The [probe](sparse_conv_probe.py) loads actual shape-decoder convolution weights.
Inputs and coordinates are synthetic, with two batches, missing neighbors,
boundaries, and active counts not divisible by 32. A separate tiny CPU test
checks both decompositions against dense Conv3d sampled at active coordinates;
the im2col version's maximum absolute difference was 1.91e-6.

| Layer / lowering | Active rows | Tensor dtype | Relative L2 error | Warm latency |
| --- | ---: | --- | ---: | ---: |
| 64 -> 64, im2col | 257 | BF16 | 0.00401 | 1.92 ms |
| 64 -> 64, im2col | 4,097 | BF16 | 0.00247 | 8.63 ms |
| 1024 -> 1024, im2col | 257 | BF16 | 0.00247 | 7.98 ms |
| 1024 -> 4096, im2col | 129 | BF16 | 0.02796 | 6.39 ms |
| 64 -> 64, im2col | 257 | FP32 | 0.00455 | 3.98 ms |
| 1024 -> 4096, im2col | 129 | FP32 | 0.02853 | 10.58 ms |
| 1024 -> 4096, im2col, HiFi3 requested | 129 | BF16 | 0.02801 | 7.54 ms |
| 1024 -> 4096, im2col, HiFi3 requested | 129 | FP32 | 0.02857 | 9.46 ms |
| 1024 -> 4096, 27 offset products, HiFi3 requested | 129 | FP32 | 0.00388 | 29.19 ms |

All nine passed the existing operator smoke gate and reported no XLA fallback
ops. The final alternative reduced the wide layer's error about 7.4-fold, at
roughly three times the corresponding im2col latency. It demonstrates that the
numerical result depends on the lowering, not that the root cause is isolated.

The roughly 2.8% error in the wide im2col layer persisted with FP32 tensors and
an explicit HiFi3 request. The runtime warned about Wormhole HiFi4 with FP32
accumulation; changing the request did not resolve this discrepancy. Do not
assume that tensor dtype alone establishes arithmetic fidelity, or attribute
the observed error solely to that warning.

Measurement conditions: TT-Forge/PJRT 1.4.0, torch 2.11.0+cpu, torch-xla
2.9.0+git56f8e61, one Wormhole B0 chip, CPU reference with 16 threads. Warm
medians cover three calls after warmup, including feature/index H2D and output
D2H transfers with resident weights. Host neighbor construction and first
compilation are excluded from warm timings. BF16 comparisons use FP32 arithmetic
on already BF16-rounded weights and inputs; they do not include the additional
error from converting the original FP16 checkpoint to BF16. FP32 cases load
the original checkpoint values into FP32.

These measurements establish operator execution, not whole-decoder fidelity,
texture-decoder validation, real-scene throughput, or a TT-versus-CPU speedup.
Results and per-run script hashes are retained in
[the JSON summary](results/sparse-vae-feasibility/summary.json). Full local logs
are under `/home/user/tt-scratchpad/genrecon/sparse-vae-study/`; the initial
64-channel BF16 log is one directory above that folder.

## Remaining engineering risks

**Memory and data movement are the leading performance risk.** A global
im2col tensor uses 27 times the input feature storage. At one million active
voxels and 64 channels, BF16 input features occupy about 122 MiB, a 32-bit
27-neighbor map 103 MiB, and expanded features 3.22 GiB. Those exclude output,
weights, allocator overhead, and other live buffers. Densifying a 512-cubed
grid with 64 BF16 channels requires 16 GiB for one feature tensor; at 1024-cubed
it requires 128 GiB. These figures are arithmetic estimates, not measured peaks.

Start with bounded output-row or offset chunks. For a performant implementation,
fuse gather and matrix multiplication so expansion stays in local memory. Keep
feature tensors on device between layers. Each decoder has about 0.95 GB of
two-byte weights, so sequential model residency can help memory use too.
Spatial scene partitioning must preserve neighbor halos; simply splitting voxel
rows and discarding cross-chunk neighbors changes the network. Processing output
row chunks while retaining access to all required input rows is exact.

**Dynamic shapes require explicit boundaries.** Child selection uses nonzero,
repeat-interleave, and compaction. Keep those on the CPU initially, with fixed
capacity buckets for compiled TT graphs or TTNN/custom kernels that manage
varying row counts. Mask padding correctly and key map caches by actual layout,
not just voxel count. Tenstorrent's own
[static-shape multimodal merge adaptation](https://github.com/tenstorrent/tt-xla/blob/fdc3b7773674081b8f8c0698de3400d8d899dc8f/integrations/vllm_plugin/vllm_tt/layers/mm_embeddings.py)
illustrates why dynamic compaction cannot be assumed to pass through the
compiler unchanged. CPU planning is feasible, but its cost at millions of voxels
must be measured. Use sufficiently wide coordinate keys for merged scenes.

**Accuracy must include discrete decisions.** A good PCC can coexist with child
logits crossing zero. Once the active sets differ, later errors are no longer
ordinary small feature perturbations. Validate subdivision masks at every
transition, geometry edge signs, vertex positions, and texture outputs. Use
fixed reference guides to isolate feature errors before allowing shape topology
to evolve independently. Inspect near-zero logits and select higher precision
or a different lowering where necessary; the wide-layer experiment shows why
this matters. Merely evaluating the final eight-logit head in FP32 may not fix
errors accumulated in its input features.

**Library support is narrower than mathematical feasibility.** A prototype can
use TT-XLA tensor operators, as demonstrated here. Efficient TTNN/Metalium
integration, layout conversion, tile packing, L1 allocation, and launch overhead
need separate work. There is no evidence yet that generic gather plus GEMM is
fast enough at actual scene sizes. Training/backward support is unnecessary for
the requested inference port.

## Recommended implementation sequence and decision gates

1. Implement a CPU sparse-convolution reference and coordinate planner, preserving
   checkpoint layout and exact channel-to-space ordering. Decouple neural
   imports from O-Voxel. Test boundaries, empty neighborhoods, multiple batches,
   empty child sets, and map reuse.
2. Implement one complete TT ConvNeXt block and one C2S block with CPU compaction.
   Test real decoder widths and activation distributions. Resolve the wide
   convolution discrepancy before accepting full-network numerics.
3. Run the complete shape decoder with fixed reference subdivision guides,
   then with its own predicted guides. Validate topology decisions and geometry,
   not only tensor correlation. Reuse the validated backend and shape maps for
   texture decoding, then validate material outputs.
4. Profile realistic GenRecon chunks: per-stage voxel counts, neighborhood
   occupancy, peak memory, map construction, transfers, compilation, and warm
   execution. Sweep representative sparse patterns as well as N and C.
5. Optimize measured bottlenecks with streaming/fused gather-GEMM, cached maps,
   shape buckets, and safe scene grouping. Only add device-side hashing or
   compaction if host planning is demonstrably significant.

The next useful milestone is a correct full sparse decoder on a real latent
chunk, followed by bounded-memory scene execution. The existing evidence is
sufficient to justify that implementation effort. It is insufficient to promise
a particular speedup, completion schedule, or reconstruction-quality match.
