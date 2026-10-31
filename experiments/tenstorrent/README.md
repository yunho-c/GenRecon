# GenRecon on Tenstorrent: conversion experiments

These are module-level inference experiments using the existing TT-Forge 1.4.0
TT-XLA frontend (`torch.compile(..., backend="tt", fullgraph=True)`). They are
not an end-to-end GenRecon implementation. No NVIDIA device is used. SfM/COLMAP
and training are outside this investigation.

## Environment and method

- Device: one Wormhole B0 chip, PCIe device 3 (cloud environment describes n150).
- Python 3.12; PyTorch `2.11.0+cpu`; torch-xla `2.9.0+git56f8e61`;
  tt-forge / pjrt-plugin-tt `1.4.0`.
- Upstream GenRecon source: `eaf1468118d20469d17079a4a19737297d2ef87b`.
- Each probe runs in a separate process, computes a CPU reference, requests a full
  graph, executes on the TT PJRT device, copies outputs to CPU to synchronize,
  and records numerical error, timing and XLA fallback counters.
- Numerical smoke gate: finite outputs, PCC >= 0.99 for nonconstant tensors,
  relative L2 error <= 0.1; boolean/integer outputs must match exactly. This is
  an operator-coverage gate, not a reconstruction-quality acceptance criterion.
- Timings include fresh input transfer, inference and output transfer, with
  weights resident. Three measured calls follow first inference and one warmup.
- Synthetic inputs are explicitly labeled. Random-weight probes use nonzero
  learned paths; in particular, the aggregation MLP's normally zero-initialized
  final layer is randomized to avoid a trivial mean-only test.

The user's prior DINO result is external to this repository:
`~/tt-scratchpad/dinov3-pretrained-check/result.json`. It reports pretrained
**ViT-B at 224x224**, PCC 0.99961 versus FP32 CPU, and no fallback ops. GenRecon
uses **ViT-L at 512x512**; that exact integration is not established by the B/224
result and is not counted as verified here.

## Initial results (2026-10-31)

Machine-readable evidence is under `results/initial/`. Full local compiler logs
are in `~/tt-scratchpad/genrecon/initial/<case>/run.log`.

| Probe | Scope | Outcome | PCC / relative L2 | Warm median |
| --- | --- | --- | --- | --- |
| Global image-feature projection | 8 views, 256 points, 1029 tokens, 1024 features; FP32 | Compiles and executes, **numerical mismatch**, including visibility mask | 0.98258 / 0.18690 | 28.18 ms |
| Dense multi-view aggregator | Actual module, random weights, 8 views x 256 voxels x 1024 channels; FP32 | Pass, no reported fallback | 0.999998 / 0.002262 | 12.86 ms |
| Sparse multi-view aggregator | Actual module, tensor wrapper with explicit sparse shape; same dimensions; FP32 | Pass, no reported fallback | See JSON | 16.50 ms |
| Sparse-structure decoder | **Official pretrained weights**, full 73,671,201-parameter network, latent `[1,8,16,16,16]` -> logits `[1,1,64,64,64]`; FP32 | Pass, no reported fallback | 0.999847 / 0.002608 | 2960.44 ms |
| DiT block | Actual dense cross-attention block, width 1536, 12 heads, 64 latent tokens, 1029 image tokens; random weights | Dynamo graph capture fails at `globals()` in SDPA dispatch | Not executed | — |

The decoder CPU reference took 2.98 s with four threads; the current TT run is
about the same latency. Successful conversion is not a speedup claim. The
decoder's occupancy threshold decisions and generated-scene quality still need
validation with real latents. No full-scene peak-memory claim follows from this
single-chunk experiment.

## Attention, projection and orchestration follow-up

The following results use the compatibility changes in this branch. Each named
directory under `results/` contains the corresponding raw summary, source-file
hashes, input shapes, numerical errors, fallback counters and failure messages.
All successful rows below reported **no XLA fallback operations**.

| Probe | Result | Evidence directory |
| --- | --- | --- |
| Dense DiT block, including complex RoPE | Pass after moving SDPA import out of `globals()` dispatch; PCC 0.999937, relative L2 0.01208, 21.61 ms | `sdpa-dispatch` |
| Native sparse attention | Original FlashAttention path fails CPU reference because `flash_attn` is unavailable | `dependencies` |
| Sparse attention using PyTorch SDPA | Pass; PCC 0.999759, relative L2 0.02567, 4.53 ms | `sparse-sdpa` |
| Native sparse DiT block | Tuple cache keys remove one capture failure, but TT overrides still fail on `SparseTensor * Tensor` | `sparse-sdpa`, `sparse-cache-key` |
| Sparse block expressed as one dense **active-token sequence** | Same state dict, CPU equality checked against native sparse block; TT PCC 0.999932, relative L2 0.01262, 25.81 ms | `boundaries` |
| All-TT projection, elementwise camera transforms | Still mismatches; 7/2048 visibility decisions differ. This experiment does not establish the root cause | `projection-elementwise` |
| CPU camera projection + TT feature gather/masking | Exact match in this fixture; 16.76 ms for the TT portion | `projection-host-indices` |
| Euler/CFG step with guidance rescale | Original `sample_once` fails on `EasyDict` construction. Tensor-return wrapper using the same numerical functions passes: PCC 1.0, relative L2 <= 7.4e-6, 11.83 ms | `boundaries`, `euler-tensor` |
| Native dense overlap aggregation | Graph capture fails on data-dependent bounds derived through `.tolist()` | `boundaries` |
| CPU overlap indexing + TT scatter/average/gather | Matches native sparse aggregation on CPU, passes on TT; PCC 0.999997, relative L2 0.002262, 2.66 ms | `boundaries` |
| Shape decoder and mesh exporter | Stop before compilation: O-Voxel's compiled `_C` extension is unavailable | `boundaries`, `dependencies` |
| Texture decoder | Stops before compilation: `flex_gemm` is unavailable | `dependencies` |

The source modifications are deliberately small: a traceable dense SDPA import,
an opt-in sparse SDPA backend that attends each ragged sample separately, and
tuple-valued spatial-cache keys. Five CPU tests verify ragged sample isolation,
the packed/separate QKV calling conventions, and cache separation by scale.
The existing default sparse backend is unchanged. The tensor adapters live only
under `experiments/tenstorrent`; none is installed into the scene pipeline.

The sparse block adapter keeps only the active voxels and does **not** expand to
a dense 3D grid. It establishes a useful compiler boundary for one fixed chunk,
not arbitrary ragged batching or full-scene memory/performance. The sampler probe
supplies positive/negative velocities directly; it is not a learned denoiser or
a full sampling trajectory. The overlap probe uses two 64-voxel chunks with a
32-voxel overlap; boundary-sensitive averaging is not covered.

The shape/texture decoder checkpoints have been downloaded, but their original
CUDA-extension dependency paths prevent constructing/running them here. These
are dependency failures, **not compiler rejections of decoder math**. Replacing
those extensions and O-Voxel mesh processing remains separate porting work.
Specifically, sparse convolution dispatch calls
`flex_gemm.ops.spconv.sparse_submanifold_conv3d`; O-Voxel builds a
`torch.utils.cpp_extension.CUDAExtension`, and its exporter also imports CuMesh,
nvdiffrast and FlexGEMM grid sampling and explicitly calls `.cuda()`.

## Complete pretrained occupancy flow

The **full 30-block occupancy denoiser** executes on TT with no reported XLA
fallback. This includes the timestep embedding, trained LoRA paths, learned
per-block 3D-condition projections, self/cross attention and output head.
It uses the official checkpoint with strict loading, 1,352,475,145 parameters
loaded (including the aggregator, which is tested separately), latent input
`[1,8,16,16,16]`, image condition `[1,1029,1024]` and 3D condition
`[1,4096,1024]`. Inputs are synthetic and conditions are supplied directly.

| Pretrained probe | Reference | PCC / relative L2 | TT warm median | Evidence |
| --- | --- | --- | --- | --- |
| Occupancy aggregator | Native BF16, 4 CPU threads | 0.999978 / 0.01003 | 11.00 ms | `pretrained-ss` |
| Complete occupancy flow | FP32, 16 CPU threads; TT remains pipeline mixed precision | 0.997091 / 0.08665 | 16.58 s | `ss-flow-math-log` |

Full occupancy compile + first inference took 233.9 s; the FP32 CPU reference
took 30.79 s. These timings use different compute precision and are not a
controlled speedup benchmark. The 8.7% relative L2 discrepancy passes the
declared smoke gate but is substantial: real-condition, multi-step and scene
quality validation remains necessary. This is one full network forward, not
an end-to-end reconstruction.

The first full-graph attempt failed on NumPy's timestep scalar; its evidence is
in `pretrained-ss-fp32ref`. The two interrupted native-BF16 CPU reference attempts
are retained under `pretrained-ss` and `pretrained-ss-threads16` with explicit
notes. Neither is counted as a TT compiler failure.

## Reproduce

Use the same environment that passed DINO; do not install the CUDA setup script.
The runner clears shell variables that would select a different TT-Metal runtime.

```bash
hf download microsoft/TRELLIS-image-large \
  --revision 25e0d31ffbebe4b5a97464dd851910efc3002d96 \
  --include 'ckpts/ss_dec_conv3d_16l8_fp16.*' \
  --local-dir "$HOME/models/genrecon/trellis-image-large"

hf download microsoft/TRELLIS.2-4B \
  --revision af44b45f2e35a493886929c6d786e563ec68364d \
  --include 'ckpts/shape_dec_next_dc_f16c32_fp16.*' 'ckpts/tex_dec_next_dc_f16c32_fp16.*' \
  --local-dir "$HOME/models/genrecon/trellis2"

/home/user/tt-forge-venv/bin/python experiments/tenstorrent/run.py \
  --output-dir "$HOME/tt-scratchpad/genrecon/new-run" --timeout 300
```

Every run needs a fresh output directory. `result.json` preserves the last phase
even after a Python exception, child-process crash or timeout. The parent returns
nonzero if any case fails, differs numerically, or uses reported fallback ops.
Inspect each result, not just a compiler success message.

Additional probes and CPU regression checks:

```bash
/home/user/tt-forge-venv/bin/python -m pip install easydict==1.13
/home/user/tt-forge-venv/bin/python -m unittest \
  experiments.tenstorrent.test_sdpa experiments.tenstorrent.test_flow_adapter -v
/home/user/tt-forge-venv/bin/python experiments/tenstorrent/run.py \
  --cases dit_block sparse_attention_sdpa sparse_dit_block sparse_dit_tensor \
    projection_cpu_indices euler_cfg_tensor overlap_dense_stock overlap_cpu_indices \
    shape_decoder_stock tex_decoder_stock mesh_export_stock \
  --output-dir "$HOME/tt-scratchpad/genrecon/follow-up" --timeout 300
```

Expect a nonzero exit for the deliberately included blocked native paths.

Official GenRecon flow weights can be downloaded independently:

```bash
python experiments/tenstorrent/fetch_checkpoints.py \
  --destination "$HOME/models/genrecon"
```

This defaults to eight concurrent HTTP range requests (`--workers 16` was used
to finish the slow university-host download), validates byte ranges against the
source size/ETag, retains completed chunks for resume, and records SHA256.
Weights and large compiler artifacts are kept outside the repository.

Full pretrained flow probes use the source configs and strict checkpoint
loading, including the trained LoRA, aggregation and per-block conditioning
weights. `ss_flow` uses all 4096 latent tokens. `shape_flow_tensor` and
`tex_flow_tensor` use one fixed chunk with 512 active voxels and all 30 blocks;
texture inputs are preconcatenated shape/texture features. Conditions are
synthetic, precomputed tensors, not images from an end-to-end reconstruction.

```bash
/home/user/tt-forge-venv/bin/python experiments/tenstorrent/run.py \
  --cases ss_flow shape_aggregator shape_flow_stock shape_flow_tensor \
    tex_aggregator tex_flow_stock tex_flow_tensor \
  --cpu-threads 16 --reference-precision float32 --timeout 1200 \
  --output-dir "$HOME/tt-scratchpad/genrecon/pretrained-reproduction"
```

For these full networks, **CPU reference precision differs from TT precision**.
The EPYC 7352 host lacks native BF16 support. A diagnostic matmul
`[512,1536] @ [1536,8192]` took 30.54 s in BF16 versus 0.0374 s in FP32 with
16 threads. Two native-BF16 occupancy reference attempts were deliberately
interrupted before TT compilation; these are not compiler failures. The practical
reference promotes the loaded, already quantized weights and activations to
FP32, then restores every model dtype before TT inference. The TT model retains
the pipeline's mixed precision. Reported error therefore includes precision
differences as well as backend differences. CPU tests check exact restoration
of weights/dtypes and preservation of complex RoPE buffers and LoRA behavior.

The full sparse-flow tensor adapters also check CPU equivalence against the
native sparse models in FP32 before attempting TT compilation. Their small
regression fixture additionally checks native mixed-precision equivalence.
Full-graph capture required replacing `np.log(max_period)` in the timestep
embedding with `math.log(max_period)`; this matched the original embedding
exactly for the tested timesteps and even/odd dimensions.

Concurrent harness invocations serialize device use with a process-lifetime
file lock. CPU reference work and downloads can overlap; TT workers cannot.

## Remaining investigation

- Finish the texture checkpoint download and full pretrained shape/texture
  denoiser and aggregator probes.
- Check the tensor-only flow adapters against the original complete CPU models
  before compiling pretrained full networks. Block results do not establish that
  all 30 layers fit or compile together.
- Validate real projected conditions, occupancy threshold decisions, multi-step
  sampling and scene quality. Current numerical gates use synthetic inputs.
- Implement/replace the sparse convolution and mesh-processing extensions for
  a complete pipeline; automated conversion has not supplied these backends.

Reference: [TT-XLA documentation](https://docs.tenstorrent.com/tt-xla/) and
[performance/precision options](https://docs.tenstorrent.com/tt-xla/performance.html).
