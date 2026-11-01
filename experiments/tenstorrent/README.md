# GenRecon on Tenstorrent: conversion experiments

The flow experiments use the existing TT-Forge 1.4.0 TT-XLA frontend
(`torch.compile(..., backend="tt", fullgraph=True)`). The new complete sparse
decoder adapter uses TTNN and FlexGEMM directly in a separate runtime process.
These do not yet form an end-to-end image-to-mesh GenRecon implementation. No
NVIDIA device is used. SfM/COLMAP and training are outside this investigation.

## Current outcome

All three complete pretrained flow networks execute on TT without reported XLA
fallback in these fixed-chunk, synthetic-input tests. The sparse flows use the
original network forward, explicit modulation broadcasting and CPU-prepared
RoPE/layout caches. **A complete image-to-mesh GenRecon pipeline is not yet
running on TT.**

| Component | Current result |
| --- | --- |
| Occupancy flow, all 30 blocks, 4096 latent tokens | Pass; 16.58 s warm median; relative L2 0.08665 against FP32 CPU |
| Shape flow, all 30 blocks, 512 active voxels | Pass; 2.35 s; relative L2 0.06822 against FP32 CPU |
| Texture flow, all 30 blocks, 512 active voxels | Pass; 2.34 s; relative L2 0.07984 against FP32 CPU |
| Three trained multi-view aggregators | Pass, no reported fallback |
| Complete pretrained occupancy decoder | Pass in FP32, no reported fallback |
| Camera projection / feature gathering | All-TT version has index/mask mismatches; CPU indices + TT gather/mask passes |
| Euler/CFG and overlap arithmetic | Tensor-boundary probes pass; orchestration/index preparation remains on host |
| Shape and texture sparse VAE decoders | Complete TTNN/FlexGEMM integration; synthetic parity passes. Real-image fixture executes through 1.71M voxels, but a near-zero subdivision decision breaks exact topology parity; see [integration](SPARSE_DECODER_INTEGRATION.md) |
| Mesh extraction, processing and export | Original CUDA-dependent path needs CPU/TT replacements |

The full flows' roughly 7–9% relative error passes the declared smoke gate;
it does **not** establish reconstruction quality. GenRecon's DINO variant on TT,
real image conditions and multi-step scene generation **on TT**, visual scene
quality, and end-to-end image-to-mesh integration remain unverified. The exact
DINOv3-L/512 configuration and all three samplers have run on CPU to generate a
real-image decoder fixture; both TT decoders have now been evaluated on it.
The sections below retain both successful results and the failures that defined
the working compilation boundaries.

A [follow-up sparse VAE feasibility study](SPARSE_VAE_FEASIBILITY.md) inspects the
decoder architecture and non-CUDA prior art, and tests individual pretrained
sparse convolutions through gather/GEMM on Wormhole. These operator probes pass
without reported fallback. The subsequent [decoder integration](SPARSE_DECODER_INTEGRATION.md)
implements the complete networks with resident features, an explicit accuracy
mode, and numerical/topology validation. [Required real-scene data and a CPU
fixture-generation path](../../docs/required-validation-data.md) are documented separately.

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

At this initial stage, the original CUDA-extension dependency paths prevented
constructing/running the shape/texture decoders. These were dependency failures,
**not compiler rejections of decoder math**. The subsequent FlexGEMM integration
replaces the neural decoder dependencies; O-Voxel mesh processing remains
separate porting work.
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

## Complete pretrained shape flow and native sparse path

The complete **30-block shape flow** also passes with 512 active voxels and the
official checkpoint. Both implementations below were executed, with no reported
fallback and an FP32 CPU reference; the tensor adapter additionally checked
equivalence to the original full CPU sparse network before device execution.

| Pretrained probe | PCC / relative L2 | TT warm median | Evidence |
| --- | --- | --- | --- |
| Shape aggregator | 0.999970 / 0.01231 | 12.29 ms | `pretrained-shape-fp32ref` |
| Full shape flow, dense active-token wrapper | 0.997706 / 0.06808 | 2.23 s | `pretrained-shape-fp32ref` |
| Full shape flow, **original sparse forward with cached RoPE** | 0.997756 / 0.06822 | 2.35 s | `native-cached-rope` |

The native sparse path became traceable after expressing modulation/gating on
`h.feats` with explicit batch broadcasting. Its sparse DiT initially executed
with **`aten::polar.out` CPU fallback** (`sparse-explicit-broadcast`), despite
`fullgraph=True`. Preparing the RoPE phases once on CPU and registering them in
the sparse spatial cache removes this fallback. The native block and complete
native shape flow then pass with empty fallback lists (`native-cached-rope`).
Ragged two-sample CPU tests cover both per-sample and shared modulation so this
change preserves batch broadcasting. The default FlashAttention backend is
still unchanged; these probes explicitly select SDPA.

The native shape forward loaded 1,352,548,897 parameters (including its unused
aggregator), took 262.4 s for compilation/first inference, and used 3.56 s for
its FP32 CPU reference. The tensor adapter remains an experimental comparison;
the successful native route does not replace sparse blocks with dense blocks.
Both use fixed coordinates/layout and synthetic conditions. Arbitrary scene
layouts, compilation reuse across varying token counts and scene quality are
not established by these results.

## Complete pretrained texture flow

The complete **30-block texture flow** passes using the original sparse forward,
explicit modulation broadcasting and cached CPU-prepared RoPE/layout. The input
is `[512,64]` (preconcatenated texture/shape features), with `[1,1029,1024]` image
condition and `[512,1024]` 3D condition; output is `[512,32]`. This is a synthetic
single-chunk forward with all checkpoint keys loaded strictly. No XLA fallback
operations or `aten::` fallback counters were reported.

| Pretrained probe | PCC / relative L2 against FP32 CPU | TT warm median | Evidence |
| --- | --- | --- | --- |
| Texture aggregator | 0.999932 / 0.01577 | 11.84 ms | `pretrained-tex-native` |
| Full native texture flow with cached RoPE | 0.997025 / 0.07984 | 2.34 s | `pretrained-tex-native` |

The texture flow loads 1,352,598,049 parameters (including the separately tested
aggregator). Compilation/first inference took 237.2 s; the FP32 CPU reference
took 3.48 s. As with shape and occupancy, this establishes module execution and
the stated numerical gate, not end-to-end texture or scene quality.

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
  --cases ss_flow shape_aggregator shape_flow_cached \
    tex_aggregator tex_flow_cached \
  --cpu-threads 16 --reference-precision float32 --timeout 1200 \
  --output-dir "$HOME/tt-scratchpad/genrecon/pretrained-reproduction"
```

For these full networks, **CPU reference precision differs from TT precision**.
The EPYC 7352 host lacks native BF16 support. A diagnostic matmul
`[512,1536] @ [1536,8192]` took 30.54 s in BF16 versus 0.0374 s in FP32 with
16 threads. Two native-BF16 occupancy reference attempts were deliberately
interrupted before TT compilation; these are not compiler failures. The practical
reference promotes the loaded, already quantized weights and activations to
FP32, then restores every model dtype before TT inference. The TT model uses
the source's explicit mixed-precision path: FP32 input/output layers and latent
inputs, BF16 torso linears, and autocast disabled. The original CUDA pipeline
also uses CUDA autocast and low-precision noise; a TT scene integration must
handle those input/output dtypes explicitly. Reported error therefore includes precision
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

- Validate real projected conditions, occupancy threshold decisions, multi-step
  sampling and scene quality on TT. Flow probes still use synthetic conditions;
  decoder probes now include a CPU-generated real-image fixture.
- Integrate explicit input/output dtypes, CPU-prepared coordinate/RoPE caches,
  variable token counts and device transfers into the scene pipeline. The exact
  DINOv3 ViT-L/512 configuration still needs verification.
- Assess the decoder's real-input topology differences across more scenes and
  tune latency; the accurate mode currently offers no CPU speedup on the poster
  fixture. Replace the remaining mesh-processing extensions for a complete pipeline.

Reference: [TT-XLA documentation](https://docs.tenstorrent.com/tt-xla/) and
[performance/precision options](https://docs.tenstorrent.com/tt-xla/performance.html).
