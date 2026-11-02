# GenRecon on Tenstorrent: conversion experiments

The flow experiments use the existing TT-Forge 1.4.0 TT-XLA frontend
(`torch.compile(..., backend="tt", fullgraph=True)`). The new complete sparse
decoder adapter uses TTNN and FlexGEMM directly in a separate runtime process.
These do not yet form an end-to-end image-to-mesh GenRecon implementation. No
NVIDIA device is used. SfM/COLMAP and training are outside this investigation.

## Current outcome

The [real-scene staged pipeline](SCENE_PIPELINE_MEASUREMENTS.md) now completes
DINO, all three original 12-step samplers and both VAE decoders on one n150
Wormhole chip, producing matched raw geometry/material outputs. Its measured
warm components sum to **21.85 minutes** for one 512-resolution poster chunk
with 6,181 active latent voxels and 2.40 million decoded voxels. Sampling already
includes all 61 guidance evaluations. Mesh extraction/export, n300 execution and
the default 1024 cascade remain unmeasured/unintegrated.

The earlier synthetic results below use the native sparse flow boundary. At
6,181 real-scene tokens that boundary runs out of memory; the successful scene
run uses the existing active-token sequence adapter, which matches the original
sparse model exactly in FP32 CPU at the full scene size. All learned stages run
on TT, with host projection/sampling/topology and isolated TT-XLA/TTNN processes.

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

The synthetic flows' roughly 7–9% relative error passes the declared smoke gate;
it does **not** establish reconstruction quality. The subsequent real-scene run
measures about 3.3–3.7% first-forward flow error against FP32 CPU with matched
inputs. Full-trajectory quality, visual scene quality, exact topology parity
and end-to-end image-to-mesh integration remain unverified. DINOv3-L/512 and
the complete samplers now run on TT in the staged harness.
The ordinary [BF16 decoder path](BF16_SCENE_VALIDATION.md) now has full-fixture
warm medians of 15.09 seconds for shape and 14.63 seconds for texture, with
substantial additional topology/geometry differences from the FP32 reference.
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

The user's earlier DINO result is external to this repository:
`~/tt-scratchpad/dinov3-pretrained-check/result.json`. It reports pretrained
**ViT-B at 224x224**, PCC 0.99961 versus FP32 CPU, and no fallback ops. GenRecon
uses **ViT-L at 512x512**. That exact GenRecon feature path has since been validated
on TT, including all eight posed images in the staged scene experiment: 1.55 s
total warm encoder execution and 1.35% relative L2 against saved FP32 features.

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

Compiler-performance experiments can explicitly select `--optimization-level
0`, `1`, or `2`, and optionally `--runtime-trace` (with
`--trace-region-size`, default 10 MB). These options are passed to each isolated
worker before compilation. Reports preserve the submitted XLA options; omitted
settings retain the SDK/environment defaults and are not evidence of an effective
optimization level. Compare identical cases, seeds, precision, CPU threads and
timing boundaries in fresh directories. Trace capture happens during warmup.
Neither compiler success nor an operator smoke pass establishes scene quality.

Frozen inference LoRA candidates are available with `--lora-mode merged`,
`lowrank`, or `frozen_delta`; the default remains `stock`. Preparation happens
after strict checkpoint loading, dtype conversion and the **unmodified** CPU
reference. Both candidate CPU drift and total device error against that stock
reference are recorded. Merging and low-rank evaluation change rounding and
are experimental, with no scene-quality acceptance implied. `frozen_delta`
retains the two full-width projections but prepares the delta weight once.
CPU tests exercise nonzero LoRA self/cross attention in FP32 and BF16.

Complete shape/texture flow probes also accept `--active-voxels 6181` to test a
larger synthetic active-token layout. This supplies synthetic noise, image/3D
conditions and grid coordinates; matching a scene's token count does not make
it a real-scene or reconstruction-quality benchmark.

New staged scene runs enforce the local numerical smoke gates before reporting
completion. This still does not validate the full sampled trajectory or mesh.
`scene_pipeline.py --resume` now requires identical source/configuration content,
checkpoint contents and execution settings, and verifies recorded stage inputs
and outputs. Checkpoint fingerprinting is outside timed inference. Legacy reports
without this identity require a fresh output directory; historical results are
retained as originally measured.

`codegen_probe.py export --case shape_flow_tensor --directory /path/to/new-export`
runs under the Forge Python and preserves the generated TTNN program, compiler
IR, tensor hashes and stock FP32 reference. Run `codegen_probe.py execute
--directory /path/to/export` under a separate Metal Python to validate and time
the generated program. `--profile-markers` emits Tracy signposts that distinguish
first inference, warmup and measured calls. The ordinary probe runner also
supports these signposts.

The installed Forge wheel rejects device profiling because its Metal runtime
was built without Tracy. The separate Metal build has Tracy enabled. Generated
TTNN execution there is an **alternate boundary**, with potentially different
numerics and host dispatch behavior; it is not a measured replacement for PJRT.
Use it for lowering inspection and device profiling, retain both runtimes'
results, and validate before attributing the original sampler's wall time.
Clear inherited `LD_LIBRARY_PATH`/`PYTHONPATH` before launching the Forge Tracy
wrapper: otherwise it can load incompatible libraries from the separate SDK.

The first [profiling infrastructure evidence](results/autoresearch/003-profiling-infrastructure/summary.json)
includes a representative `.tracy` capture, compressed operations CSV, marked
warm-call summary, and complete shape-flow generated program/TTNN IR. Use
`profile_report.py --csv /path/to/ops_perf_results.csv --output /path/to/new.json`
to select `GENRECON_MEASURED_0` (or `--window` for another measured call).
It rejects incomplete device durations and preserves operation/input-shape
groupings. Its percentages describe sums of operation kernel durations;
they are not a decomposition of end-to-end wall time.

The [complete 512-token shape profile](results/autoresearch/004-shape-profile-512/summary.json)
measures 4,943 device operations. Matrix products are 6.16% of summed kernel
duration; the largest shape groups correspond to the complex-pair RoPE layout
with a last dimension of one or two. This prompted the experimental
`--rope-mode real_pairs` probe option. It retains adjacent pair ordering and
FP32 arithmetic but computes `x*cos + swap_adjacent(x)*signed_sin` without a
`[..., head_dim/2, 2]` intermediate. Fixed sine/cosine tables are shared across
blocks. Preparation occurs after the stock CPU reference; reports include CPU
preparation drift, and TT outputs are still compared against the stock model.
The production implementation and probe default remain unchanged. Supported
boundaries are dense fixed-layout flows/blocks and the active-token sequence
adapter, with inference-only weights and immutable coordinates/phases.
The same `--rope-mode real_pairs` option is available when exporting with
`codegen_probe.py`; it retains the unmodified stock CPU reference and records
preparation drift before generating an independently validated TTNN program.

Complete tensor-flow probes support `--synthetic-trajectory` together with
`--reference-precision float32`. This runs the original 12-step Euler/CFG
schedule against the unmodified CPU model before preparing any candidate, then
executes the same noise and conditioning on TT. Dense occupancy, single-chunk
shape and preconcatenated texture are supported. Texture shape channels remain
fixed throughout sampling. Reports check every predicted state and final latent,
record graph-counter changes, and save `trajectory-reference.pt` and
`trajectory-tt.pt` with hashes. New artifacts also fingerprint both conditioning
branches, concatenated shape features and sampler configuration. Explicit
legacy-artifact inspection is diagnostic and reports missing identity checks.
These are trajectory smoke checks on synthetic conditions, not mesh quality.

`precision_reference.py --stage shape --active-voxels 512 --output-dir /path/to/new`
adds a CPU-only mixed-precision diagnostic. It preserves BF16 tensor boundaries
but evaluates BF16 GEMMs and CPU flash attention through FP32 accumulation,
avoiding this host's slow native BF16 products. Small native BF16 comparisons
test the emulation. This is not a CUDA/TT kernel oracle and does not replace
the original FP32 reference or acceptance gates.

The [optimized full-flow profile](results/autoresearch/011-real-rope-profile/summary.json)
retains both the failed original export execution and a validated compatibility
repair. For this real-RoPE export, `codegen_probe.py execute
--repair-embedding-indices` restores the CPU-hoisted unsigned index dtype that
native IR declares but emitted Python omits. It verifies the exact 128 adjacent
pair indices before applying the cast, records the repair, and leaves generated
files and learned arithmetic unchanged. The option is off by default.

Both runners accept an explicit `--math-fidelity` compiler option. Omission
retains the SDK default. The [HiFi3 trajectory experiment](results/autoresearch/012-hifi3-trajectory/summary.json)
failed the same numerical gates and increased final relative L2 error from
0.383 to 0.422 against the identical stock FP32 CPU trajectory. It is not an
accuracy fix; defaults remain unchanged.

The [matched larger-flow measurements](results/autoresearch/014-matched-larger-flow/summary.json)
confirm 23.51 s stock versus 6.62 s real-pair RoPE for a 6,181-token shape
forward on PJRT (3.55x). The texture candidate passes its local forward smoke
at 0.629 s for 512 tokens. These use synthetic conditioning; the failed shape
trajectory checks still prevent a quality-preserving optimization claim.

`--rope-mode real_pairs` also supports the fixed-coordinate native sparse
`shape_flow_cached` / `tex_flow_cached` probes. This replaces only sparse RoPE;
original attention, QKV packing, LoRA and sparse block operations remain in
place. It requires one prepared chunk and is not a dynamic sparse runtime.
CPU regression checks cover full sparse blocks in FP32 and BF16 with nonzero
LoRA, including preserved coordinates and layout.

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

Restore the pinned public poster inputs on a fresh machine with:

```bash
/home/user/tt-forge-venv/bin/python experiments/tenstorrent/restore_poster_inputs.py \
  --source-dir "$HOME/tt-scratchpad/genrecon/poster-fixture/source" \
  --report "$HOME/tt-scratchpad/genrecon/poster-fixture/restoration.json"
```

The script downloads the eight selected images and COLMAP inputs, preserves
all 100 published camera frames, and verifies source hashes and view selection.
Use `source/transforms_available.json` for the filtered camera file. The
[restoration evidence](results/autoresearch/013-poster-restoration/summary.json)
also verifies the original crop and official occupancy-decoder hashes. DINO
features and sampled latents are separate prerequisites. In the Forge CPU
environment, a CUDA-linked torchvision wheel caused `torchvision::nms` import
failure; replacing it with the matching `torchvision==0.26.0+cpu` wheel from
`https://download.pytorch.org/whl/cpu` restored preprocessing.

For a separately identified DINO checkpoint from the pinned public timm release,
install `timm==1.0.30` in the Forge environment and run `convert_dino_timm.py
--help`. The tool downloads the pinned weights and license, checks their hash,
maps every learned tensor with strict loading, restores zero Q/V biases, and
requires the official reference-image smoke plus full 512-pixel feature
comparisons against timm on supplied images. The
[poster conversion evidence](results/autoresearch/015-dino-public-conversion/summary.json)
records agreement on all eight views. Use its output through the fixture
generator's explicit `--dino` argument. This is validated for unmasked GenRecon
inference; the missing mask token is a zero placeholder, and the original gated
checkpoint file hash is not reproduced. Historical scene outputs still need a
fresh matched reference.

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

- Extend the successful single-cube staged run to a unified public pipeline,
  actual n300 device placement/parallelism, and the default 1024 cascade.
- Validate full sampled trajectories and scene quality across more scenes.
  Local numerical checks and successful execution do not establish quality;
  occupancy and VAE threshold differences remain.
- Investigate the native sparse feature layout's real-size allocation failure;
  the active-token sequence adapter is the measured working boundary.
- Assess the decoder's real-input topology differences across more scenes and
  tune latency; the accurate mode currently offers no CPU speedup on the poster
  fixture. Replace the remaining mesh-processing extensions for a complete pipeline.

Reference: [TT-XLA documentation](https://docs.tenstorrent.com/tt-xla/) and
[performance/precision options](https://docs.tenstorrent.com/tt-xla/performance.html).
