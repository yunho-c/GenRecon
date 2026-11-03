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

Generated execution preloads activations before the marked forward. The bridge
now synchronizes after loading, records preparation separately, and times the
resident-activation forward plus synchronized output D2H. This differs from
`run.py`'s fresh-transfer PJRT timing. A
[scope correction](results/autoresearch/024-codegen-timing-scope/summary.json)
identifies earlier generated reports whose labels incorrectly included input
transfer; raw data is retained, and pending upload work may have crossed their
old boundary because there was no explicit preload barrier.

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
For the same real-poster occupancy fixture, use `precision_reference.py --stage
ss --flow-inputs /path/to/poster-inputs/inputs.pt --output-dir /path/to/new`.
This calibrates intended tensor-rounding drift before assigning all observed
TT/FP32 trajectory differences to backend kernels. It remains a CPU diagnostic.

`block_diagnostic.py --output-dir /path/to/new-diagnostic` compares all shape-flow
block outputs against stock FP32 and CPU pipeline-precision diagnostics. It
accepts attribution only when the captured final TT output is bitwise equal to
an ordinary compiled forward and every captured state repeats exactly. The
[first full diagnostic](results/autoresearch/026-block-diagnostic/summary.json)
passes those checks: first-block error is already about 2%, while final hidden
error is 1.97% and prediction error is 6.69%. Error does not grow monotonically
across blocks. This localizes the next investigation; it changes no numerical
gates and establishes no sampled quality or speed improvement.

The [optimized full-flow profile](results/autoresearch/011-real-rope-profile/summary.json)
retains both the failed original export execution and a validated compatibility
repair. For this real-RoPE export, `codegen_probe.py execute
--repair-embedding-indices` restores the CPU-hoisted unsigned index dtype that
native IR declares but emitted Python omits. It verifies the exact 128 adjacent
pair indices before applying the cast, records the repair, and leaves generated
files and learned arithmetic unchanged. The option is off by default.

For large generated profiles, `codegen_probe.py execute
--flush-device-profiler` reads and resets device profiler buffers after each
forward, outside the marked timing window. It requires an instrumented SDK.
Also size Tracy's `--op-support-count` for all programs within one forward
(10,000 is the initial large-shape retry setting). The default count of 1,000
[overflowed on the 6,181-token profile](results/autoresearch/016-profile-buffer-overflow/summary.json);
that failed capture is retained and rejected for kernel attribution. Reject
captures containing dropped-marker warnings even if forward validation passes.
The [successful 6,181-token capture](results/autoresearch/022-shape-profile-6181/summary.json)
contains 4,973 complete device operations in the marked window: 23.58 s summed
kernel time, with 9.68% matrix products and 2.38% standalone softmax. Large
binary/untilize/slice groups still carry the padded complex-RoPE pair shapes.
These kernel shares are not end-to-end attention/MLP or PJRT wall-time shares.
The [full occupancy candidate profile](results/autoresearch/025-occupancy-real-rope-profile/summary.json)
covers all 4,096 latent tokens and 30 blocks: 4,406 device operations and 3.97 s
summed kernel time. Reshape now accounts for 29.8%, matrix products 25.1%, and
binary operations 16.7%. Remaining layout changes need separate mathematical
and trajectory checks; this profile establishes no reconstruction-quality result.

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
The [6,181-token native sparse experiment](results/autoresearch/017-native-sparse-real-rope/summary.json)
passes the forward smoke at 7.86 s with zero FP32 preparation drift and no
reported fallback. It is slower than the 6.62 s sequence candidate. The
[matched stock-native control](results/autoresearch/021-native-sparse-control/summary.json)
fails on the exact historical 607,617,024-byte allocation. Real-pair RoPE alone
avoids this failure. The allocator had sufficient aggregate free space but
insufficient contiguous space per DRAM bank. Shape arithmetic identifies the
padded complex-pair tensor as the likely source; no allocator stack trace names it.

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
records agreement on all eight views. The
[final restoration](results/autoresearch/020-dino-exact-restoration/summary.json)
also reproduces the original DINO weights-file SHA256 exactly, including the
restored zero biases and mask token. The initial summary omitted this comparison
and has been corrected. The default local DINO path is restored. Config follows
official conversion; its historical file hash is unavailable, and current
software versions still require fresh paired scene references.

After authorized HF login, the
[canonical Facebook source check](results/autoresearch/023-authorized-original-dino/summary.json)
downloaded the pinned original weights, config and preprocessing config. Every
effective model-config field matches the conversion, and all eight cropped
poster features plus camera tensors are bitwise identical to the prepared
sampler fixture. The ongoing paired sampler runs can therefore keep those inputs.

`generate_decoder_fixture.py --conditioning-only` restores posed-image features
and cameras without running the flow samplers. It records the DINO config,
weights and conversion-report hashes and finishes as `conditioning_ready`,
distinct from a sampled decoder fixture. The
[fresh poster conditioning](results/autoresearch/018-poster-conditioning/summary.json)
has shape `[1,8,1029,1024]` with finite values and the original crop.

`prepare_ss_inputs.py --conditioning-dir /path/to/conditioning --output-dir
/path/to/new-inputs` applies the original FP32 CPU projection and learned
aggregation, then saves independent seed-123 occupancy noise. Run the matched
inputs with `run.py --cases ss_flow --flow-inputs /path/to/new-inputs/inputs.pt
--sampler-trajectory --reference-precision float32`. Repeat with
`--rope-mode real_pairs` in a fresh directory. The loader enforces checkpoint
identity, supported tensor shapes/dtypes and finite values; trajectory artifacts
check both guidance conditions, noise and original sampler configuration.
[Prepared poster inputs](results/autoresearch/019-poster-ss-inputs/summary.json)
isolate the sampler from upstream TT feature/aggregation errors. Real-image
inputs currently support `ss_flow`; sparse shape/texture fixtures remain future work.

Avoid repeating long stock CPU trajectories with a sealed reference cache:

```bash
python -m experiments.tenstorrent.reference_cache \
  --source-case /path/to/completed-cpu-run/ss_flow --output-dir /path/to/new-cache
```

Pass `--cpu-trajectory-cache /path/to/new-cache` to a new sampler run. The loader
checks the artifact hash, original model/reference sources, strict checkpoint,
versions and every actual fixture tensor/config. Experimental preparation can
change; stock reference math cannot. Stored CPU durations describe the original
computation, and cache loading is recorded separately. The
[retained poster CPU reference](results/autoresearch/027-cpu-reference-cache/summary.json)
matches both independently computed references bitwise. It can be restored from
the committed manifest and compressed tensor file on a stateless machine. Use
a generous `--timeout`, since the harness includes CPU setup and device wait.
Cache reuse also hashes the actual checkpoint file, so altered weights cannot
pass with unchanged provenance metadata. Full validation took 2.52 s for the
poster reference; the initial 70 ms measurement covered tensors/manifest only.
The [stock real-poster occupancy trajectory](results/autoresearch/028-stock-poster-occupancy/summary.json)
fails its full FP32 check: 364.94 s for 22 forwards, final relative L2 0.748
and PCC 0.678, despite a passing first forward. There is no reported fallback
or trajectory recompilation. This comparison includes BF16/FP32 differences;
decoded occupancy and CUDA-pipeline equivalence require separate assessment.
The [paired real-RoPE run](results/autoresearch/029-paired-poster-occupancy/summary.json)
finishes the same sampler in 88.12 s, a 4.14x speedup. It also fails FP32 gates,
and its final latent differs 29.4% from stock TT. Similar error magnitudes
against FP32 do not mean the TT variants agree. Keep the candidate experimental.
`compare_occupancy.py` decodes matched trajectories with the same official FP32
CPU decoder to isolate sampler effects. The
[poster occupancy check](results/autoresearch/030-poster-occupancy-quality/summary.json)
finds IoU 0.913 between TT variants, and 0.509/0.514 for stock/candidate against
the FP32 sampler. Those occupied-set changes prevent a preserved-quality claim.
This is one crop's occupancy, not a reconstructed mesh or a TT decoder test.

The [real-poster precision calibration](results/autoresearch/031-real-poster-precision/summary.json)
finds 41.5% CPU pipeline-diagnostic latent drift against FP32, but occupancy IoU
0.868 under the common decoder. TT still differs about 71% from that diagnostic
and has occupancy IoU about 0.54 against it. Intended rounding alone does not
explain the remaining gap. These errors are not additive, and the CPU diagnostic
does not establish CUDA equivalence or relax any gate.

`attention_diagnostic.py` extracts pretrained first-block Q/K/V and compares
stock attention with FP32 SDPA on identical tensors. It checks original CPU
attention reconstruction, retains native BF16 CPU references, measures repeated
PJRT outputs, and can export IR **after all measurements**. The
[controlled experiment](results/autoresearch/033-isolated-attention-precision/summary.json)
reduces self-attention error from 10.2% to 3.28%; cross-attention remains about
1.8–1.9%. This is an isolated arithmetic result, not a full-model quality fix.

Do not interleave `codegen_py` with inference: it changes process-global compile
options and mutates the exported module. The
[initial interleaved run](results/autoresearch/032-rejected-export-interleaving/summary.json)
was rejected. Fresh export modules and a separate no-export process now verify
that native results match bitwise.
The [explicit FP32 post-matmul scaling trial](results/autoresearch/034-attention-postscale/summary.json)
reduces self-attention error to 2.72%, while cross-attention stays near 1.8%.
Its baseline outputs and independent no-export control match bitwise. This
remains an isolated diagnostic; no full-flow defaults or quality gates change.

`attention_stages.py --fixture-directory /path/to/attention-run --output-dir
/path/to/new` captures QK, scaled scores, probabilities and AV. CPU local
comparisons consume the actual TT intermediate inputs; attribution requires
bitwise equality to ordinary TT attention and exact repeatability. The
[stage diagnostic](results/autoresearch/035-attention-stage-errors/summary.json)
finds softmax error of 3.99%/2.36% on identical self/cross score tensors, while
QK error is below 0.06% and scaling is exact. HiFi3 does not fix it. Full state
hashes and representative rows are retained without treating local errors as
additive whole-model effects.

`softmax_diagnostic.py` compares native softmax, renormalization and stable
exponential formulations on identical observed score tensors. The
[unfused reciprocal formulation](results/autoresearch/036-softmax-formulations/summary.json)
reduces self/cross error from 3.99%/2.36% to 0.0631%/0.0255% without fallback.
Direct exp/div is fused back to native softmax; exp times reciprocal avoids the
compiler's DivOp fusion pattern. Native IR is dumped during ordinary execution,
with dry run explicitly disabled. An `exp2` attempt was rejected for CPU fallback.
These local improvements still require complete-flow and sampled-quality tests.

`run.py --attention-mode fp32_reciprocal --rope-mode real_pairs
--sampler-trajectory --native-ir` tests that formulation in complete dense or
fixed-sequence flows. Q/K/V and attention products use FP32; the original
projections, trained LoRA, normalization, RoPE, output dtype and sampler stay
intact. The stock CPU references are captured before preparation. A 30-block
cross-attention flow must report **60 prepared attention modules**. Preparation
rejects unknown wrappers instead of silently skipping them; all helper imports
use the same package path. This is an opt-in experiment, with defaults unchanged.
The [complete shape trial](results/autoresearch/038-full-reciprocal-shape/summary.json)
confirms zero native softmax and 60 FP32 exponential sequences, with no fallback
or trajectory recompilation. Single-forward relative error is 5.62%, but final
sampler error remains 39.2% vs FP32 and 40.0% vs the CPU precision diagnostic.
The isolated softmax improvement has not resolved full-trajectory drift.
The [matched shape control](results/autoresearch/039-matched-reciprocal-shape/summary.json)
confirms a 13–14% runtime increase and slightly worse final sampler error.
On the [real poster sampler](results/autoresearch/040-poster-reciprocal-sampler/summary.json),
runtime rises from 88.12s to 129.73s, while final latent error falls from 74.9%
to 60.7%. The [common-decoder occupancy comparison](results/autoresearch/041-poster-reciprocal-occupancy/summary.json)
improves IoU vs FP32 from 0.514 to 0.678 (CPU precision diagnostic: 0.868).
This is an opt-in accuracy/latency tradeoff on one crop. Numerical gates still
fail, and mesh/material quality has not been established.

`attention_precision.py --fixture-directory <sealed QKV run> --output-dir <new>`
tests narrower precision boundaries on identical learned tensors, with native
IR and an exact legacy-expression control. `fp32_scores_bf16_av` retains FP32
scores/softmax and rounds probabilities for a BF16 attention–value product.
`bf16_qk_fp32_softmax` also rounds the QK product to BF16 before FP32 scaling
and softmax. BF16 denotes the original torso dtype; FP32 CPU preparation checks
retain FP32. Both modes are opt-in through `run.py --attention-mode` and require
full sampler/quality checks. They do not change weights or sampler schedules.

`output_head_diagnostic.py` applies the original CPU head to accepted captured
hidden states, verifying checkpoint/artifact hashes and exact reconstruction
of both CPU references. The [teacher-forced result](results/autoresearch/043-output-head-attribution/summary.json)
finds 0.52% combined head arithmetic error on identical TT hidden values, while
the CPU head on those values already produces 6.77% prediction error vs stock
(observed TT: 6.69%). This shape forward points toward amplified upstream
differences. It does not separate native normalization from projection errors,
or establish the same attribution for other stages or scenes.

`run.py --rope-mode real_pairs_head_first` is an experimental dense/fixed-sequence
layout. It moves Q/K to `[batch,heads,tokens,dim]` for real-pair RoPE and retains
that layout through SDPA or the reciprocal attention kernel. Original learned
Q/K normalization precedes the move; weights, LoRA and outer dtypes are retained.
CPU checks cover bitwise RoPE equivalence with 12 heads, complete blocks and
both attention precision modes. Native sparse attention is explicitly rejected.
The [first source-view trial](results/autoresearch/047-rejected-head-first-view/summary.json)
was rejected: native IR retained token-first cosine arithmetic with extra casts,
forward time increased, and sampled states differed substantially from control.
Python axis order alone did not establish the intended physical layout.

The [complete BF16-AV poster trial](results/autoresearch/045-poster-bf16-av-sampler/summary.json)
measures 128.155s vs the refreshed FP32 control's 129.725s. Its
[common-decoder occupancy IoU](results/autoresearch/046-poster-bf16-av-occupancy/summary.json)
improves from 0.678 to 0.713, with all control grids unchanged bitwise. This
single-crop accuracy result and small measured runtime decrease require broader
validation; strict latent gates remain failed and defaults stay unchanged.

`rope_layout_probe.py` compares source views, explicit contiguous tensors and
flattened arithmetic before whole-model trials. The
[native kernel probe](results/autoresearch/048-rope-layout-probe/summary.json)
retains a head-first cosine multiply with flattened arithmetic: at 4096 tokens,
kernel plus output-transfer time falls 28.06ms to 12.60ms and matches CPU values
bitwise on the seeded fixture. At 512 tokens it is slightly slower. Complete-flow
results follow below; these kernel timings alone did not establish them.
The [integration control](results/autoresearch/049-integrated-flat-rope/summary.json)
confirms the helper matches its prototype bitwise on hardware. Use
`--rope-mode real_pairs_flat_head_first` for complete-flow trials. This retains
the same rotation and uses flattened head/token rows around its arithmetic.
Independent kernel outputs repeat bitwise; microtiming varies across processes,
so full sampler timing and common-decoder quality remain required.

`--device-precision float32 --reference-precision float32` establishes an FP32
device-tensor baseline after the original stock reference and cache checks.
It promotes the already quantized learned values, preserves LoRA parameter
identities and complex/integer buffers, and updates model dtype attributes.
Reports include parameter bytes/dtypes and the promoted CPU forward check.
The original load/reference metadata stays intact; native IR and measured
error establish device behavior. FP32 tensor storage does not guarantee IEEE
FP32 arithmetic inside every native kernel. This is an opt-in accuracy
diagnostic with higher resident weight memory, and defaults remain unchanged.

The [complete flattened-RoPE poster trial](results/autoresearch/050-poster-flat-rope-sampler/summary.json)
reduces the original 22-forward sampler from 128.155s to 104.503s (18.46%).
Native IR confirms all 60 cosine multiplies use the intended head-first layout;
nominal result/phase payloads fall 64/64 MiB to 24/2 MiB. There is no fallback
or trajectory recompilation. The [common-decoder comparison](results/autoresearch/051-poster-flat-rope-occupancy/summary.json)
raises occupancy IoU vs stock FP32 from 0.713 to 0.727, with all five historical
control grids unchanged bitwise. This is an opt-in improvement on one crop;
strict latent gates still fail and mesh/material quality remains unverified.

The [matched FP32-model control](results/autoresearch/052-matched-fp32-device/summary.json)
reduces shape512 final error from 39.2% to 29.4%, with about 26% longer sampler
time and 5.377 GB of parameter payload. Its linear/matmul operands are FP32,
while native lowering still uses BF16 tables in all 60 RoPE gathers. The
[complete prepared CPU control](results/autoresearch/053-prepared-cpu-fp32-control/summary.json)
has only 0.0106% final drift against stock and passes every state gate; the large
native error persists against that same prepared model. Model/helper source
hashes, checkpoint bytes and full fixture identity are verified.

`precision_reference.py --precision float32 --rope-mode real_pairs
--attention-mode fp32_reciprocal` reproduces that complete prepared CPU control.
Its default remains the original BF16-boundary diagnostic. Model preparation
and CPU reference controls retain all original sampler settings.

The [FP32 lookup counterfactual](results/autoresearch/054-fp32-rope-lookup-loss/summary.json)
shows that native flattened RoPE exactly matches CPU math with only its gathered
partner rounded to BF16. A [simple residual split](results/autoresearch/055-rejected-cast-roundtrip-split/summary.json)
fails because native cast folding eliminates the rounding residual. The
[compensated lookup](results/autoresearch/056-compensated-fp32-lookup/summary.json)
uses three BF16 payloads with a BF16 multiply between casts. At 512 and 4096
tokens, integrated native outputs match this compensated CPU formula and an
independent prototype process bitwise; error against the original rotation is
about 1e-8 relative L2. This adds kernel cost and has only been checked on normal
activation ranges. Use `--rope-mode real_pairs_fp32_split` for opt-in full-flow
validation; sampler quality and speed require separate measurements.

The [complete matched shape sampler](results/autoresearch/057-compensated-fp32-shape-sampler/summary.json)
limits that local gain: final error is 29.58% with compensation versus 29.99%
with the same flattened layout and ordinary gather, while runtime increases
16.866s to 18.810s. Both fail the original gates. All 21 model calls execute
without fallback or recompilation. The [complete compensated CPU control](results/autoresearch/058-compensated-prepared-cpu-control/summary.json)
passes every state gate with 0.00643% final drift against stock; native error
remains 29.58% against that identical prepared model. Source hashes and physical
checkpoint bytes match. Lookup compensation remains an opt-in diagnostic.

The runtime's Wormhole HiFi4/FP32-accumulation warning also motivated a
[matched FP32 HiFi3 trial](results/autoresearch/059-matched-fp32-hifi3/summary.json).
Actual runtime IR confirms all 457 linear/matmul operations use the requested
fidelity with FP32 operands and accumulation. HiFi3 gives 30.18% final error
versus HiFi4's 29.99%, with essentially unchanged sampler time (16.895s versus
16.866s). It does not fix accuracy on this fixture. This is distinct from the
earlier BF16-pipeline HiFi3 experiment; defaults and gates remain unchanged.

`fp32_flow_diagnostic.py --first-block --output-dir <new>` observes the same
FP32/HiFi4 shape recipe as the matched control. It retains all block outputs
and first-block normalization, attention, MLP and residual boundaries. Stock
and identically prepared CPU values are saved separately. The observer must
match the ordinary native final output bitwise, repeat every state exactly,
and report no fallback before those intermediates support attribution. MLP
preactivations are deliberately omitted so linear/GELU fusion can survive.
This numerical capture is separate from sampler timing and scene quality.
After an accepted detailed capture, `fp32_first_block_analysis.py
--capture-directory <capture> --output-dir <new>` reconstructs its prepared CPU
states bitwise, then compares each native boundary with CPU math on the same
observed inputs. It verifies checkpoint bytes, model/helper sources, input
digests and tensor artifacts. Combined attention, MLP and output-head errors
are reported separately from propagated differences; they are not additive
contributions to the final prediction error.

The [FP32 detailed capture](results/autoresearch/060-fp32-first-block-observation/summary.json)
passes that hardware guard: its final output matches the ordinary forward
bitwise, all 45 states repeat exactly, and no fallback occurs. Native IR retains
FP32 linear/matmul operands with HiFi4 accumulation. First-block/native depth
states and both runtime IRs are retained for controlled local analysis.

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
