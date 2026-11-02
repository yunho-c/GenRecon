# Single-cube scene pipeline measurements

The staged chain completed successfully from eight real images through matched
raw geometry/material outputs on **2,402,851 voxels**. Measured warm components
sum to **1,310.99 seconds (21.85 minutes)**. All **61 flow evaluations** completed
with finite outputs, no reported XLA fallback, and no additional compiled graphs
during sampling. This is a single-chip baseline, not an n300 timing.

This experiment connects the pretrained image encoder, three original flow
samplers, occupancy decoder, and both sparse VAE decoders through saved stage
outputs. It measures one bounded poster-scene cube at **512 resolution** on
**one n150 Wormhole chip**. It does not measure an n300 or the default
`1024_cascade` pipeline. CUDA-dependent mesh extraction/export is excluded.

The runner is [scene_pipeline.py](scene_pipeline.py). Large tensors and compiler
logs live under `/home/user/tt-scratchpad/genrecon/scene-pipeline-poster/`.

The complete shape sequence adapter matched the original sparse model **exactly**
in FP32 CPU on all 6,181 input tokens. Its first TT execution completed with
3.67% relative L2 error against that reference, avoiding the observed native
layout allocation failure. This does not establish the precise compiler-level
cause of that failure.

## Measured timing

| Stage | Warm measured time | Scope |
| --- | ---: | --- |
| Image preprocessing and DINOv3-L | 1.90 s | Eight views; DINO execution alone 1.55 s |
| Occupancy sampler | 364.97 s | 12 steps, 22 network evaluations |
| Occupancy decoder and coordinate extraction | 2.96 s | One invocation, three-call decoder median |
| Shape sampler | 493.62 s | 12 steps, 21 evaluations, 6,181 active voxels |
| Texture sampler | 423.11 s | 12 steps, 18 evaluations, same active voxels |
| Shape VAE decoder | 11.90 s | One invocation, median of three complete warm calls |
| Texture VAE decoder | 11.59 s | One invocation, median of three complete warm calls |
| Projection, aggregation and condition setup | 0.94 s | Sum of measured remaining conditioning components |
| **Warm component sum** | **1,310.99 s / 21.85 min** | Includes the full sampling schedules; no further ×12 multiplier |

The three samplers account for **97.8%** of that sum. Warm forward medians are
16.59 s for occupancy, 23.50 s for shape and 23.51 s for texture. Guided steps
take twice the forward time; measured host sampling/instrumentation overhead is
0.031/0.066/0.032 s over the entire respective trajectory.

Initial synchronized network calls were 32.53 s for DINO, 186.42 s for occupancy
flow, 9.62 s for occupancy decoding, 209.49 s for shape flow, 195.95 s for texture
flow, and 157.63/11.60 s for shape/texture VAE decoding. Replacing one warm call
per network with these initial calls, including aggregators, adds **746.39 s
(12.44 min)** to the component accounting. This uses existing caches and still
excludes checkpoint loading and process/device startup; it is not a measured
cold end-to-end run. Successful validation workers total 47.72 minutes because
they additionally perform CPU references, warmups, repeated measurements and
artifact I/O. Failed attempts are excluded from that worker sum.

Both VAE decoders repeated their coordinates and feature values bitwise across
three warm calls, and their final coordinate orders match each other. Watcher
was disabled. These are different generated latents/topologies from the prior
4,490-input-voxel [BF16 benchmark](BF16_SCENE_VALIDATION.md), which also enabled
Watcher at interval 30; the 11.90/11.59 versus 15.09/14.63 figures are not a
controlled speedup comparison.

Machine-readable evidence is in [the summary](results/scene-pipeline-poster/summary.json)
and its linked stage reports. Source/input/checkpoint hashes and all five XLA
metric reports are retained. Compile sample counts are 1/2/1/2/2 for
DINO/occupancy flow/occupancy decoder/shape/texture respectively; each flow's
two compilations cover its aggregator and complete flow. Dynamo graph counts
remain unchanged across the sampler loops.

## Method and timing boundaries

- Eight posed, undistorted, principal-point-cropped poster images, with the
  exact camera normalization and image hashes from the existing CPU fixture.
- DINOv3-L runs in BF16 with its original FP32 RoPE frequencies and GenRecon's
  non-affine output normalization. Flow networks use the pipeline's mixed
  precision: BF16 torso with FP32 embedding/output/normalization parameters.
  The dense occupancy decoder uses FP32. Sparse VAE decoders use the ordinary
  BF16 FlexGEMM path with `row_chunk=1024`.
- Original 12-step Euler samplers, time warping, guidance intervals, strengths,
  and guidance rescaling are retained. Occupancy makes **22** network calls,
  shape **21**, and texture **18**: **61** calls altogether. VAE decoding happens
  once per generated chunk, after sampling.
- Camera projection/indexing, Euler/CFG arithmetic, and sparse topology stay on
  CPU. Learned image, aggregation, flow, and decoder operations execute on TT.
  Each timed flow call includes its input transfer and synchronized CPU output;
  positive and negative condition tensors are prepared for the device once per
  stage and reused. XLA may defer physical transfers until execution, so `.to`
  dispatch timings must not be read as complete transfer-cost measurements.
- Per-stage noise seeds are explicitly 123/124/125, independent of checkpoint
  construction. This is a new sampled trajectory, not a replay of the old CPU
  fixture's random stream. A difference from that fixture's 4,490-voxel count
  cannot be attributed solely to precision.
- TT-generated occupancy determines all downstream coordinates. Shape's own
  predicted subdivision guides drive texture decoding. No rows are truncated,
  and no CPU fixture latents are substituted into the chain.
- TT-XLA and TTNN run in separate processes because they use different Metal
  runtimes. Device access is serialized through the process-lifetime lock
  `/tmp/genrecon-tt-pjrt.lock`. Six stages run sequentially.
- Three identical-input warm forward measurements accompany each flow, then
  one complete sampler trajectory is timed. Decoder medians use three complete
  warm calls. A 12-step sampler timing is one loop's wall time, not a median of
  three trajectories.
- Warm component sums exclude checkpoint loading, initial compilation,
  device/process startup/teardown, reference comparisons, and stage-file I/O.
  They are **not a timed fully resident end-to-end invocation**. Initial-call
  measurements use the existing compiler/kernel caches, not cleared caches.

## Reproduction

```bash
python experiments/tenstorrent/scene_pipeline.py \
  --fixture /home/user/tt-scratchpad/genrecon/poster-fixture/cpu-available \
  --sparse-flow-layout sequence \
  --output-dir /home/user/tt-scratchpad/genrecon/scene-pipeline-poster
```

The driver defaults to `/home/user/tt-forge-venv/bin/python` for TT-XLA and
`/home/user/tt-metal/python_env/bin/python` for TTNN. Use a fresh output directory
for a new run. `--resume` reuses completed stages with the same fixture and seed;
failed stage directories must first be preserved under a different name.
The CLI defaults to the successful `sequence` sparse layout. Use explicit
`--sparse-flow-layout native` to reproduce the original flow boundary, with
`--flow-group-size 5` to reproduce the grouped experiment.

CPU tests in `test_scene_sampling.py` verify exact agreement with the original
dense/sparse/texture Euler samplers, check grouped native blocks with nonzero
LoRA, and reject mismatched texture conditioning coordinates. Tests in
`test_flow_adapter.py` compare both single-chunk and ragged modulation with
native dense blocks. Hardware evidence records synchronized timings,
checkpoint/source/input hashes, fallback counters, and compiler graph counts.

## Real-size memory failures

The TT-generated occupancy contained **6,181** active voxels. The first full
shape-flow execution failed to allocate a 607,617,024-byte DRAM intermediate.
Although about 1.33 GB total free space remained, its largest free allocation
per bank was below the requested size. A second attempt with synchronized
five-block groups also failed on a buffer of the same size. Both workers closed
normally; no hardware reset was used.

That allocation is exactly `6181 * 32 * 1536 * 2` bytes: 32 times one BF16 feature
matrix, consistent with tile-padding expansion. Replacing the single-chunk
modulation's indexed batch broadcast with a direct broadcast passed CPU tests
but failed on the same allocation in a third hardware attempt. That production
code change was reverted; the specific offending lowering was not established.

The subsequent experiment uses the existing `SingleChunkSLatProbe`: all active
voxels form one `[1,N,C]` token sequence, with the same native dense transformer
blocks, strictly loaded original weights/LoRA, and CPU-prepared coordinate RoPE.
It retains every active voxel and full attention, without creating a dense 3D
grid. Before TT compilation, the complete adapter is checked against the
original sparse model on the actual 6,181-token input in FP32 CPU.

The failed attempts are preserved in `shape-attempt1-full-oom/` and
`shape-attempt2-grouped-oom/` and `shape-attempt3-direct-broadcast-oom/` under the
raw artifact directory. The optional
`--flow-group-size` harness setting reproduces the grouped experiment; grouping
alone is not a demonstrated solution to this allocation failure.

## Numerical scope

The first conditional forward of each flow is compared with FP32 CPU using
the same noise and TT-generated conditioning. The occupancy decoder is compared
on the actual sampled latent. These are local numerical checks, not a matched
CPU-versus-TT comparison of the complete sampled trajectory or rendered scene.

The earlier [BF16 decoder validation](BF16_SCENE_VALIDATION.md) established that
successful BF16 execution does not imply exact topology or geometry parity.
More scenes, a matched full trajectory reference, and visual mesh/material
validation remain necessary for reconstruction-quality conclusions.

| Local comparison against FP32 CPU | Relative L2 |
| --- | ---: |
| DINO features, all eight views | 1.35% |
| First occupancy flow prediction, same input/conditioning | 3.30% |
| First shape flow prediction versus original sparse model | 3.67% |
| First texture flow prediction versus original sparse model | 3.65% |
| Occupancy decoder logits, same TT-sampled latent | 0.295% |

Both complete sparse-flow sequence adapters are bitwise equal to their native
sparse counterparts in FP32 CPU on the actual 6,181-token inputs. Occupancy
decoding disagrees on 22 raw logit signs and changes one coarse coordinate:
6,180 common, one missing and one extra voxel, with equal 6,181-voxel counts
(99.9676% coordinate IoU). This is not exact threshold/topology parity.

Remaining deployment uncertainties are n300 device placement/parallel execution,
the default 1024 cascade's additional shape pass and larger token count, scene
variation, reconstruction quality and the original public pipeline's unified
runtime/mesh integration. The successful harness intentionally uses isolated
TT-XLA/TTNN processes and raw output files.
