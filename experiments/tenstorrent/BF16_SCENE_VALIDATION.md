# BF16 decoder speed and accuracy on the poster fixture

Both complete pretrained decoders executed the full real-image fixture using
the ordinary BF16 path on one Wormhole B0. Warm medians were **15.09 seconds for
shape** and **14.63 seconds for texture**. This path is much faster than the
previous accurate-mode calls, but fails the declared numerical and exact-topology
gates. The model backend and its precision algorithms were unchanged for this
measurement; the scene harness gained BF16 selection and warm repeats.

## Method

- Same 4,490-voxel image-conditioned input and saved FP32 CPU references used
  by the accurate-mode experiment; no voxel truncation or regenerated sampling.
- Input SHA-256:
  `ec164c1e73ee3f9117068a45e2e75100f26345b25aba5f954ca6db91d3d69d41`.
- `precision=bf16`, `algorithm=explicit_gemm`, `row_chunk=1024`, 16 CPU threads,
  TT-Metal `9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9`, FlexGEMM `6492d51`.
- One BF16 matrix-product pass per convolution/linear operation, ordinary TTNN
  normalization and BF16 feature rearrangement. This is BF16, not IEEE FP16.
- Sequential shape/texture weight preparation, bounded topology cache and
  Watcher interval 30. Device access was serialized with the concurrent DINOv3
  validation using `/tmp/genrecon-tt-pjrt.lock`; no simultaneous Tensix use.
- Fresh fixed-guide and independently predicted-guide checks in BF16. The
  report retains prior accurate-mode results as context, but they do not stand
  in for the BF16 fixed-guide comparison or its acceptance checks.
- Three warm predicted-guide calls per decoder after initial execution. Timers
  include input/output transfers, CPU topology work and synchronization, and
  exclude weight preparation, comparisons and file writes. All repeated output
  coordinates and features matched the initial BF16 output bitwise.

## Timing

| Measurement | Shape | Texture |
| --- | ---: | ---: |
| First fixed-guide call | 124.77 s | 14.59 s |
| First predicted-guide call | 74.76 s | 14.68 s |
| Warm predicted-guide median, 3 calls | **15.09 s** | **14.63 s** |
| Warm range | 15.06–15.23 s | 14.60–14.65 s |
| Weight preparation | 1.80 s | 1.30 s |

The first shape calls compiled many kernels for these large BF16 tensor shapes;
texture reused much of that compiled work. These are initial-call timings, not
measurements from an entirely cleared disk cache. The run's JIT summary reported
508/1,316 cache hits. The earlier accurate-mode predicted calls were 197.73 and
189.76 seconds, but were not collected as a matching warm benchmark: do not
interpret the ratio as a controlled precision-only speedup. The output voxel
counts also differ slightly between modes.

Live DRAM after shape preparation was 960.36 MB and after texture preparation
980.88 MB, including retained runtime allocations. These are allocator snapshots,
not peak memory. The device closed normally after all calls.

## Accuracy against the FP32 CPU decoder

| Comparison | Shape relative L2 | Texture relative L2 |
| --- | ---: | ---: |
| Fixed CPU subdivision guides, all 1,711,800 voxels | **1.49%** | **0.733%** |
| BF16-predicted guides, 1,704,477 common voxels | **18.48%** | **0.859%** |

These errors apply to seven raw shape channels and six raw material channels.
Intersection errors exclude missing/extra coordinates and must not be treated
as a complete output-set match. With fixed guides, 19,014 of 4,015,392 predicted
subdivision signs and 11,795 of 5,135,400 final edge signs differ from CPU.

Allowing BF16 shape predictions to determine the topology produces 1,724,799
output voxels versus CPU's 1,711,800:

- 7,323 reference voxels are missing; 20,322 voxels are extra.
- Coordinate-set intersection-over-union is 98.40%.
- 75,636 of 5,113,431 edge signs differ on common voxels (1.48%).
- Counting positive edge identifiers over the union of coordinates gives
  96.67% precision, 98.15% recall and 94.94% intersection-over-union.
- The transformed vertex-offset relative L2 error on common voxels is 5.56%.
  Absolute per-component offset error averages 0.00956 voxel units; its 95th
  percentile is 0.0337, 99th percentile 0.1289, and maximum 1.159.
- After the original material transform `raw * 0.5 + 0.5`, without clamping,
  mean per-channel absolute errors range from 0.00148 to 0.00282. Local color
  errors reach 0.273. These metrics do not establish rendered material quality.

The isolated captured 1024→4096 convolution executed with relative L2 0.267%,
but failed its existing per-element tolerance (`atol=rtol=0.02`). The complete
report remains **`numerical_failure`**; no threshold or gate was relaxed. BF16 is
a substantially cheaper execution option whose reconstruction-quality tradeoff
still needs visual and multi-scene evaluation. The result does not establish
FP16 accuracy or performance.

## Reproduce and inspect

```bash
TT_METAL_WATCHER=30 python experiments/tenstorrent/sparse_decoder_scene_probe.py \
    --reference-dir /home/user/tt-scratchpad/genrecon/sparse-decoder-poster-bounded \
    --output-dir /path/to/new-bf16-report \
    --precision bf16 --row-chunk 1024 --warm-repeats 3
```

The harness checks input/checkpoint/reference-source provenance before device
execution and holds the shared lock through teardown. A nonzero exit is expected
for the recorded parity failure. Raw tensors and logs are retained locally under
`/home/user/tt-scratchpad/genrecon/sparse-decoder-poster-bf16` and
`/tmp/genrecon-poster-bf16.log`.

- [Complete benchmark report](results/sparse-decoder-integration/poster-bf16.json)
- [Geometry/material field diagnostics](results/sparse-decoder-integration/poster-bf16-fields.json)
- [Original accurate-mode integration report](SPARSE_DECODER_INTEGRATION.md)

The CPU comparison/gate regression suite passes **14 tests**, including checks
that fresh fixed-guide failures override older results and that nondeterministic
warm outputs cannot pass. No neural backend or kernel implementation was changed.
