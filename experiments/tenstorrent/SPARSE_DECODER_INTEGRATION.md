# FlexGEMM sparse decoder integration

Both complete pretrained shape and texture decoders now have a TTNN/FlexGEMM
inference adapter. It preserves the learned ConvNeXt/C2S architecture and keeps
feature arithmetic on the device. CPU code manages coordinates and subdivision
selection. This covers neural decoder inference; image sampling and mesh export
are separate pipeline stages.

Integration and end-to-end decoder execution are implemented. Synthetic parity
tests pass. A real-image fixture also executes both complete decoders, but fails
the exact topology gate after a near-zero subdivision decision changes sign.
Further scene-quality and performance validation is needed before production use.

## Use

Use the TT-Metal Python environment and FlexGEMM's `hardware/tenstorrent` branch
(commit `6492d51` or later). The tested SDK is TT-Metal
`9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9`, on one Wormhole B0 chip with an 8×8
compute grid. TT-XLA/PJRT and TTNN should run in separate processes.

Before constructing a model without CUDA extensions, select the portable
checkpoint-compatible convolution backend:

```python
from genrecon.modules.sparse import config
config.set_conv_backend("reference")
```

Strictly load the original checkpoints into CPU models, select evaluation mode,
and prepare resident weights once. The validation runner contains a complete
local checkpoint loader.

```python
import ttnn
from genrecon.backends.tenstorrent.sparse_decoder import TopologyCache

# Caller supplies CPU SparseTensors with matching coordinates and denormalized
# decoder inputs. The caller also owns the TT device's lifetime.
device = ttnn.open_device(device_id=0)
cache = TopologyCache()
try:
    shape_model.eval().enable_tenstorrent(device, topology_cache=cache)
    geometry, subdivisions = shape_model.forward_features(shape_slat, return_subs=True)
    # Release the shape weights before preparing texture for large scenes.
    shape_model.disable_tenstorrent()
    cache.clear()
    texture_model.eval().enable_tenstorrent(device, topology_cache=cache)
    materials = texture_model(texture_slat, guide_subs=subdivisions)
finally:
    shape_model.disable_tenstorrent()
    texture_model.disable_tenstorrent()
    cache.clear()
    ttnn.close_device(device)
```

`forward_features` exposes the seven raw geometry channels before mesh
extraction. The original base decoder `forward` and shape `upsample` methods
also dispatch to the prepared adapter. Texture output contains six raw material
channels. Original mesh-producing shape `forward` still requires its mesh
extensions. Training is unsupported. Parameter mutation after preparation raises
an error; disable and re-enable to upload new weights.

## Precision and performance choices

The default `precision="accurate"` uses FP32 features, explicit FP32 LayerNorm,
and thirteen small-integer BF16 matrix products per convolution/linear layer.
Four signed base-64 operand parts avoid precision loss observed in Wormhole
floating-point dot products. C2S rearrangement uses three BF16 residual parts
because the pinned SDK's FP32 gather loses low mantissa bits. All these operations
run on the device; this mode makes no CPU feature-computation fallback.
Skip-channel repetition also runs in bounded 16,384-row chunks. The SDK's
transpose/concat implementation otherwise exceeds L1 capacity at 388,659 rows;
the tiled implementation preserves FP32 values exactly through 1,711,800 rows.

`precision="bf16"` selects ordinary BF16 features and one matrix product. It is
faster and uses less weight memory, but changed subdivision and edge decisions
in pretrained tests. It is an explicit performance/quality tradeoff and is not
the validated default for topology parity.

The default `algorithm="explicit_gemm"` uses bounded device im2col chunks and
TTNN's optimized matrix multiply. `row_chunk=256` controls expanded temporary
storage. Every chunk can access every input neighbor. A bounded concat tree
combines at most four outputs per call, avoiding a reproduced SDK stall with
17 chunks. The four custom
implicit/masked/split-K algorithms remain selectable; current measurements favor
the explicit path. A faster sparse implementation is not assumed merely because
it skips absent neighbors.

Maps are cached by exact coordinate equality, batch/spatial shape, kernel,
dilation, stage, mask mode, and device identity. Shape and texture may share the
cache. Only the last layout per stage/kernel is retained, and layouts above
262,144 rows are used for the current convolution without being retained.
`TopologyCache(max_cached_rows=...)` controls this limit; `clear()` releases all
device metadata. Retaining million-row maps across passes caused DRAM
fragmentation in the real-scene test. Large scene validation prepares one
decoder's weights at a time and releases them before preparing the next.
The accurate mode
keeps four BF16 copies of each matrix, so both pretrained decoders occupy about
7.68 GB of live device DRAM after preparation in the eight-input test. This is a
live allocator snapshot, not a peak-memory or full-scene capacity measurement.

## Validation

The full probe executes both 474M-parameter pretrained decoders with strict
checkpoint loading. It compares fixed reference guides and independently
predicted guides, verifies exact coordinate ordering and subdivision/edge
signs, measures continuous geometry/material outputs, and replays the actual
internal 1024→4096 convolution input captured after four pretrained ConvNeXt
blocks. It records source/checkpoint hashes, device metadata, warm timings, and
raw tensors outside git.

The current accurate-mode numerical gate is relative L2 ≤ 1e-4 and PCC ≥ 0.999999,
plus exact topology and edge decisions. These are decoder parity checks, not a
substitute for real-scene reconstruction quality assessment.

Curated machine-readable evidence is in
[results/sparse-decoder-integration](results/sparse-decoder-integration/).

Complete pretrained synthetic cases, all with exact coordinates, subdivision
signs and geometry-edge signs:

| Inputs / seed | Algorithm / row chunk | Output voxels | Shape / texture relative L2 | Shape / texture warm median |
| --- | --- | ---: | --- | --- |
| 8 / 123 | Explicit / 256 | 732 | 7.88e-7 / 4.25e-7 | 1.988 / 1.954 s |
| 32 / 7 | Explicit / 64 | 3,100 | 9.54e-7 / 5.83e-7 | 5.698 / 5.664 s |
| 129 / 321 | Explicit / 256 | 15,438 | 8.47e-7 / 6.43e-7 | 7.772 / 7.724 s |
| 8 / 123 | Masked implicit split-K / 256 | 732 | 7.88e-7 / 4.25e-7 | 4.365 / 4.332 s |

The 129-input case matched all 46,314 edge decisions. Its captured 1024→4096
convolution had relative L2 3.22e-7 and maximum absolute error 6.10e-5. All captured
wide convolutions passed. The first eight-input report recorded the earlier
operator smoke threshold; its measured errors also satisfy the current gate.

The 64-row case deliberately stresses chunk boundaries, so these timings are
not a controlled voxel-scaling comparison. The complete masked split-K run
supports the default explicit algorithm choice for these tested inputs.
Timings include CPU boundary transfers, coordinate decisions and map lookups,
exclude weight preparation and initial compilation, and use three warm calls
with Watcher enabled. This small-input accuracy mode is not a speedup claim.

Targeted decoder regression tests: **19 passed** with Watcher. Coverage includes
complete shape/texture paths for explicit and masked split-K algorithms, ragged
chunks, batches, changed-coordinate cache invalidation, empty children,
parameter invalidation, FP32 normalization, cancellation, lossless C2S, and
million-row FP32/BF16 channel repetition, and bounded cache retention.
CPU validation tests: **12 passed** for coordinate intersection and enforcement
of all numerical/discrete gates; **2 passed** for fixture camera preprocessing.
The main probe's `--sequential-weights --warm-repeats 0` mode also passed a
fresh full pretrained eight-input run with exact topology and edge signs;
see [the sequential smoke report](results/sparse-decoder-integration/synthetic-sequential-8.json).
FlexGEMM's hardware suite: **75 passed, 4 expected failures**; the expected
failures retain the documented BF16 wide-layer strict-gate limitation.

```bash
flock /tmp/genrecon-tt-pjrt.lock env TT_METAL_WATCHER=120 \
    python -m pytest experiments/tenstorrent/test_sparse_decoder.py -q

TT_METAL_WATCHER=120 python experiments/tenstorrent/sparse_decoder_probe.py \
    --points 8 --precision accurate --output-dir /path/to/new-report-directory
```

The probe acquires the shared device lock itself. With `--latent`, supply an NPZ
containing `coords [N,4]`, `shape_feats [N,32]`, and `texture_feats [N,32]` in
matching row order and already denormalized decoder units. Without it, reports
explicitly identify synthetic Gaussian inputs. Raw artifacts are not committed.

## Real-scene fixture generation

The requested input format and local CPU generation command are documented in
[required-validation-data.md](../../docs/required-validation-data.md). The CPU
path uses the actual DINOv3-L and original pretrained sampling modules. Complete
CPU generation succeeded on the public Nerfstudio poster capture: eight views,
all three 12-step samplers, 4,490 matching shape/texture input voxels, and 33.9
minutes on 16 CPU threads. Generation provenance is in
[poster-cpu-fixture.json](results/sparse-decoder-integration/poster-cpu-fixture.json).
Its CPU shape reference expands to 1,711,800 voxels. Batched host topology lookup
reduced that full CPU forward from 470.39 to 73.07 seconds with bitwise-identical
coordinates and features; see
[host preparation comparison](results/sparse-decoder-integration/poster-host-preparation.json).
The initial large TT attempt exposed the skip-repetition L1 limit described
above, preserved in [the failure report](results/sparse-decoder-integration/poster-large-repeat-failure.json).
After that fix, the complete shape pass under fixed CPU guides had relative L2
1.15e-6 and zero mismatches among 5,135,400 edge decisions. **Strict subdivision
parity failed** at one of 4,015,392 decisions: CPU +5.01e-6 versus TT -2.86e-5 at
stage 2. The zero threshold is unchanged. See the
[margin diagnostic](results/sparse-decoder-integration/poster-subdivision-margin.json)
and [fixed-shape/memory report](results/sparse-decoder-integration/poster-fixed-shape-and-memory-failure.json).
This is a precision-sensitive topology difference, not a passed exact-parity
test. The final [sequential-weight run](results/sparse-decoder-integration/poster-sequential.json)
completed independent shape and both texture forwards without runtime errors,
using `row_chunk=1024`, the bounded topology cache, and one decoder's weights
resident at a time. No rows were dropped to fit memory.

| Real-image case | Output voxels | Relative L2 vs CPU | Outcome |
| --- | ---: | ---: | --- |
| Shape, fixed CPU guides | 1,711,800 | 1.15e-6 | Continuous outputs and all edge signs pass; one predicted subdivision sign differs |
| Texture, fixed CPU guides | 1,711,800 | 4.40e-7 | Exact coordinates and continuous output gate pass |
| Shape, own predicted guides | 1,711,801 | 2.02e-3 on common voxels | Exact-coordinate and continuous gates fail; 11 common edge signs differ |
| Texture, TT shape guides | 1,711,801 | 3.78e-6 on common voxels | Exact-coordinate gate fails; shape/texture coordinates match each other |

The predicted chain shares 1,711,798 coordinates with CPU, with two missing and
three extra voxels. The original one-sign difference propagates to four later
subdivision signs. Among common voxels, 765 shape rows and 11 texture rows have
maximum raw-channel error above 1e-3; this is a diagnostic count, not an adjusted
acceptance gate. Maximum raw shape error is 26.26. Applying the original geometry
transforms gives a maximum vertex-offset difference of 0.391 voxel units and
vertex relative L2 4.94e-4 on the intersection; see
[geometry diagnostics](results/sparse-decoder-integration/poster-predicted-geometry.json).
No visual mesh-quality claim follows from these metrics. The report remains
**`numerical_failure`**, and the zero subdivision threshold is unchanged.

Complete TT calls took 197.73 seconds for predicted shape and 189.68/189.76
seconds for fixed/predicted texture. CPU reference calls took about 73–80 seconds
for shape and 70–74 seconds for texture. These individual calls include transfers,
host topology work and initial shape-specific compilation, and are not a
controlled warm benchmark. They show no current speedup over CPU. Resident
weights were released between shape and texture; the post-release live DRAM
snapshot was 18.43 MB, not a peak-memory measurement.

The real fixture's captured 1024→4096 convolution passed with relative L2
1.39e-7 and maximum absolute error 2.44e-4. The wide convolution is operational;
the outstanding issue is whole-decoder sensitivity to near-zero decisions.
DL3DV generation still requires access to the official gated sample files.
Broader scene quality, memory limits, and useful production latency remain to
be established.
