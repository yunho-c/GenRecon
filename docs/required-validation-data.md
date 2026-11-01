# Data needed for real-scene Tenstorrent decoder validation

The FlexGEMM integration has complete pretrained shape/texture decoder tests.
A matching real-image latent pair has now been generated locally from the
public Nerfstudio poster capture, as documented below. Additional DL3DV or
deployment-scene pairs remain useful for broader quality, memory, and performance
validation. No images, camera calibration, or upstream sampling rerun are
required when these latents are available.

Both complete TT decoders executed this fixture through about 1.71 million
output voxels. Fixed-guide continuous errors are small, but the independently
predicted chain fails exact topology parity after a near-zero subdivision
decision changes sign. See the [results and limitations](../experiments/tenstorrent/SPARSE_DECODER_INTEGRATION.md).
More scenes and a native mesh/material reference would help assess the practical
effect; they are no longer needed just to establish real-input decoder execution.

## Minimum input: one matching shape/texture latent pair

Please provide an uncompressed or compressed NumPy `.npz` containing:

| Key | Shape | Type | Meaning |
| --- | --- | --- | --- |
| `coords` | `[N, 4]` | int32 or int64 | `[batch, x, y, z]` active voxel coordinates in decoder input lattice units |
| `shape_feats` | `[N, 32]` | float32 | Shape SLAt features immediately before the shape decoder |
| `texture_feats` | `[N, 32]` | float32 | Texture SLAt features immediately before the texture decoder |

Both feature arrays must use exactly the same coordinate rows in the same
order. Coordinates must be unique and nonnegative. A single chunk or merged
scene with batch index zero is sufficient to start. Values must be finite.
Save features **after the pipeline's latent mean/std denormalization**, in the
units actually passed to each decoder. Do not save normalized sampler outputs
or substitute checkpoint weights for scene latents.

For two existing CPU or CUDA `SparseTensor` inputs:

```python
import numpy as np
import torch

assert torch.equal(shape_slat.coords.cpu(), texture_slat.coords.cpu())
np.savez_compressed(
    "decoder_input.npz",
    coords=shape_slat.coords.detach().cpu().numpy().astype(np.int32),
    shape_feats=shape_slat.feats.detach().float().cpu().numpy(),
    texture_feats=texture_slat.feats.detach().float().cpu().numpy(),
)
```

For joint decoding, capture the merged, deduplicated input lattice after chunk
translation and overlap averaging. A matching single-chunk pair is also useful;
please identify which kind was saved. Unrelated shape and texture samples are
not a valid pair.

## Context to include

A short JSON or text sidecar should identify the scene, input resolution, chunk
or merged-scene scope, coordinate origin/scale, checkpoint names/revisions,
GenRecon commit, and how the latents were captured. Confirm that denormalization
has already been applied. The currently tested checkpoints are
`shape_dec_next_dc_f16c32_fp16` and `tex_dec_next_dc_f16c32_fp16`; the runner records
their SHA-256 hashes.

One small representative sample is enough for the first validation. Later,
several varied scenes and a large sample near the intended deployment size will
help test sparse neighborhoods, subdivision growth, memory use, and latency.
Do not truncate arbitrary rows from a scene for a quality comparison: removing
neighbors changes the convolution and the resulting reconstruction.

## Helpful optional reference artifacts

- Native decoder outputs: final coordinates, seven raw shape channels, and six
  raw material channels, with their row order retained.
- Four shape subdivision tensors, each containing parent coordinates and eight
  raw logits before thresholding at zero.
- A reference mesh and material preview from the same inputs, with the export
  settings recorded. This is useful for a later visual quality check; mesh
  extraction/export is outside the current decoder integration scope.

The runner can compute an FP32 CPU reference if native outputs are unavailable.
Native CUDA/FP16 results would additionally characterize the precision difference
between the original deployment and that FP32 reference.

## Generating a fixture locally from DL3DV

The official [DL3DV sample](https://huggingface.co/datasets/DL3DV/DL3DV-10K-Sample)
provides images, Nerfstudio camera transforms, and COLMAP sparse points for 11
scenes. Its files require accepting the dataset's access conditions. The local
account's download request was denied on 2026-11-01; listing public metadata does
not establish file access. Once access is enabled, only selected frames and
camera/point metadata are needed, rather than the full dataset.

[generate_decoder_fixture.py](../experiments/tenstorrent/generate_decoder_fixture.py)
implements CPU fixture generation. It uses pretrained DINOv3-L at 512 pixels,
all three original GenRecon flows, the configured 12-step Euler/CFG schedules,
and the sparse-structure decoder. It saves the matching denormalized shape and
texture inputs before neural decoding or mesh extraction. The correct DINOv3-L
checkpoint is available locally at `/home/user/models/genrecon/dinov3-vitl16`.
The complete CPU generation path has executed successfully on the public poster
capture below. DL3DV-specific generation remains pending dataset access.

Use a CPU PyTorch environment with a matching torchvision build, Transformers
with DINOv3 support, safetensors, easydict, SciPy, and OpenCV. The tested
environment is `/home/user/tt-metal/python_env/bin/python`; no CUDA device or
TT device is needed for fixture generation. The flow models are promoted to
FP32 for execution on this CPU, which has no native BF16 arithmetic. This is a
CPU-generated fixture, not a claim of bitwise equivalence to CUDA/BF16 sampling.

```bash
python experiments/tenstorrent/generate_decoder_fixture.py \
    --transforms /path/to/scene/colmap/transforms.json \
    --image-dir /path/to/scene/colmap/images_8 \
    --points3d /path/to/scene/colmap/colmap/sparse/0/points3D.bin \
    --views 8 --threads 16 \
    --output-dir /path/to/new-cpu-fixture-directory
```

The `images.bin` from the same COLMAP reconstruction must sit beside
`points3D.bin`. Matching camera pairs recover and verify the similarity transform
between the raw COLMAP world and the camera JSON before selecting a crop; this
avoids assuming that their world axes, origin, or scale already agree. The two
CPU preprocessing tests cover this alignment and projection after image cropping.

The default crop is one cube enclosing the 5th–95th percentile of aligned points,
with a 10% margin. An explicit `--center X Y Z --extent LENGTH` can select a
specific cube instead. This is a bounded real-image fixture, not GenRecon's
full-scene chunk selection. The script records images, camera transformation,
checkpoint provenance, sampling settings, and hashes. Use its resulting
`decoder_input.npz` with the validation command below.

## Public scene used while DL3DV access is pending

The Nerfstudio team's [poster capture](https://huggingface.co/datasets/nerfstudioteam/datasets/tree/main/poster)
is available without a dataset access gate. The local source is pinned to
revision `461701c17e83c3f4d2481db32315aa7df703d2f8`, under
`/home/user/tt-scratchpad/genrecon/poster-fixture/source`. The upload contains
100 of the 226 camera frames listed in its JSON. `transforms_available.json`
retains only frames present in `images_2`; eight evenly spaced entries from this
filtered list are used. The original JSON is retained unchanged.

The exact source revision, original JSON hash, and selected filenames are in
[poster-source.json](../experiments/tenstorrent/results/sparse-decoder-integration/poster-source.json).
Run the generator with the filtered JSON, `poster/images_2`, and
`poster/colmap/sparse/0/points3D.bin`. All three 12-step pretrained samplers and
the occupancy decoder completed on CPU in **2,034 seconds (33.9 minutes)** using
16 threads. The resulting pair contains **4,490 unique coordinates** and two
finite float32 `[4490,32]` feature arrays, with denormalization already applied.
No voxel rows were truncated.

The reusable 1.07 MB fixture is at:

```text
/home/user/tt-scratchpad/genrecon/poster-fixture/cpu-available/decoder_input.npz
```

Its SHA-256 is `ec164c1e73ee3f9117068a45e2e75100f26345b25aba5f954ca6db91d3d69d41`.
[poster-cpu-fixture.json](../experiments/tenstorrent/results/sparse-decoder-integration/poster-cpu-fixture.json)
records the full generation provenance, package versions, camera and image
hashes, sampling schedules, checkpoint hashes, and array checks. Raw images,
conditioning tensors, and intermediate latents remain outside git. This is a
separate real capture and does not establish DL3DV-specific reconstruction quality.

## Validation command

In the TT-Metal Python environment with this GenRecon branch and the matching
FlexGEMM Tenstorrent branch installed:

```bash
TT_METAL_WATCHER=120 python experiments/tenstorrent/sparse_decoder_probe.py \
    --latent /path/to/decoder_input.npz \
    --precision accurate --row-chunk 1024 \
    --sequential-weights --warm-repeats 0 \
    --output-dir /path/to/new-validation-directory
```

The output directory must be new. Validation compares complete shape and texture
forwards, fixed and predicted subdivision schedules, exact output coordinates,
subdivision/edge signs, and continuous features. Sequential weight preparation
releases shape weights before loading texture weights, as tested on the large
poster fixture. This mode requires zero warm repeats; smaller fixtures can omit
both options to collect warm timings with both decoders resident. Raw inputs
and outputs remain in that directory; only curated numerical reports belong in
git. Runs without `--latent` remain explicitly synthetic. A single saved-latent
test does not establish quality or capacity for every deployment scene.

The saved poster references can be reused without regenerating images, latents,
or CPU decoder outputs:

```bash
TT_METAL_WATCHER=30 python experiments/tenstorrent/sparse_decoder_scene_probe.py \
    --reference-dir /home/user/tt-scratchpad/genrecon/sparse-decoder-poster-bounded \
    --output-dir /path/to/new-poster-follow-up-directory
```

This follow-up checks reference/checkpoint/source provenance, reuses the recorded
fixed-shape comparison, and runs predicted shape plus fixed/predicted texture
with sequential weights. Its recorded result is `numerical_failure`, not a
runtime failure: two CPU voxels are missing, three are added, and 11 common
geometry-edge signs differ. A nonzero exit is expected while that strict parity
limitation remains. Do not relax the gate or threshold to relabel the result.
