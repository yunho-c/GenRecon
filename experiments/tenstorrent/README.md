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

## Reproduce

Use the same environment that passed DINO; do not install the CUDA setup script.
The runner clears shell variables that would select a different TT-Metal runtime.

```bash
hf download microsoft/TRELLIS-image-large \
  --revision 25e0d31ffbebe4b5a97464dd851910efc3002d96 \
  --include 'ckpts/ss_dec_conv3d_16l8_fp16.*' \
  --local-dir "$HOME/models/genrecon/trellis-image-large"

/home/user/tt-forge-venv/bin/python experiments/tenstorrent/run.py \
  --output-dir "$HOME/tt-scratchpad/genrecon/new-run" --timeout 300
```

Every run needs a fresh output directory. `result.json` preserves the last phase
even after a Python exception, child-process crash or timeout. The parent returns
nonzero if any case fails, differs numerically, or uses reported fallback ops.
Inspect each result, not just a compiler success message.

Official GenRecon flow weights can be downloaded independently:

```bash
python experiments/tenstorrent/fetch_checkpoints.py \
  --destination "$HOME/models/genrecon"
```

This uses up to eight HTTP range requests, validates byte ranges against the
source size/ETag, retains completed chunks for resume, and records SHA256.
Weights and large compiler artifacts are kept outside the repository.

## Remaining investigation

- Finish downloading the three official adapted flow checkpoints; test trained
  aggregators and full occupancy/shape/texture denoisers, not only one block.
- Isolate projection's index/mask disagreement before using it in a pipeline.
- Try minimal, numerically checked compiler-compatibility changes for attention
  dispatch, complex RoPE, and the sparse tensor interface.
- Attempt the original sparse decoder and mesh/export dependency paths; record
  where they stop. CUDA extensions do not become TT implementations through
  `torch.compile` alone.
- Exercise the Euler/CFG sampler and overlap reductions separately.

Reference: [TT-XLA documentation](https://docs.tenstorrent.com/tt-xla/) and
[performance/precision options](https://docs.tenstorrent.com/tt-xla/performance.html).
