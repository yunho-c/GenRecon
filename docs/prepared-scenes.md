# Prepared scenes

The Prepared input path lets camera-and-point providers share one interface to GenRecon. The initial implementation imports existing COLMAP text reconstructions. MapAnything, VGGT-Omega, and DA3 adapters remain planned; selecting those names is not supported yet.

**Prepare an existing reconstruction**

Use a Python environment with `requirements/preflight.txt` installed. The importer does not run COLMAP or load generator weights. Supply the directory containing `cameras.txt`, `images.txt`, and `points3D.txt`, and the explicit root for the image names in `images.txt`:

```sh
python prepare_scene.py \
  --frontend colmap \
  --model_dir capture/colmap \
  --images capture/rgb \
  --assume_z_up \
  --output prepared/capture
```

Use `--assume_z_up` only for a reconstruction already aligned to vertical +Z. Otherwise supply `--up_vector X Y Z`, replacing the placeholders with an up direction in the reconstruction's world frame. Cameras and points are rotated together. An optional positive `--scale` scales points and camera centers together; it does not turn an arbitrary scale into a calibrated metric estimate. Set `--scale_status externally_calibrated` only when the scale is established externally.

Image order follows `images.txt`, which is not guaranteed to be capture order. To control order and select a subset, pass `--frames frames.json`, a JSON list of at least two unique image names relative to `--images`. Names containing spaces and nested image paths are supported.

Supported camera models are PINHOLE, SIMPLE_PINHOLE, OPENCV, RADIAL, and SIMPLE_RADIAL. Distorted inputs are rectified with OpenCV and stored with updated pinhole intrinsics. Fisheye models are rejected. Images with nontrivial EXIF orientation are rejected: normalize the pixels and camera calibration explicitly before importing. Silent EXIF rotation would invalidate calibration.

By default preparation removes invalid points, keeps COLMAP points regardless of track length/reprojection threshold, and caps the cloud at one million points. `--max_reproj_error` and `--min_track_len` enable those COLMAP-specific filters. `--voxel_fraction` optionally deduplicates points at a fraction of the robust scene extent. Filtering and counts are recorded in the bundle.

The initial `--min_points_per_chunk` default is **1 after preparation**, intended to avoid importing a dense-cloud threshold into arbitrary sparse reconstructions. Choose a larger value for your data. The reconstruction command can override it. This is an explicit coverage policy, not a qualified quality default.

`--cleaning legacy` additionally runs the existing statistical and radius filters and requires Open3D. Its radius is in source world units, before `--scale`. Defaults match the existing filter implementation: 20 statistical neighbors, standard-deviation ratio 2, 10 radius neighbors, radius 0.1. Use the same thresholds, disable voxel deduplication, and avoid point capping when comparing with a legacy run. The ordinary `--cleaning none` path needs no Open3D installation.

**Inspect before generation**

```sh
python reconstruct_scene.py \
  --mode Prepared \
  --path prepared/capture \
  --output_path results/capture-preflight \
  --preflight
```

Preflight runs on CPU and does not import the generation pipeline or neural frontend packages. It validates the bundle, derives chunks, selects/crops images, and writes:

- `clean_points.ply`, `chunk_layout.png`, and `chunk_transforms.json`.
- `cameras.json`, including selected frame IDs and adjusted camera matrices.
- Selected scene images and per-chunk conditioning images.
- `scene_coordinates.json`, preserving scale status, both coordinate transforms, source provenance, and the point policy.
- `preflight.json`, with chunk/view counts and `generation_run: false`.

The current iPhone-style selection behavior is reused: rectangular frames produce two square crops by default. `--center_crop` selects the legacy single crop. `--num_imgs_per_scene` caps source frames before this expansion. `--max_chunks` defaults to 256 for Prepared input and rejects a larger grid before its Cartesian product is allocated. Empty clouds, zero vertical extent, zero retained chunks, and invalid view budgets produce errors before generator loading.

Inspect the upright orientation, bounds, retained chunks, and selected views. A successful preflight establishes valid preprocessing, not accurate cameras or a good final mesh.

**Generate using the saved bundle**

In the supported GenRecon GPU environment, omit `--preflight` and supply the usual checkpoints:

```sh
python reconstruct_scene.py \
  --mode Prepared \
  --path prepared/capture \
  --output_path results/capture \
  --ss_ckpt "$SS_CKPT" \
  --shape_ckpt "$SHAPE_CKPT" \
  --tex_ckpt "$TEX_CKPT"
```

Preparation and generation may use separate environments. The generator reads the bundle without importing its provider. Generation imports and model loading occur after preprocessing; selection and generation use explicitly seeded RNGs.

All generation artifacts stay in the aligned z-up scene frame. The inverse transform is preserved in `scene_coordinates.json` and in Prepared GLB/chunk input metadata. Do not insert that rotation into the axis-aligned voxel-volume origin: transforming back to the source frame belongs after final mesh/texturing export.

**Bundle and cache behavior**

The bundle contains `scene.json`, `geometry.npz`, and losslessly stored conditioning images under `images/`. Intrinsics are pixel-space matrices for those images, poses are rigid OpenCV camera-to-world matrices, and points use the same world frame. Bundle paths are relative, so a bundle remains usable after moving it or removing the original input files.

Source text files, selected images, preparation options, adapter code, and relevant package versions participate in the request fingerprint. Arrays and stored images have content checksums. Repeating a matching preparation validates and reuses the existing bundle; changed inputs/options or corruption cause an error. Choose a new output directory to recompute. Reconstruction outputs must be outside the bundle directory.

**Validation status**

The first slice has CPU tests for analytic reprojection, crop/resize transforms, world alignment and scale, serialization/corruption failures, COLMAP parsing/undistortion, chunk limits, cache reuse, and model-free CLI preflight. Legacy and Prepared conditioning are compared with the same points, settings, and seed.

A separate comparison against the untouched `719cfb3` loader and chunker produced identical conditioning tensors and chunk transforms for both crop modes on the six-chunk synthetic fixture (maximum tensor difference 0). This is fixture-level compatibility evidence. Full generation, GLB baking, and the optional Open3D cleaning path are not runtime-qualified by this CPU validation.

Run the focused suite with:

```sh
python -m pip install -r requirements/preflight.txt pytest
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLBACKEND=Agg python -m pytest -q tests
```

Plugin autoload is disabled to keep unrelated installed pytest plugins out of the test process. See the [implementation plan](ff-loc-plan.md) for the remaining GPU qualification and neural adapters.
