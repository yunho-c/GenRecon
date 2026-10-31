import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import prepare_scene
from inference.frontends.mapanything import adapt_predictions, checkpoint_identity, preprocessing


def inputs(tmp_path):
    frames, predictions = [], []
    for i in range(2):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (31, 25), (30, 60, 90)).save(path)
        frames.append(path)
        xyz = np.indices((4, 6)).transpose(1, 2, 0)
        xyz = np.concatenate([xyz, np.ones((4, 6, 1))], -1).astype(np.float32)
        predictions.append(dict(img_no_norm=np.full((1, 4, 6, 3), .5), pts3d=xyz[None],
                                depth_z=np.ones((1, 4, 6, 1)), mask=np.ones((1, 4, 6, 1)),
                                camera_poses=np.eye(4)[None],
                                intrinsics=np.array([[[5., 0, 2], [0, 5, 1], [0, 0, 1]]])))
    config = dict(work_dir=tmp_path / "work", conditioning="processed", max_points=40,
                  mask_edges=True, apply_confidence_mask=False, confidence_percentile=10)
    return frames, predictions, config


def test_adapter_masks_and_preserves_camera_convention(tmp_path):
    frames, pred, cfg = inputs(tmp_path)
    pred[0]["pts3d"][0, 0, 0] = np.nan
    pred[0]["depth_z"][0, 0, 1] = -1
    pred[0]["mask"][0, 0, 2] = 0
    scene = adapt_predictions(frames, pred, cfg)
    assert len(scene.points) == 40
    assert scene.point_policy["per_frame"][0]["retained"] == 21
    np.testing.assert_equal(scene.c2w[0], np.eye(4))
    assert scene.frames[0].width == 6
    assert scene.scale_status == "predicted_metric"
    assert not scene.aligned


def test_source_resolution_reprojection(tmp_path):
    frames, pred, cfg = inputs(tmp_path)
    cfg["conditioning"] = "source"
    scene = adapt_predictions(frames, pred, cfg)
    affine, _ = preprocessing((31, 25), (6, 4))
    np.testing.assert_allclose(affine @ scene.intrinsics[0], pred[0]["intrinsics"][0], atol=1e-6)
    assert (scene.frames[0].width, scene.frames[0].height) == (31, 25)


@pytest.mark.parametrize("case", ["count", "pose", "depth", "empty"])
def test_malformed_outputs(tmp_path, case):
    frames, pred, cfg = inputs(tmp_path)
    if case == "count":
        pred.pop()
    elif case == "pose":
        pred[0]["camera_poses"][0, 0, 0] = 2
    elif case == "depth":
        pred[0]["depth_z"] = np.ones((4, 6))
    else:
        for p in pred:
            p["mask"][:] = 0
    with pytest.raises(ValueError):
        adapt_predictions(frames, pred, cfg)


def test_frame_discovery_manifest_and_precedence(tmp_path):
    for i in range(6):
        Image.new("RGB", (2, 2)).save(tmp_path / f"{i}.png")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"max_frames": 3}))
    args = prepare_scene.parse_args(["--frontend", "mapanything", "--images", str(tmp_path),
                                    "--output", str(tmp_path / "out"), "--assume_z_up", "--config", str(config)])
    assert prepare_scene.select_frames(args)[0] == ["0.png", "2.png", "5.png"]
    manifest = tmp_path / "frames.json"
    manifest.write_text(json.dumps(["5.png", "1.png", "4.png"]))
    args.frames, args.max_frames = str(manifest), 2
    assert prepare_scene.select_frames(args)[0] == ["5.png", "4.png"]
    assert args.scale_status == "predicted_metric"


@pytest.mark.parametrize("config", [{"max_frames": "16"}, {"mask_edges": "false"}, {"typo": 1}])
def test_bad_configuration(tmp_path, config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        prepare_scene.parse_args(["--config", str(path)])


def test_pinned_checkpoint_required():
    with pytest.raises(ValueError, match="immutable"):
        checkpoint_identity({"revision": "main"})


def test_cache_reuse_does_not_infer(colmap_scene, tmp_path, monkeypatch):
    _, images, model, _ = colmap_scene
    args = prepare_scene.parse_args(["--images", str(images), "--model_dir", str(model),
                                    "--output", str(tmp_path / "bundle"), "--assume_z_up"])
    prepare_scene.prepare(args)
    monkeypatch.setattr(prepare_scene, "get_frontend", lambda _: pytest.fail("cache hit loaded frontend"))
    report = {}
    prepare_scene.prepare(args, report)
    assert report["cache_reused"]
    args.max_points = 2
    with pytest.raises(ValueError, match="fingerprint"):
        prepare_scene.prepare(args)


def test_exif_source_conditioning_is_normalized(tmp_path):
    frames, pred, cfg = inputs(tmp_path)
    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (31, 25)).save(path, exif=exif)
    frames[0] = path
    cfg["conditioning"] = "source"
    scene = adapt_predictions(frames, pred, cfg)
    assert (scene.frames[0].width, scene.frames[0].height) == (25, 31)
    assert scene.frames[0].preprocessing["exif_orientation"] == 6
    scene.validate()


def test_half_pixel_transform_has_expected_pixel_centers():
    # 8x6 -> 4x3: source centers .5,2.5 map to destination centers 0,1.
    affine, _ = preprocessing((8, 6), (4, 3))
    np.testing.assert_allclose(affine @ [.5, .5, 1], [0, 0, 1])
    np.testing.assert_allclose(affine @ [2.5, 2.5, 1], [1, 1, 1])


@pytest.mark.parametrize("size", [(641, 479), (479, 641), (960, 540), (31, 25)])
def test_preprocessing_against_optional_upstream(size):
    pytest.importorskip("mapanything")
    from mapanything.utils.cropping import crop_resize_if_necessary

    image = Image.fromarray(np.random.default_rng(4).integers(0, 256, (*size[::-1], 3), dtype=np.uint8))
    target = (518, 392)
    _, info = preprocessing(size, target)
    resized = image.resize(tuple(info["resized_size"]),
                           Image.Resampling.LANCZOS if max(np.array(target) / size) < 1 else Image.Resampling.BICUBIC)
    x, y = info["crop_offset"]
    expected = resized.crop((x, y, x + target[0], y + target[1]))
    actual = crop_resize_if_necessary(image, target)[0]
    np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))
