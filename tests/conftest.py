import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def colmap_scene(tmp_path):
    root = tmp_path / "capture"
    images, model = root / "rgb", root / "colmap"
    images.mkdir(parents=True)
    model.mkdir()
    width, height = 32, 24
    y, x = np.indices((height, width))
    names = ["frame 002.png", "frame 001.png"]  # Deliberately not lexical order.
    for i, name in enumerate(names):
        pixels = np.stack([x * 7, y * 9, (x + y + i * 20) * 3], axis=-1).astype(np.uint8)
        Image.fromarray(pixels).save(images / name)
    (model / "cameras.txt").write_text("# cameras\n7 PINHOLE 32 24 28 26 15 11\n")
    q = np.sqrt(0.5)
    (model / "images.txt").write_text(
        f"# images\n11 {q} {q} 0 0 0.5 1 3 7 {names[0]}\n\n29 {q} {q} 0 0 -0.5 1 3 7 {names[1]}\n\n"
    )
    points = np.stack(np.meshgrid(np.linspace(-2, 2, 9), np.linspace(0, 2, 5), np.linspace(0, 2, 5)), axis=-1).reshape(
        -1, 3
    )
    (model / "points3D.txt").write_text(
        "# points\n" + "".join(f"{i + 1} {p[0]} {p[1]} {p[2]} 128 128 128 0.0\n" for i, p in enumerate(points))
    )
    return root, images, model, points


@pytest.fixture
def scene(colmap_scene, tmp_path):
    from inference.frontends.colmap import infer

    _, images, model, _ = colmap_scene
    return infer([], {"model_dir": model, "image_root": images, "work_dir": tmp_path / "work"}).align([0, 0, 1])
