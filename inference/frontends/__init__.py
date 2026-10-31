"""Fixed, lazy frontend registry. Importing it does not load any model."""

from importlib import import_module
from pathlib import Path
from typing import Protocol, Sequence

from ..scene_geometry import SceneGeometry


class Frontend(Protocol):
    def __call__(self, frames: Sequence[Path], config: dict) -> SceneGeometry: ...


_FRONTENDS = {"colmap": ".colmap", "mapanything": ".mapanything"}


def available_frontends() -> tuple[str, ...]:
    return tuple(_FRONTENDS)


def get_frontend(name: str) -> Frontend:
    if name not in _FRONTENDS:
        raise ValueError(f"Unsupported frontend {name!r}; available: {', '.join(_FRONTENDS)}")
    return import_module(_FRONTENDS[name], __name__).infer
