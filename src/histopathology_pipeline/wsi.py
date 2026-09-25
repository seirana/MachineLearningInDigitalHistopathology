from __future__ import annotations

from pathlib import Path

import numpy as np


SUPPORTED_WSI_EXTENSIONS = (
    ".ndpi",
    ".svs",
    ".tif",
    ".tiff",
)


def discover_slides(
    root: str | Path,
    extensions: tuple[str, ...] = SUPPORTED_WSI_EXTENSIONS,
) -> list[Path]:
    """Discover WSI files recursively using pathlib."""

    base = Path(root).expanduser().resolve()
    normalized = {extension.lower() for extension in extensions}

    return sorted(
        path
        for path in base.rglob("*")
        if path.is_file() and path.suffix.lower() in normalized
    )


class OpenSlideSource:
    """Small OpenSlide wrapper that keeps the native dependency optional."""

    def __init__(self, path: str | Path):
        try:
            import openslide
        except ImportError as exc:
            raise RuntimeError(
                "OpenSlide support is optional. Install the wsi extra and "
                "ensure the OpenSlide system library is installed."
            ) from exc

        self.path = Path(path).expanduser().resolve()
        self._slide = openslide.OpenSlide(str(self.path))

    @property
    def dimensions(self) -> tuple[int, int]:
        width, height = self._slide.dimensions
        return int(width), int(height)

    def read_patch_rgb(
        self,
        x: int,
        y: int,
        patch_size: int,
    ) -> np.ndarray:
        """Read one level-0 patch and return uint8 RGB."""

        region = self._slide.read_region(
            (int(x), int(y)),
            0,
            (int(patch_size), int(patch_size)),
        ).convert("RGB")

        return np.asarray(region, dtype=np.uint8)

    def close(self) -> None:
        self._slide.close()

    def __enter__(self) -> "OpenSlideSource":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
