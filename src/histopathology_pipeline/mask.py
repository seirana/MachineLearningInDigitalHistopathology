from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import PatchExtractionConfig
from .sampling import reservoir_sample


@dataclass(frozen=True)
class TissueMask:
    """Downsampled tissue mask mapped to level-0 slide coordinates."""

    mask: np.ndarray
    slide_width: int
    slide_height: int

    def __post_init__(self) -> None:
        if self.mask.ndim != 2:
            raise ValueError("mask must be a 2D array")
        if self.mask.size == 0:
            raise ValueError("mask must not be empty")
        if self.slide_width <= 0 or self.slide_height <= 0:
            raise ValueError("slide dimensions must be greater than 0")

        object.__setattr__(self, "mask", self.mask.astype(bool, copy=False))

    @classmethod
    def from_image(
        cls,
        path: str | Path,
        *,
        slide_width: int,
        slide_height: int,
        threshold: int = 0,
    ) -> "TissueMask":
        """Load a grayscale mask image; pixels above threshold are tissue."""

        image = Image.open(path).convert("L")
        array = np.asarray(image)
        return cls(
            mask=array > threshold,
            slide_width=slide_width,
            slide_height=slide_height,
        )

    @property
    def scale_x(self) -> float:
        return self.slide_width / self.mask.shape[1]

    @property
    def scale_y(self) -> float:
        return self.slide_height / self.mask.shape[0]

    def tissue_fraction(self, x: int, y: int, patch_size: int) -> float:
        """Estimate tissue coverage for a level-0 square patch."""

        if patch_size <= 0:
            raise ValueError("patch_size must be greater than 0")
        if x < 0 or y < 0:
            raise ValueError("x and y must be non-negative")
        if x + patch_size > self.slide_width or y + patch_size > self.slide_height:
            raise ValueError("patch extends outside slide dimensions")

        left = max(0, int(math.floor(x / self.scale_x)))
        right = min(
            self.mask.shape[1],
            int(math.ceil((x + patch_size) / self.scale_x)),
        )
        top = max(0, int(math.floor(y / self.scale_y)))
        bottom = min(
            self.mask.shape[0],
            int(math.ceil((y + patch_size) / self.scale_y)),
        )

        region = self.mask[top:bottom, left:right]
        if region.size == 0:
            return 0.0
        return float(region.mean())


def iter_tissue_coordinates(
    tissue_mask: TissueMask,
    config: PatchExtractionConfig,
) -> Iterator[tuple[int, int]]:
    """Yield eligible level-0 patch coordinates without materializing all patches."""

    max_x = tissue_mask.slide_width - config.patch_size
    max_y = tissue_mask.slide_height - config.patch_size

    if max_x < 0 or max_y < 0:
        return

    for y in range(0, max_y + 1, config.stride):
        for x in range(0, max_x + 1, config.stride):
            if (
                tissue_mask.tissue_fraction(x, y, config.patch_size)
                >= config.min_tissue_fraction
            ):
                yield x, y


def select_tissue_coordinates(
    tissue_mask: TissueMask,
    config: PatchExtractionConfig,
):
    """Return a stream, or a bounded reservoir sample when max_patches is set."""

    coordinates = iter_tissue_coordinates(tissue_mask, config)
    if config.max_patches is None:
        return coordinates

    return reservoir_sample(
        coordinates,
        k=config.max_patches,
        seed=config.seed,
    )
