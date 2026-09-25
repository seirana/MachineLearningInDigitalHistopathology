from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PatchExtractionConfig:
    """Configuration for level-0 patch extraction."""

    patch_size: int = 256
    stride: int = 256
    min_tissue_fraction: float = 0.80
    max_patches: int | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        if self.patch_size <= 0:
            raise ValueError("patch_size must be greater than 0")
        if self.stride <= 0:
            raise ValueError("stride must be greater than 0")
        if not 0.0 <= self.min_tissue_fraction <= 1.0:
            raise ValueError("min_tissue_fraction must be between 0 and 1")
        if self.max_patches is not None and self.max_patches <= 0:
            raise ValueError("max_patches must be greater than 0 when provided")
