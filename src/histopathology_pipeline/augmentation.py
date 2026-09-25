from __future__ import annotations

import numpy as np


TRANSFORM_NAMES = (
    "identity",
    "rotate_90",
    "rotate_180",
    "rotate_270",
    "flip_horizontal",
    "flip_horizontal_rotate_90",
    "flip_horizontal_rotate_180",
    "flip_horizontal_rotate_270",
)


def transform_patch(patch: np.ndarray, transform_index: int) -> np.ndarray:
    """Apply one of the eight square dihedral transforms without mutating input."""

    if patch.ndim not in (2, 3):
        raise ValueError("patch must be a 2D or 3D array")
    if not 0 <= transform_index < len(TRANSFORM_NAMES):
        raise ValueError("transform_index must be between 0 and 7")

    if transform_index < 4:
        transformed = np.rot90(patch, k=transform_index, axes=(0, 1))
    else:
        transformed = np.flip(patch, axis=1)
        transformed = np.rot90(
            transformed,
            k=transform_index - 4,
            axes=(0, 1),
        )

    return np.ascontiguousarray(transformed)


def random_transform(
    patch: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Apply a seeded random transform and return the transform index."""

    transform_index = int(rng.integers(0, len(TRANSFORM_NAMES)))
    return transform_patch(patch, transform_index), transform_index
