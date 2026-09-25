import numpy as np
import pytest

from histopathology_pipeline.augmentation import (
    TRANSFORM_NAMES,
    random_transform,
    transform_patch,
)


def test_all_transforms_preserve_shape_dtype_and_input():
    patch = np.arange(3 * 3 * 3, dtype=np.uint8).reshape(3, 3, 3)
    original = patch.copy()

    outputs = [
        transform_patch(patch, index)
        for index in range(len(TRANSFORM_NAMES))
    ]

    for output in outputs:
        assert output.shape == patch.shape
        assert output.dtype == patch.dtype
        assert output.flags.c_contiguous

    np.testing.assert_array_equal(patch, original)


def test_random_transform_is_reproducible():
    patch = np.arange(16, dtype=np.uint8).reshape(4, 4)
    rng_1 = np.random.default_rng(123)
    rng_2 = np.random.default_rng(123)

    transformed_1, index_1 = random_transform(patch, rng_1)
    transformed_2, index_2 = random_transform(patch, rng_2)

    assert index_1 == index_2
    np.testing.assert_array_equal(transformed_1, transformed_2)


def test_invalid_transform_is_rejected():
    with pytest.raises(ValueError, match="transform_index"):
        transform_patch(np.zeros((4, 4), dtype=np.uint8), 8)
