import numpy as np

from histopathology_pipeline.config import PatchExtractionConfig
from histopathology_pipeline.mask import (
    TissueMask,
    iter_tissue_coordinates,
    select_tissue_coordinates,
)


def test_tissue_fraction_maps_downsampled_mask_to_slide():
    mask = TissueMask(
        mask=np.array(
            [
                [1, 1],
                [0, 0],
            ],
            dtype=bool,
        ),
        slide_width=4,
        slide_height=4,
    )

    assert mask.tissue_fraction(0, 0, 2) == 1.0
    assert mask.tissue_fraction(0, 2, 2) == 0.0


def test_coordinate_iteration_filters_by_tissue_fraction():
    mask = TissueMask(
        mask=np.array(
            [
                [1, 1],
                [0, 0],
            ],
            dtype=bool,
        ),
        slide_width=4,
        slide_height=4,
    )
    config = PatchExtractionConfig(
        patch_size=2,
        stride=2,
        min_tissue_fraction=1.0,
    )

    assert list(iter_tissue_coordinates(mask, config)) == [
        (0, 0),
        (2, 0),
    ]


def test_max_patches_uses_seeded_reservoir_sampling():
    mask = TissueMask(
        mask=np.ones((8, 8), dtype=bool),
        slide_width=8,
        slide_height=8,
    )
    config = PatchExtractionConfig(
        patch_size=2,
        stride=2,
        min_tissue_fraction=1.0,
        max_patches=3,
        seed=99,
    )

    first = select_tissue_coordinates(mask, config)
    second = select_tissue_coordinates(mask, config)

    assert first == second
    assert len(first) == 3
