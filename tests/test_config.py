import pytest

from histopathology_pipeline.config import PatchExtractionConfig


def test_default_config_is_valid():
    config = PatchExtractionConfig()

    assert config.patch_size == 256
    assert config.stride == 256
    assert config.min_tissue_fraction == 0.80
    assert config.max_patches is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"patch_size": 0}, "patch_size"),
        ({"stride": 0}, "stride"),
        ({"min_tissue_fraction": -0.1}, "min_tissue_fraction"),
        ({"min_tissue_fraction": 1.1}, "min_tissue_fraction"),
        ({"max_patches": 0}, "max_patches"),
    ],
)
def test_invalid_config_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PatchExtractionConfig(**kwargs)
