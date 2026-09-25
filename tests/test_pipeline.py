import csv

import numpy as np
from PIL import Image

from histopathology_pipeline.config import PatchExtractionConfig
from histopathology_pipeline.pipeline import extract_patches


class FakeSlide:
    def __init__(self, path):
        self.path = path
        self.dimensions = (4, 4)

    def read_patch_rgb(self, x, y, patch_size):
        value = x + y
        return np.full(
            (patch_size, patch_size, 3),
            value,
            dtype=np.uint8,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None


def test_extract_patches_streams_to_files_and_manifest(tmp_path, monkeypatch):
    slide_path = tmp_path / "slide.ndpi"
    slide_path.write_bytes(b"fake")

    mask_path = tmp_path / "mask.png"
    Image.fromarray(
        np.array(
            [
                [255, 255],
                [0, 0],
            ],
            dtype=np.uint8,
        )
    ).save(mask_path)

    output_dir = tmp_path / "patches"

    monkeypatch.setattr(
        "histopathology_pipeline.pipeline.OpenSlideSource",
        FakeSlide,
    )

    summary = extract_patches(
        slide_path,
        mask_path,
        output_dir,
        config=PatchExtractionConfig(
            patch_size=2,
            stride=2,
            min_tissue_fraction=1.0,
            seed=5,
        ),
        augment=False,
    )

    assert summary.patches_written == 2
    assert summary.manifest_path.exists()

    patch_files = sorted(output_dir.glob("*.png"))
    assert len(patch_files) == 2

    with summary.manifest_path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert [row["x"] for row in rows] == ["0", "2"]
    assert [row["y"] for row in rows] == ["0", "0"]
    assert {row["transform"] for row in rows} == {"identity"}
