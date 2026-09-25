from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .augmentation import TRANSFORM_NAMES, random_transform
from .config import PatchExtractionConfig
from .mask import TissueMask, select_tissue_coordinates
from .wsi import OpenSlideSource


@dataclass(frozen=True)
class ExtractionSummary:
    slide_path: Path
    output_dir: Path
    manifest_path: Path
    patches_written: int


def extract_patches(
    slide_path: str | Path,
    mask_path: str | Path,
    output_dir: str | Path,
    *,
    config: PatchExtractionConfig,
    augment: bool = False,
) -> ExtractionSummary:
    """Stream level-0 patches from a WSI directly to disk.

    The function never loads the full WSI into memory. A downsampled binary
    tissue mask can be used to filter coordinates before any WSI patch is read.
    """

    slide_path = Path(slide_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.seed)
    manifest_path = output_dir / "manifest.csv"
    written = 0

    with OpenSlideSource(slide_path) as slide:
        slide_width, slide_height = slide.dimensions
        tissue_mask = TissueMask.from_image(
            mask_path,
            slide_width=slide_width,
            slide_height=slide_height,
        )
        coordinates = select_tissue_coordinates(tissue_mask, config)

        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "patch_file",
                    "x",
                    "y",
                    "patch_size",
                    "tissue_fraction",
                    "transform",
                ],
            )
            writer.writeheader()

            for x, y in coordinates:
                patch = slide.read_patch_rgb(x, y, config.patch_size)
                transform_index = 0

                if augment:
                    patch, transform_index = random_transform(patch, rng)

                filename = (
                    f"{slide_path.stem}_x{x}_y{y}_"
                    f"t{transform_index}.png"
                )
                Image.fromarray(patch).save(output_dir / filename)

                writer.writerow(
                    {
                        "patch_file": filename,
                        "x": x,
                        "y": y,
                        "patch_size": config.patch_size,
                        "tissue_fraction": (
                            f"{tissue_mask.tissue_fraction(x, y, config.patch_size):.6f}"
                        ),
                        "transform": TRANSFORM_NAMES[transform_index],
                    }
                )
                written += 1

    return ExtractionSummary(
        slide_path=slide_path,
        output_dir=output_dir,
        manifest_path=manifest_path,
        patches_written=written,
    )
