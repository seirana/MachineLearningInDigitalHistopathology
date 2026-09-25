from __future__ import annotations

import argparse

from .config import PatchExtractionConfig
from .pipeline import extract_patches


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract level-0 WSI patches using a downsampled tissue mask."
        )
    )
    parser.add_argument("--slide", required=True, help="Path to NDPI/SVS/TIFF slide")
    parser.add_argument("--mask", required=True, help="Path to grayscale tissue-mask image")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--min-tissue-fraction", type=float, default=0.80)
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply one seeded rotation/reflection per extracted patch",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    config = PatchExtractionConfig(
        patch_size=args.patch_size,
        stride=args.stride,
        min_tissue_fraction=args.min_tissue_fraction,
        max_patches=args.max_patches,
        seed=args.seed,
    )

    summary = extract_patches(
        args.slide,
        args.mask,
        args.output_dir,
        config=config,
        augment=args.augment,
    )

    print(
        f"Wrote {summary.patches_written} patches to "
        f"{summary.output_dir}"
    )
    print(f"Manifest: {summary.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
