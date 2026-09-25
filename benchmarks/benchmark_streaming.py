from __future__ import annotations

import argparse
import json
import math
import time
import tracemalloc
from pathlib import Path

import numpy as np

from histopathology_pipeline.config import PatchExtractionConfig
from histopathology_pipeline.mask import TissueMask, select_tissue_coordinates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark streaming tissue-coordinate selection."
    )
    parser.add_argument("--slide-width", type=int, default=32768)
    parser.add_argument("--slide-height", type=int, default=32768)
    parser.add_argument("--mask-width", type=int, default=1024)
    parser.add_argument("--mask-height", type=int, default=1024)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--max-patches", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.seed)

    # Synthetic mask keeps the benchmark distributable and independent of
    # private/large WSI data while exercising the same coordinate-selection code.
    mask_array = rng.random((args.mask_height, args.mask_width)) > 0.30
    tissue_mask = TissueMask(
        mask=mask_array,
        slide_width=args.slide_width,
        slide_height=args.slide_height,
    )
    config = PatchExtractionConfig(
        patch_size=args.patch_size,
        stride=args.stride,
        min_tissue_fraction=0.50,
        max_patches=args.max_patches,
        seed=args.seed,
    )

    grid_columns = (
        math.floor((args.slide_width - args.patch_size) / args.stride) + 1
    )
    grid_rows = (
        math.floor((args.slide_height - args.patch_size) / args.stride) + 1
    )
    grid_positions = max(grid_columns, 0) * max(grid_rows, 0)

    tracemalloc.start()
    started = time.perf_counter()
    coordinates = select_tissue_coordinates(tissue_mask, config)
    elapsed = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result = {
        "slide_dimensions": [args.slide_width, args.slide_height],
        "mask_dimensions": [args.mask_width, args.mask_height],
        "patch_size": args.patch_size,
        "stride": args.stride,
        "grid_positions_scanned": grid_positions,
        "selected_coordinates": len(coordinates),
        "max_patches": args.max_patches,
        "seed": args.seed,
        "elapsed_seconds": elapsed,
        "grid_positions_per_second": (
            grid_positions / elapsed if elapsed > 0 else None
        ),
        "python_peak_memory_megabytes": peak_bytes / (1024 * 1024),
    }

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
