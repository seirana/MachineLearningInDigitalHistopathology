# Performance benchmark

Whole-slide images are too large to load eagerly in many practical workflows. The maintained pipeline therefore separates a small/downsampled tissue mask from patch reads and yields coordinates as a stream.

When a maximum number of patches is requested, reservoir sampling limits coordinate-selection memory to **O(k)**, where `k` is the requested patch count.

## Reproducible synthetic benchmark

The repository includes a data-independent benchmark:

```bash
python benchmarks/benchmark_streaming.py \
  --slide-width 32768 \
  --slide-height 32768 \
  --mask-width 1024 \
  --mask-height 1024 \
  --patch-size 256 \
  --stride 256 \
  --max-patches 1000 \
  --seed 42 \
  --output benchmarks/results/streaming.json
```

The result records:

- synthetic slide and mask dimensions;
- number of grid positions scanned;
- selected coordinate count;
- elapsed time;
- grid positions processed per second;
- peak Python-tracked memory;
- random seed.

The benchmark deliberately uses a synthetic mask so it can run in CI without distributing private or multi-gigabyte WSI data.

GitHub Actions runs a smaller benchmark smoke test and uploads the JSON result as an artifact. Performance numbers should be interpreted only for the hardware and software environment that produced them; the repository does not hard-code a claimed throughput value.

## What is and is not measured

This benchmark measures the coordinate-selection path, including mask mapping and reservoir sampling. It does **not** measure disk throughput, OpenSlide decoding, stain normalization, GPU training, or end-to-end WSI processing.

Those components depend strongly on slide format, storage, native libraries, CPU/GPU hardware, and model implementation, so they should be benchmarked separately when the corresponding modern modules are added.


## Reference CI measurement

A GitHub-hosted Ubuntu runner executed the benchmark on 2026-09-25 with:

- synthetic level-0 dimensions: 8192 × 8192;
- mask dimensions: 512 × 512;
- patch size: 256;
- stride: 128;
- maximum selected coordinates: 250;
- seed: 42.

That run scanned 3,969 grid positions and selected 250 coordinates in approximately 0.220 seconds, corresponding to about 18,048 grid positions/second. Python `tracemalloc` reported approximately 0.031 MiB peak Python-managed memory during the measured selection block.

These numbers are a **reference CI measurement, not a hardware-independent performance claim**. They measure the synthetic coordinate-selection benchmark only; they do not include OpenSlide decoding, file I/O, stain normalization, or model inference/training. The CI workflow uploads the raw JSON result as the `streaming-benchmark` artifact so future runs can be compared on their own runner environment.
