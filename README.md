# Machine Learning in Digital Histopathology

This repository contains the original 2018–2019 digital-histopathology research scripts together with a new maintained package for reproducible, memory-aware whole-slide-image (WSI) patch extraction.

The modernization deliberately preserves the historical scripts while separating them from code that is intended to be portable, testable, and reusable.

## Project context

The historical project explored an unsupervised WSI workflow that included:

- tissue/background preprocessing;
- artifact handling;
- tissue-region segmentation;
- patch extraction;
- patch augmentation;
- convolutional-autoencoder experiments;
- unsupervised feature clustering;
- mapping patch-level outputs back to slides.

The original project notes describe work on hundreds of large Hamamatsu-format WSIs and 256×256-pixel patch extraction. The raw WSIs and validation materials are not included in this public repository, so historical performance/result claims cannot be independently reproduced from this repository alone.

## What is maintained now

The maintained code lives under:

```text
src/histopathology_pipeline/
```

It currently provides:

- typed patch-extraction configuration;
- recursive, portable WSI discovery;
- optional lazy OpenSlide access;
- mapping of downsampled tissue masks to level-0 slide coordinates;
- streaming coordinate generation;
- O(k)-memory reservoir sampling when a maximum patch count is requested;
- deterministic train/test splits;
- deterministic rotations/reflections;
- streaming patch extraction directly to disk;
- a CSV manifest with patch coordinates, tissue fraction, and augmentation metadata;
- automated tests, CI, Docker, and a reproducible synthetic benchmark.

The modern path is intentionally focused on data/patch engineering first. Historical autoencoder, clustering, and preprocessing experiments remain available as archival research code until they can be migrated without silently changing their scientific behavior.

## Architecture

```text
WSI file
   |
   | dimensions only
   v
downsampled tissue mask
   |
   v
stream candidate grid coordinates
   |
   +---- no patch limit ----------> lazy coordinate iterator
   |
   +---- max patch limit ---------> seeded reservoir sampling, O(k) memory
                                         |
                                         v
                                   OpenSlide read_region
                                         |
                                         v
                                optional seeded transform
                                         |
                                         v
                              PNG patch + CSV manifest
```

This design avoids loading a complete multi-gigabyte WSI into a NumPy array.

## Repository structure

```text
.
├── src/
│   └── histopathology_pipeline/
│       ├── augmentation.py
│       ├── cli.py
│       ├── config.py
│       ├── mask.py
│       ├── pipeline.py
│       ├── sampling.py
│       └── wsi.py
├── tests/
├── benchmarks/
│   └── benchmark_streaming.py
├── scripts/
│   └── audit_legacy_paths.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── BENCHMARKS.md
├── LEGACY.md
├── MIGRATION.md
└── historical root-level research scripts
```

## Installation

### Core package

```bash
git clone https://github.com/seirana/MachineLearningInDigitalHistopathology.git
cd MachineLearningInDigitalHistopathology

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
```

The core package is intentionally lightweight and can be tested without OpenSlide.

### WSI extraction support

For real NDPI/SVS/TIFF reading, install the OpenSlide system library and the optional Python dependency.

On Debian/Ubuntu-like systems:

```bash
sudo apt-get update
sudo apt-get install -y libopenslide0

python -m pip install -e ".[wsi]"
```

## Patch extraction

The maintained CLI expects:

1. a WSI file;
2. a grayscale tissue-mask image;
3. an output directory.

The mask can be downsampled. Its dimensions do not need to match the level-0 WSI; the package maps mask pixels to level-0 coordinates using the WSI dimensions.

Example:

```bash
histopathology-extract \
  --slide /path/to/slide.ndpi \
  --mask /path/to/tissue_mask.png \
  --output-dir outputs/slide_001 \
  --patch-size 256 \
  --stride 256 \
  --min-tissue-fraction 0.80 \
  --max-patches 5000 \
  --seed 42
```

Add `--augment` to apply one seeded rotation/reflection per selected patch.

The output directory contains patch PNG files and:

```text
manifest.csv
```

with the level-0 coordinates, patch size, estimated tissue fraction, and transform name.

## Why reservoir sampling matters

A large WSI can contain a very large number of eligible candidate positions. Building a full Python list of every eligible coordinate is unnecessary when only a bounded sample is required.

When `--max-patches K` is used, the package applies reservoir sampling:

```text
memory complexity ≈ O(K)
```

instead of materializing all eligible candidate coordinates.

The implementation is deterministic for a fixed seed.

## Testing

Install the development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run:

```bash
python -m pytest
python -m ruff check src tests benchmarks scripts
```

The tests cover:

- configuration validation;
- weighted allocation;
- deterministic reservoir sampling;
- deterministic train/test splitting;
- all eight rotation/reflection transforms;
- downsampled mask mapping;
- tissue-fraction filtering;
- WSI discovery;
- streaming patch writing and manifest generation.

Historical root-level scripts are not collected as the modern test suite.

## Continuous integration

GitHub Actions tests the maintained package on Python 3.10, 3.11, and 3.12.

CI also:

- checks the command-line interface;
- runs a synthetic streaming benchmark;
- uploads benchmark JSON as a workflow artifact;
- builds the OpenSlide Docker image.

## Benchmarking

A reproducible synthetic benchmark is provided so memory/throughput behavior can be measured without distributing private or multi-gigabyte WSI files:

```bash
python benchmarks/benchmark_streaming.py \
  --output benchmarks/results/streaming.json
```

See [BENCHMARKS.md](BENCHMARKS.md).

The benchmark reports measured runtime and Python-tracked peak memory for coordinate selection. It does not claim to represent WSI disk or OpenSlide decoding throughput.

## Docker

Build:

```bash
docker build -t digital-histopathology .
```

Run:

```bash
docker run --rm \
  -v "/path/to/data:/data:ro" \
  -v "$PWD/outputs:/outputs" \
  digital-histopathology \
  --slide /data/slide.ndpi \
  --mask /data/tissue_mask.png \
  --output-dir /outputs/slide_001 \
  --patch-size 256 \
  --stride 256 \
  --min-tissue-fraction 0.80 \
  --max-patches 5000 \
  --seed 42
```

The WSI input mount is read-only in this example.

## Historical scripts

Many root-level files are original exploratory scripts. Some contain user-specific absolute paths, older APIs, or import-time execution patterns.

They remain in the repository to preserve research history, but they are not presented as production-quality code.

See:

- [LEGACY.md](LEGACY.md) for the boundary between archival and maintained code;
- [MIGRATION.md](MIGRATION.md) for the mapping from historical scripts to the maintained architecture.

An informational audit can be run with:

```bash
python scripts/audit_legacy_paths.py
```

## Reproducibility principles

The maintained package follows several explicit rules:

- no hard-coded workstation paths;
- local seeded random-number generators;
- no process-global random-state mutation;
- synthetic tests that do not depend on private WSIs;
- bounded-memory coordinate sampling;
- explicit configuration through CLI arguments and dataclasses;
- package/dependency metadata in `pyproject.toml`;
- CI across multiple Python versions.

## Scope and limitations

This modernization does not convert the historical autoencoder or clustering experiments into a new model and does not invent new scientific results.

The current maintained package covers the WSI patch-engineering layer. Stain normalization, model training, self-supervised learning, clustering, uncertainty estimation, and slide-level model evaluation should be modernized separately and validated against appropriate datasets.

This repository is research software and is not a clinical diagnostic system.
