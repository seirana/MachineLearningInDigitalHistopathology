# Legacy research scripts

The root-level Python files in this repository are preserved as historical research artifacts from the original 2018–2019 project.

They are **not** the maintained software interface.

## Why they are still here

The original scripts document how the research workflow evolved: WSI reading, tissue segmentation, patch sampling, augmentation, autoencoder experiments, clustering, and exploratory analysis. Removing them would erase useful project history.

However, many of those scripts reflect the computing environment of that period. Examples include:

- machine-specific paths such as `/home/.../`;
- direct execution at import time;
- old NumPy serialization patterns;
- older TensorFlow/Keras APIs;
- mixed exploratory and reusable logic;
- state persisted through `.npy` files;
- script-to-script imports from the repository root.

These characteristics make them unsuitable as the current public API.

## Maintained replacement

New reusable code lives in:

```text
src/histopathology_pipeline/
```

The maintained package currently covers:

- typed extraction configuration;
- portable WSI discovery;
- lazy OpenSlide access;
- downsampled tissue-mask mapping;
- streaming patch-coordinate generation;
- O(k)-memory reservoir sampling;
- deterministic train/test splitting;
- deterministic rotations/reflections;
- streaming WSI patch extraction;
- per-run patch manifests.

The modern code contains no user-specific absolute paths.

## Mapping from historical scripts

| Historical script | Maintained replacement / direction |
|---|---|
| `OpenSlide_reader.py` | `histopathology_pipeline.wsi.OpenSlideSource` |
| `Random_Rotation_Mirroring.py` | `histopathology_pipeline.augmentation` |
| `sizebased_sampling.py` | `histopathology_pipeline.sampling` |
| `get_patch_tissues.py` | `histopathology_pipeline.mask` + `pipeline` |
| `patch_batch_generator.py` | streaming extraction in `pipeline.py` |
| preprocessing prototypes | future modular preprocessing package |
| autoencoder / clustering prototypes | retained as historical ML experiments |

## Auditing the historical scripts

Run:

```bash
python scripts/audit_legacy_paths.py
```

This reports machine-specific `/home/.../` paths still present in historical root scripts. The audit is informational: CI tests and lints the maintained package rather than pretending the archival scripts are production-ready.
