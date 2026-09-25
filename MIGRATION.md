# Modernization map

This repository now separates the historical research record from a maintained, testable core.

## Current modernization scope

The first modernization stage focuses on the part of the historical project where software-engineering quality matters most for large WSI workflows: **patch discovery, sampling, augmentation, and extraction**.

```text
downsampled tissue mask
        |
        v
stream candidate coordinates
        |
        +---- no limit ----------> yield coordinates lazily
        |
        +---- max_patches -------> reservoir sample with O(k) memory
                                      |
                                      v
                                OpenSlide patch read
                                      |
                                      v
                           seeded augmentation (optional)
                                      |
                                      v
                              patch file + CSV manifest
```

## Design decisions

### No hard-coded workstation paths

All maintained functions accept paths as arguments or use `pathlib`. No `/home/<user>/...` path is required.

### Lazy WSI access

The WSI is opened once and patches are read individually. The maintained extraction code never converts the entire WSI into a NumPy array.

### Downsampled masks

A tissue mask can be much smaller than the level-0 slide. `TissueMask` maps mask pixels back to level-0 coordinates using the slide dimensions.

### Reproducible randomness

Randomness is local and explicitly seeded. The maintained code does not modify Python's process-wide global random state.

### Bounded sampling memory

When `max_patches` is set, eligible coordinates are reservoir-sampled while they stream past, rather than storing every eligible coordinate first.

### Testable without private WSIs

Core mask, sampling, augmentation, and pipeline behavior is tested with synthetic arrays and a fake slide reader. OpenSlide itself remains an optional integration dependency.

## Future modernization

The historical repository also contains preprocessing, autoencoder, feature-clustering, and visualization experiments. Those should be migrated only when their exact scientific behavior and data assumptions can be validated. They are not silently rewritten here because doing so could change the original research logic.
