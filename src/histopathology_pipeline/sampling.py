from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from typing import TypeVar

import numpy as np


T = TypeVar("T")


def allocate_by_weights(total: int, weights: Sequence[float]) -> np.ndarray:
    """Allocate an integer total proportionally using the largest-remainder method."""

    if total < 0:
        raise ValueError("total must be non-negative")

    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("weights must be a non-empty 1D sequence")
    if np.any(values < 0):
        raise ValueError("weights must be non-negative")

    weight_sum = float(values.sum())
    if weight_sum <= 0:
        raise ValueError("at least one weight must be positive")

    exact = values / weight_sum * total
    allocated = np.floor(exact).astype(int)
    remaining = total - int(allocated.sum())

    if remaining:
        order = np.argsort(-(exact - allocated), kind="stable")
        allocated[order[:remaining]] += 1

    return allocated


def reservoir_sample(items: Iterable[T], k: int, seed: int) -> list[T]:
    """Sample k items from a stream using O(k) memory."""

    if k <= 0:
        raise ValueError("k must be greater than 0")

    rng = random.Random(seed)
    sample: list[T] = []

    for index, item in enumerate(items):
        if index < k:
            sample.append(item)
            continue

        replacement_index = rng.randint(0, index)
        if replacement_index < k:
            sample[replacement_index] = item

    return sample


def train_test_split_indices(
    n_items: int,
    train_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic train/test index arrays."""

    if n_items <= 0:
        raise ValueError("n_items must be greater than 0")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_items)
    n_train = int(round(n_items * train_fraction))
    n_train = min(max(n_train, 1), n_items - 1)

    train = np.sort(shuffled[:n_train])
    test = np.sort(shuffled[n_train:])
    return train, test
