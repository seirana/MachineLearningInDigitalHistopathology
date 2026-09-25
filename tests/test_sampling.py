import numpy as np
import pytest

from histopathology_pipeline.sampling import (
    allocate_by_weights,
    reservoir_sample,
    train_test_split_indices,
)


def test_allocate_by_weights_preserves_total():
    allocation = allocate_by_weights(11, [0.5, 0.3, 0.2])

    assert allocation.sum() == 11
    assert allocation.tolist() == [6, 3, 2]


def test_allocate_by_weights_rejects_zero_sum():
    with pytest.raises(ValueError, match="positive"):
        allocate_by_weights(10, [0, 0])


def test_reservoir_sample_is_deterministic_and_bounded():
    items = range(10_000)

    first = reservoir_sample(items, k=25, seed=7)
    second = reservoir_sample(range(10_000), k=25, seed=7)

    assert first == second
    assert len(first) == 25
    assert len(set(first)) == 25


def test_train_test_split_is_deterministic_and_complete():
    train_1, test_1 = train_test_split_indices(20, 0.8, seed=11)
    train_2, test_2 = train_test_split_indices(20, 0.8, seed=11)

    np.testing.assert_array_equal(train_1, train_2)
    np.testing.assert_array_equal(test_1, test_2)
    assert len(train_1) == 16
    assert len(test_1) == 4
    assert sorted(np.concatenate([train_1, test_1]).tolist()) == list(range(20))
