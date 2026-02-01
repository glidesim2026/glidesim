import numpy as np

from glidesim.ui.results import find_run_index, RunSelection


def test_find_run_index_best():
    portfolio_values = np.array(
        [
            [100, 150],
            [100, 200],
            [100, 100],
        ]
    )

    idx = find_run_index(portfolio_values, RunSelection.BEST)

    assert idx == 1


def test_find_run_index_worst_simple():
    portfolio_values = np.array(
        [
            [100, 150],
            [100, 50],
            [100, 100],
        ]
    )

    idx = find_run_index(portfolio_values, RunSelection.WORST)

    assert idx == 1


def test_find_run_index_worst_multiple_zeros():
    portfolio_values = np.array(
        [
            [100, 50, 0, 0],
            [100, 0, 0, 0],
            [100, 80, 60, 0],
        ]
    )

    idx = find_run_index(portfolio_values, RunSelection.WORST)

    assert idx == 1


def test_find_run_index_median():
    portfolio_values = np.array(
        [
            [100, 100],
            [100, 200],
            [100, 150],
        ]
    )

    idx = find_run_index(portfolio_values, RunSelection.MEDIAN)

    assert idx == 2


def test_find_run_index_random_deterministic():
    portfolio_values = np.array([[100, x] for x in range(100)])

    idx1 = find_run_index(portfolio_values, RunSelection.RANDOM, rng_seed=42)
    idx2 = find_run_index(portfolio_values, RunSelection.RANDOM, rng_seed=42)

    assert idx1 == idx2


def test_find_run_index_random_different_seeds():
    portfolio_values = np.array([[100, x] for x in range(100)])

    idx1 = find_run_index(portfolio_values, RunSelection.RANDOM, rng_seed=42)
    idx2 = find_run_index(portfolio_values, RunSelection.RANDOM, rng_seed=123)

    assert idx1 != idx2


def test_find_run_index_single_simulation():
    portfolio_values = np.array([[100, 150]])

    for selection in RunSelection:
        idx = find_run_index(portfolio_values, selection, rng_seed=42)
        assert idx == 0
