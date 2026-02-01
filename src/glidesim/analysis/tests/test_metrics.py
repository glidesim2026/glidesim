import numpy as np

from glidesim.analysis.metrics import calculate_metrics


def test_success_rate_all_successful(make_results):
    results = make_results([100_000, 200_000, 300_000, 400_000, 500_000])
    metrics = calculate_metrics(results)

    assert metrics.success_rate == 1.0


def test_success_rate_all_failed(make_results):
    results = make_results([0, 0, 0, 0, 0])
    metrics = calculate_metrics(results)

    assert metrics.success_rate == 0.0


def test_success_rate_mixed(make_results):
    results = make_results([100_000, 0, 200_000, 0, 300_000])
    metrics = calculate_metrics(results)

    assert metrics.success_rate == 0.6


def test_default_percentiles(make_results):
    results = make_results([100_000, 200_000, 300_000, 400_000, 500_000])
    metrics = calculate_metrics(results)

    assert set(metrics.final_value_percentiles.keys()) == {10, 25, 50, 75, 90}


def test_custom_percentiles(make_results):
    results = make_results([100_000, 200_000, 300_000, 400_000, 500_000])
    metrics = calculate_metrics(results, percentiles=[5, 50, 95])

    assert set(metrics.final_value_percentiles.keys()) == {5, 50, 95}


def test_percentile_values(make_results):
    final_values = list(range(1, 101))
    results = make_results(final_values)
    metrics = calculate_metrics(results)

    assert np.isclose(metrics.final_value_percentiles[50], 50.5)
    assert np.isclose(metrics.final_value_percentiles[10], 10.9)
    assert np.isclose(metrics.final_value_percentiles[90], 90.1)
