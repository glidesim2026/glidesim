"""Tests for the regime distribution analysis script."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from analyze_regime_distributions import (
    DistributionStats,
    RegimeStats,
    compute_distribution_stats,
    results_to_dict,
    run_analysis,
)


class TestComputeDistributionStats:
    def test_basic_statistics(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0.10, 0.15, 10000)

        stats = compute_distribution_stats(data)

        assert stats.count == 10000
        assert np.isclose(stats.mean, 0.10, atol=0.01)
        assert np.isclose(stats.std, 0.15, atol=0.01)
        assert stats.min < stats.p1 < stats.p5 < stats.p25 < stats.p50
        assert stats.p50 < stats.p75 < stats.p95 < stats.p99 < stats.max

    def test_empty_data(self):
        stats = compute_distribution_stats(np.array([]))

        assert stats.count == 0
        assert np.isnan(stats.mean)
        assert np.isnan(stats.std)
        assert np.isnan(stats.skewness)
        assert np.isnan(stats.kurtosis)

    def test_percentiles_are_correct(self):
        data = np.arange(1, 101)
        stats = compute_distribution_stats(data)

        assert np.isclose(stats.p50, 50.5, atol=0.5)
        assert stats.p1 < 3
        assert stats.p99 > 98

    def test_skewness_positive_for_right_skewed(self):
        rng = np.random.default_rng(42)
        data = np.exp(rng.normal(0, 1, 10000))
        stats = compute_distribution_stats(data)

        assert stats.skewness > 0

    def test_skewness_near_zero_for_symmetric(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 10000)
        stats = compute_distribution_stats(data)

        assert np.isclose(stats.skewness, 0, atol=0.1)

    def test_kurtosis_positive_for_heavy_tails(self):
        rng = np.random.default_rng(42)
        data = rng.standard_t(df=3, size=10000)
        stats = compute_distribution_stats(data)

        assert stats.kurtosis > 0


class TestRunAnalysis:
    def test_returns_all_regimes(self):
        results = run_analysis(n_simulations=100, n_years=30, seed=42)

        assert "NORMAL" in results
        assert "RECESSION" in results
        assert "STAGFLATION" in results
        assert "DEPRESSION" in results

    def test_regime_stats_structure(self):
        results = run_analysis(n_simulations=100, n_years=30, seed=42)

        for regime_name, regime_stats in results.items():
            assert isinstance(regime_stats, RegimeStats)
            assert regime_stats.regime == regime_name
            assert isinstance(regime_stats.stock, DistributionStats)
            assert isinstance(regime_stats.bond, DistributionStats)
            assert isinstance(regime_stats.inflation, DistributionStats)

    def test_deterministic_with_seed(self):
        results1 = run_analysis(n_simulations=100, n_years=30, seed=42)
        results2 = run_analysis(n_simulations=100, n_years=30, seed=42)

        assert results1["NORMAL"].stock.mean == results2["NORMAL"].stock.mean
        assert results1["NORMAL"].stock.std == results2["NORMAL"].stock.std
        assert results1["RECESSION"].bond.p50 == results2["RECESSION"].bond.p50

    def test_different_seeds_produce_different_results(self):
        results1 = run_analysis(n_simulations=100, n_years=30, seed=42)
        results2 = run_analysis(n_simulations=100, n_years=30, seed=99)

        assert results1["NORMAL"].stock.mean != results2["NORMAL"].stock.mean

    def test_normal_regime_has_most_observations(self):
        results = run_analysis(n_simulations=500, n_years=50, seed=42)

        normal_count = results["NORMAL"].stock.count
        recession_count = results["RECESSION"].stock.count
        stagflation_count = results["STAGFLATION"].stock.count
        depression_count = results["DEPRESSION"].stock.count

        assert normal_count > recession_count
        assert normal_count > stagflation_count
        assert normal_count > depression_count


class TestResultsToDict:
    def test_converts_to_serializable_dict(self):
        results = run_analysis(n_simulations=50, n_years=20, seed=42)
        result_dict = results_to_dict(results)

        json_str = json.dumps(result_dict)
        parsed = json.loads(json_str)

        assert "NORMAL" in parsed
        assert "stock" in parsed["NORMAL"]
        assert "mean" in parsed["NORMAL"]["stock"]

    def test_preserves_values(self):
        results = run_analysis(n_simulations=50, n_years=20, seed=42)
        result_dict = results_to_dict(results)

        assert result_dict["NORMAL"]["stock"]["mean"] == results["NORMAL"].stock.mean
        assert result_dict["RECESSION"]["bond"]["std"] == results["RECESSION"].bond.std
