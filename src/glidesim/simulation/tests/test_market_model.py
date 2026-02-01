import numpy as np
import pytest
from scipy import stats

from glidesim.simulation.engine import _MARKET_PARAMS_PATH
from glidesim.simulation.market_model import (
    MarketModel,
    MarketModelParams,
    Regime,
    RegimeParams,
)


@pytest.fixture
def model():
    return MarketModel(MarketModelParams.from_yaml(_MARKET_PARAMS_PATH))


@pytest.fixture
def params():
    return MarketModelParams.from_yaml(_MARKET_PARAMS_PATH)


def test_output_shapes(model):
    rng = np.random.default_rng(42)
    n_years = 30
    n_simulations = 100

    data = model.generate(n_years, n_simulations, rng)

    assert data.stock_returns.shape == (n_simulations, n_years)
    assert data.bond_returns.shape == (n_simulations, n_years)
    assert data.inflation.shape == (n_simulations, n_years)
    assert data.regimes.shape == (n_simulations, n_years)


def test_deterministic_with_seed(model):
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    data1 = model.generate(30, 100, rng1)
    data2 = model.generate(30, 100, rng2)

    np.testing.assert_array_equal(data1.stock_returns, data2.stock_returns)
    np.testing.assert_array_equal(data1.bond_returns, data2.bond_returns)
    np.testing.assert_array_equal(data1.inflation, data2.inflation)
    np.testing.assert_array_equal(data1.regimes, data2.regimes)


def test_regime_frequencies(model):
    rng = np.random.default_rng(42)
    n_simulations = 5000
    n_years = 100

    data = model.generate(n_years, n_simulations, rng)
    regimes_flat = data.regimes.flatten()

    total = len(regimes_flat)
    normal_freq = np.sum(regimes_flat == Regime.NORMAL) / total
    recession_freq = np.sum(regimes_flat == Regime.RECESSION) / total
    stagflation_freq = np.sum(regimes_flat == Regime.STAGFLATION) / total
    depression_freq = np.sum(regimes_flat == Regime.DEPRESSION) / total

    assert np.isclose(normal_freq, 0.91, atol=0.05)
    assert np.isclose(recession_freq, 0.04, atol=0.02)
    assert np.isclose(stagflation_freq, 0.04, atol=0.02)
    assert np.isclose(depression_freq, 0.03, atol=0.02)


def test_regime_persistence(model):
    rng = np.random.default_rng(42)
    n_simulations = 1000
    n_years = 200

    data = model.generate(n_years, n_simulations, rng)

    def avg_duration(regimes, target_regime):
        durations = []
        for sim in range(regimes.shape[0]):
            in_regime = False
            duration = 0
            for t in range(regimes.shape[1]):
                if regimes[sim, t] == target_regime:
                    in_regime = True
                    duration += 1
                elif in_regime:
                    durations.append(duration)
                    in_regime = False
                    duration = 0
            if in_regime:
                durations.append(duration)
        return np.mean(durations) if durations else 0

    normal_dur = avg_duration(data.regimes, Regime.NORMAL)
    recession_dur = avg_duration(data.regimes, Regime.RECESSION)
    stagflation_dur = avg_duration(data.regimes, Regime.STAGFLATION)
    depression_dur = avg_duration(data.regimes, Regime.DEPRESSION)

    assert 10 < normal_dur < 15
    assert 1 < recession_dur < 1.5
    assert 1.3 < stagflation_dur < 2.2
    assert 1.3 < depression_dur < 2.2


def test_regime_transitions(model, params):
    rng = np.random.default_rng(42)
    n_simulations = 2000
    n_years = 200

    data = model.generate(n_years, n_simulations, rng)

    counts = np.zeros((4, 4))
    for sim in range(n_simulations):
        for t in range(n_years - 1):
            from_regime = data.regimes[sim, t]
            to_regime = data.regimes[sim, t + 1]
            counts[from_regime, to_regime] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    empirical_transitions = counts / row_sums

    for i in range(4):
        for j in range(4):
            if row_sums[i] > 100:
                assert np.isclose(
                    empirical_transitions[i, j],
                    params.transition_matrix[i, j],
                    atol=0.05,
                )


def test_normal_regime_statistics(model, params):
    rng = np.random.default_rng(42)
    n_simulations = 3000
    n_years = 100

    data = model.generate(n_years, n_simulations, rng)
    normal_mask = data.regimes == Regime.NORMAL

    normal_stocks = data.stock_returns[normal_mask]
    normal_bonds = data.bond_returns[normal_mask]

    normal_params = params.normal
    assert np.isclose(np.mean(normal_stocks), normal_params.stock_mean, atol=0.02)
    assert np.isclose(np.mean(normal_bonds), normal_params.bond_mean, atol=0.01)


def test_recession_regime_statistics(model, params):
    rng = np.random.default_rng(42)
    n_simulations = 5000
    n_years = 100

    data = model.generate(n_years, n_simulations, rng)
    recession_mask = data.regimes == Regime.RECESSION
    normal_mask = data.regimes == Regime.NORMAL

    recession_stocks = data.stock_returns[recession_mask]
    normal_stocks = data.stock_returns[normal_mask]
    recession_params = params.recession

    assert np.isclose(np.mean(recession_stocks), recession_params.stock_mean, atol=0.05)
    assert np.std(recession_stocks) > np.std(normal_stocks)


def test_stagflation_regime_statistics(model, params):
    rng = np.random.default_rng(42)
    n_simulations = 5000
    n_years = 100

    data = model.generate(n_years, n_simulations, rng)
    stagflation_mask = data.regimes == Regime.STAGFLATION

    stagflation_inflation = data.inflation[stagflation_mask]
    stagflation_stocks = data.stock_returns[stagflation_mask]
    stagflation_params = params.stagflation

    assert np.isclose(
        np.mean(stagflation_inflation), stagflation_params.inflation_mean, atol=0.03
    )
    assert np.isclose(
        np.mean(stagflation_stocks), stagflation_params.stock_mean, atol=0.05
    )


def test_depression_regime_statistics(model, params):
    rng = np.random.default_rng(42)
    n_simulations = 5000
    n_years = 100

    data = model.generate(n_years, n_simulations, rng)
    depression_mask = data.regimes == Regime.DEPRESSION
    depression_params = params.depression

    if depression_mask.sum() > 0:
        depression_inflation = data.inflation[depression_mask]
        assert np.isclose(
            np.mean(depression_inflation), depression_params.inflation_mean, atol=0.03
        )


def test_inflation_allows_deflation_in_depression(model):
    rng = np.random.default_rng(42)
    n_simulations = 5000
    n_years = 100

    data = model.generate(n_years, n_simulations, rng)
    depression_mask = data.regimes == Regime.DEPRESSION

    if depression_mask.sum() > 0:
        depression_inflation = data.inflation[depression_mask]
        deflation_count = np.sum(depression_inflation < -0.02)
        assert deflation_count > 0


def test_inflation_bounded_below_in_normal(model):
    rng = np.random.default_rng(42)
    n_simulations = 5000
    n_years = 100

    data = model.generate(n_years, n_simulations, rng)
    normal_mask = data.regimes == Regime.NORMAL
    normal_inflation = data.inflation[normal_mask]

    deep_deflation_count = np.sum(normal_inflation < -0.05)
    assert deep_deflation_count < len(normal_inflation) * 0.01


def test_crisis_correlations_higher(model):
    rng = np.random.default_rng(42)
    n_simulations = 5000
    n_years = 100

    data = model.generate(n_years, n_simulations, rng)

    normal_mask = data.regimes == Regime.NORMAL
    recession_mask = data.regimes == Regime.RECESSION

    if normal_mask.sum() > 100 and recession_mask.sum() > 100:
        normal_stock_bond_corr = np.corrcoef(
            data.stock_returns[normal_mask], data.bond_returns[normal_mask]
        )[0, 1]
        recession_stock_bond_corr = np.corrcoef(
            data.stock_returns[recession_mask], data.bond_returns[recession_mask]
        )[0, 1]

        assert recession_stock_bond_corr > normal_stock_bond_corr


def test_fat_tails(model):
    rng = np.random.default_rng(42)
    n_simulations = 5000
    n_years = 100

    data = model.generate(n_years, n_simulations, rng)
    all_returns = data.stock_returns.flatten()

    from scipy import stats

    kurtosis = stats.kurtosis(all_returns, fisher=True)
    assert kurtosis > 0


def test_warm_up_excluded_from_output(params):
    model = MarketModel(params)
    rng = np.random.default_rng(42)
    n_years = 30

    data = model.generate(n_years, 100, rng)

    assert data.stock_returns.shape[1] == n_years
    assert data.bond_returns.shape[1] == n_years
    assert data.inflation.shape[1] == n_years
    assert data.regimes.shape[1] == n_years


def test_warm_up_affects_starting_state():
    params_with_warmup = MarketModelParams.from_yaml(_MARKET_PARAMS_PATH)
    params_with_warmup.warm_up_years = 5

    params_no_warmup = MarketModelParams.from_yaml(_MARKET_PARAMS_PATH)
    params_no_warmup.warm_up_years = 0

    model_with = MarketModel(params_with_warmup)
    model_without = MarketModel(params_no_warmup)

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    data_with = model_with.generate(30, 1000, rng1)
    data_without = model_without.generate(30, 1000, rng2)

    assert not np.allclose(data_with.stock_returns, data_without.stock_returns)


def test_regime_params_covariance_matrix():
    params = RegimeParams(
        stock_mean=0.10,
        bond_mean=0.05,
        inflation_mean=0.03,
        stock_std=0.15,
        bond_std=0.05,
        inflation_std=0.01,
        stock_ar=0.0,
        bond_ar=0.0,
        inflation_ar=0.0,
        stock_bond_corr=0.5,
        stock_inflation_corr=0.0,
        bond_inflation_corr=-0.2,
        inflation_model="log",
        stock_skewness=0.0,
        bond_skewness=0.0,
        inflation_skewness=0.0,
    )

    cov = params.build_covariance_matrix()

    assert cov.shape == (3, 3)
    assert np.isclose(cov[0, 0], 0.15**2)
    assert np.isclose(cov[1, 1], 0.05**2)
    assert np.isclose(cov[2, 2], 0.01**2)
    assert np.isclose(cov[0, 1], 0.5 * 0.15 * 0.05)
    assert np.isclose(cov[1, 2], -0.2 * 0.05 * 0.01)

    eigenvalues = np.linalg.eigvalsh(cov)
    assert np.all(eigenvalues >= 0)


def test_stationary_distribution_sums_to_one(model):
    dist = model._compute_stationary_distribution()
    assert np.isclose(np.sum(dist), 1.0)
    assert np.all(dist >= 0)


def test_market_model_params_from_yaml(tmp_path):
    yaml_content = """
regimes:
  normal:
    means:
      stock: 0.12
      bond: 0.05
      inflation: 0.025
    std_devs:
      stock: 0.15
      bond: 0.05
      inflation: 0.012
    ar_coefficients:
      stock: -0.05
      bond: 0.10
      inflation: 0.50
    correlations:
      stock_bond: 0.20
      stock_inflation: 0.00
      bond_inflation: -0.20
    skewness:
      stock: -2.0
      bond: 0.0
      inflation: 1.0
    stock_model: log
    inflation_model: log
  recession:
    means:
      stock: -0.25
      bond: 0.06
      inflation: 0.01
    std_devs:
      stock: 0.35
      bond: 0.10
      inflation: 0.02
    ar_coefficients:
      stock: 0.20
      bond: 0.15
      inflation: 0.40
    correlations:
      stock_bond: 0.40
      stock_inflation: 0.20
      bond_inflation: -0.30
    skewness:
      stock: -4.0
      bond: 1.0
      inflation: -0.5
    inflation_model: log
  stagflation:
    means:
      stock: -0.05
      bond: 0.02
      inflation: 0.07
    std_devs:
      stock: 0.22
      bond: 0.08
      inflation: 0.025
    ar_coefficients:
      stock: 0.10
      bond: 0.20
      inflation: 0.75
    correlations:
      stock_bond: 0.30
      stock_inflation: 0.25
      bond_inflation: -0.50
    skewness:
      stock: -3.0
      bond: -1.5
      inflation: 3.0
    inflation_model: log
  depression:
    means:
      stock: -0.20
      bond: 0.04
      inflation: -0.05
    std_devs:
      stock: 0.30
      bond: 0.12
      inflation: 0.03
    ar_coefficients:
      stock: 0.25
      bond: 0.20
      inflation: 0.60
    correlations:
      stock_bond: 0.35
      stock_inflation: 0.30
      bond_inflation: -0.20
    skewness:
      stock: -5.0
      bond: 1.5
      inflation: -2.0
    inflation_model: direct
transitions:
  normal:      [0.90, 0.05, 0.03, 0.02]
  recession:   [0.60, 0.30, 0.08, 0.02]
  stagflation: [0.25, 0.05, 0.65, 0.05]
  depression:  [0.20, 0.05, 0.05, 0.70]
warm_up_years: 5
"""
    yaml_file = tmp_path / "test_params.yaml"
    yaml_file.write_text(yaml_content)

    params = MarketModelParams.from_yaml(yaml_file)

    assert params.normal.stock_mean == 0.12
    assert params.recession.stock_mean == -0.25
    assert params.stagflation.inflation_mean == 0.07
    assert params.depression.inflation_model == "direct"
    assert params.normal.stock_skewness == -2.0
    assert params.recession.bond_skewness == 1.0
    assert params.normal.stock_model == "log"  # explicitly set
    assert params.recession.stock_model == "direct"  # defaults when not specified
    assert params.warm_up_years == 5
    assert params.max_sigma == float("inf")
    assert params.transition_matrix.shape == (4, 4)


def test_max_sigma_defaults_to_inf_when_missing(tmp_path):
    yaml_content = """
regimes:
  normal:
    means: {stock: 0.10, bond: 0.05, inflation: 0.025}
    std_devs: {stock: 0.15, bond: 0.05, inflation: 0.012}
    ar_coefficients: {stock: 0.0, bond: 0.0, inflation: 0.0}
    correlations: {stock_bond: 0.0, stock_inflation: 0.0, bond_inflation: 0.0}
    skewness: {stock: 0.0, bond: 0.0, inflation: 0.0}
    inflation_model: log
  recession:
    means: {stock: -0.08, bond: 0.05, inflation: 0.015}
    std_devs: {stock: 0.25, bond: 0.10, inflation: 0.02}
    ar_coefficients: {stock: 0.0, bond: 0.0, inflation: 0.0}
    correlations: {stock_bond: 0.0, stock_inflation: 0.0, bond_inflation: 0.0}
    skewness: {stock: 0.0, bond: 0.0, inflation: 0.0}
    inflation_model: log
  stagflation:
    means: {stock: 0.02, bond: 0.01, inflation: 0.06}
    std_devs: {stock: 0.20, bond: 0.10, inflation: 0.025}
    ar_coefficients: {stock: 0.0, bond: 0.0, inflation: 0.0}
    correlations: {stock_bond: 0.0, stock_inflation: 0.0, bond_inflation: 0.0}
    skewness: {stock: 0.0, bond: 0.0, inflation: 0.0}
    inflation_model: log
  depression:
    means: {stock: -0.15, bond: 0.02, inflation: -0.02}
    std_devs: {stock: 0.30, bond: 0.12, inflation: 0.03}
    ar_coefficients: {stock: 0.0, bond: 0.0, inflation: 0.0}
    correlations: {stock_bond: 0.0, stock_inflation: 0.0, bond_inflation: 0.0}
    skewness: {stock: 0.0, bond: 0.0, inflation: 0.0}
    inflation_model: direct
transitions:
  normal:      [0.92, 0.04, 0.025, 0.015]
  recession:   [0.85, 0.10, 0.04, 0.01]
  stagflation: [0.50, 0.05, 0.40, 0.05]
  depression:  [0.50, 0.05, 0.05, 0.40]
warm_up_years: 5
"""
    yaml_file = tmp_path / "no_max_sigma.yaml"
    yaml_file.write_text(yaml_content)
    params = MarketModelParams.from_yaml(yaml_file)
    assert params.max_sigma == float("inf")


def test_innovation_clamping_prevents_extreme_returns():
    params = MarketModelParams.from_yaml(_MARKET_PARAMS_PATH)
    params.max_sigma = 2.0
    model = MarketModel(params)

    rng = np.random.default_rng(42)
    data = model.generate(n_years=30, n_simulations=5000, rng=rng)

    # With max_sigma=2.0, single-year innovations are capped at 2σ.
    # Even with AR compounding, stock returns should stay well under 100%.
    assert data.stock_returns.max() < 1.0
    assert data.bond_returns.max() < 0.5
    assert data.bond_returns.min() > -0.5


def test_no_clamping_when_max_sigma_is_inf():
    params = MarketModelParams.from_yaml(_MARKET_PARAMS_PATH)
    params.max_sigma = float("inf")
    model = MarketModel(params)

    rng = np.random.default_rng(42)
    data = model.generate(n_years=30, n_simulations=5000, rng=rng)

    # Without clamping, returns can reach extreme values.
    # With negative skewness, right tail is pulled in so >100% is rare.
    assert data.stock_returns.max() > 0.5


def test_tighter_max_sigma_produces_narrower_distributions():
    params_base = MarketModelParams.from_yaml(_MARKET_PARAMS_PATH)
    params_base.max_sigma = float("inf")
    model_base = MarketModel(params_base)

    params_tight = MarketModelParams.from_yaml(_MARKET_PARAMS_PATH)
    params_tight.max_sigma = 2.5
    model_tight = MarketModel(params_tight)

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    data_base = model_base.generate(n_years=30, n_simulations=5000, rng=rng1)
    data_tight = model_tight.generate(n_years=30, n_simulations=5000, rng=rng2)

    assert data_tight.stock_returns.std() < data_base.stock_returns.std()
    assert data_tight.stock_returns.max() < data_base.stock_returns.max()
    assert data_tight.stock_returns.min() > data_base.stock_returns.min()


def test_no_economically_absurd_values(model):
    """Regression guard: model output must stay within plausible ranges."""
    rng = np.random.default_rng(42)
    data = model.generate(n_years=30, n_simulations=10000, rng=rng)

    assert data.stock_returns.max() < 2.0, "Stock return >200% is implausible"
    assert data.stock_returns.min() > -1.0, "Stock return <-100% is impossible"
    assert data.bond_returns.max() < 0.75, "Bond return >75% is implausible"
    assert data.bond_returns.min() > -0.75, "Bond return <-75% is implausible"
    assert data.inflation.max() < 0.50, "Inflation >50% is implausible"
    assert data.inflation.min() > -0.50, "Deflation <-50% is implausible"


def test_skewnorm_copula_preserves_correlation():
    """Copula sampling should preserve correlation structure."""
    params = MarketModelParams.from_yaml(_MARKET_PARAMS_PATH)
    model = MarketModel(params)
    rng = np.random.default_rng(42)

    cov = params.normal.build_covariance_matrix()
    skewness = np.array(
        [
            params.normal.stock_skewness,
            params.normal.bond_skewness,
            params.normal.inflation_skewness,
        ]
    )
    means = np.zeros(3)

    samples = model._sample_skewnorm_copula(50000, means, cov, skewness, rng)

    stds = np.sqrt(np.diag(cov))
    target_corr = cov / np.outer(stds, stds)
    empirical_corr = np.corrcoef(samples.T)

    np.testing.assert_allclose(empirical_corr, target_corr, atol=0.05)


def test_skewnorm_copula_zero_skewness_is_symmetric():
    """Zero skewness should produce symmetric distribution."""
    params = MarketModelParams.from_yaml(_MARKET_PARAMS_PATH)
    model = MarketModel(params)
    rng = np.random.default_rng(42)

    cov = params.normal.build_covariance_matrix()
    skewness = np.zeros(3)
    means = np.zeros(3)

    samples = model._sample_skewnorm_copula(100000, means, cov, skewness, rng)

    for i in range(3):
        sample_skew = stats.skew(samples[:, i])
        assert abs(sample_skew) < 0.1, f"Variable {i} should be symmetric"


def test_skewnorm_copula_negative_skewness_produces_left_tail():
    """Negative skewness parameter should produce negatively skewed samples."""
    params = MarketModelParams.from_yaml(_MARKET_PARAMS_PATH)
    model = MarketModel(params)
    rng = np.random.default_rng(42)

    cov = np.eye(3) * 0.04
    skewness = np.array([-5.0, 0.0, 5.0])
    means = np.zeros(3)

    samples = model._sample_skewnorm_copula(100000, means, cov, skewness, rng)

    assert stats.skew(samples[:, 0]) < -0.3, (
        "Negative skew param should produce negative skew"
    )
    assert abs(stats.skew(samples[:, 1])) < 0.2, (
        "Zero skew param should produce ~zero skew"
    )
    assert stats.skew(samples[:, 2]) > 0.3, (
        "Positive skew param should produce positive skew"
    )


def test_skewnorm_copula_maintains_target_std():
    """Samples should have approximately the target standard deviation."""
    params = MarketModelParams.from_yaml(_MARKET_PARAMS_PATH)
    model = MarketModel(params)
    rng = np.random.default_rng(42)

    target_stds = np.array([0.15, 0.06, 0.02])
    cov = np.diag(target_stds**2)
    skewness = np.array([-3.0, 0.0, 2.0])
    means = np.zeros(3)

    samples = model._sample_skewnorm_copula(100000, means, cov, skewness, rng)

    empirical_stds = np.std(samples, axis=0)
    np.testing.assert_allclose(empirical_stds, target_stds, rtol=0.05)


def test_model_with_skewness_produces_asymmetric_returns(tmp_path):
    """Model with negative stock skewness should produce negatively skewed returns."""
    yaml_content = """
regimes:
  normal:
    means: {stock: 0.10, bond: 0.05, inflation: 0.025}
    std_devs: {stock: 0.15, bond: 0.06, inflation: 0.012}
    ar_coefficients: {stock: 0.0, bond: 0.0, inflation: 0.0}
    correlations: {stock_bond: 0.0, stock_inflation: 0.0, bond_inflation: 0.0}
    skewness: {stock: -5.0, bond: 0.0, inflation: 0.0}
    stock_model: direct
    inflation_model: log
  recession:
    means: {stock: -0.08, bond: 0.05, inflation: 0.015}
    std_devs: {stock: 0.25, bond: 0.10, inflation: 0.02}
    ar_coefficients: {stock: 0.0, bond: 0.0, inflation: 0.0}
    correlations: {stock_bond: 0.0, stock_inflation: 0.0, bond_inflation: 0.0}
    skewness: {stock: -5.0, bond: 0.0, inflation: 0.0}
    stock_model: direct
    inflation_model: log
  stagflation:
    means: {stock: 0.02, bond: 0.01, inflation: 0.06}
    std_devs: {stock: 0.20, bond: 0.10, inflation: 0.025}
    ar_coefficients: {stock: 0.0, bond: 0.0, inflation: 0.0}
    correlations: {stock_bond: 0.0, stock_inflation: 0.0, bond_inflation: 0.0}
    skewness: {stock: -5.0, bond: 0.0, inflation: 0.0}
    stock_model: direct
    inflation_model: log
  depression:
    means: {stock: -0.15, bond: 0.02, inflation: -0.02}
    std_devs: {stock: 0.30, bond: 0.12, inflation: 0.03}
    ar_coefficients: {stock: 0.0, bond: 0.0, inflation: 0.0}
    correlations: {stock_bond: 0.0, stock_inflation: 0.0, bond_inflation: 0.0}
    skewness: {stock: -5.0, bond: 0.0, inflation: 0.0}
    stock_model: direct
    inflation_model: direct
transitions:
  normal:      [0.97, 0.01, 0.01, 0.01]
  recession:   [0.97, 0.01, 0.01, 0.01]
  stagflation: [0.97, 0.01, 0.01, 0.01]
  depression:  [0.97, 0.01, 0.01, 0.01]
warm_up_years: 0
max_sigma: 10.0
"""
    yaml_file = tmp_path / "skewed.yaml"
    yaml_file.write_text(yaml_content)

    params = MarketModelParams.from_yaml(yaml_file)
    model = MarketModel(params)

    rng = np.random.default_rng(42)
    data = model.generate(n_years=1, n_simulations=50000, rng=rng)

    stock_skew = stats.skew(data.stock_returns.flatten())
    bond_skew = stats.skew(data.bond_returns.flatten())

    assert stock_skew < -0.3, f"Stocks should be negatively skewed, got {stock_skew}"
    assert abs(bond_skew) < 0.3, f"Bonds should be ~symmetric, got {bond_skew}"


def test_direct_stock_model_with_ar_dynamics(tmp_path):
    """Test stock_model=direct path through AR dynamics loop (t >= 1)."""
    yaml_content = """
regimes:
  normal:
    means: {stock: 0.08, bond: 0.04, inflation: 0.02}
    std_devs: {stock: 0.12, bond: 0.05, inflation: 0.01}
    ar_coefficients: {stock: 0.1, bond: 0.0, inflation: 0.0}
    correlations: {stock_bond: 0.0, stock_inflation: 0.0, bond_inflation: 0.0}
    skewness: {stock: 0.0, bond: 0.0, inflation: 0.0}
    stock_model: direct
    inflation_model: direct
  recession:
    means: {stock: -0.05, bond: 0.04, inflation: 0.01}
    std_devs: {stock: 0.20, bond: 0.08, inflation: 0.015}
    ar_coefficients: {stock: 0.1, bond: 0.0, inflation: 0.0}
    correlations: {stock_bond: 0.0, stock_inflation: 0.0, bond_inflation: 0.0}
    skewness: {stock: 0.0, bond: 0.0, inflation: 0.0}
    stock_model: direct
    inflation_model: direct
  stagflation:
    means: {stock: 0.01, bond: 0.02, inflation: 0.05}
    std_devs: {stock: 0.15, bond: 0.06, inflation: 0.02}
    ar_coefficients: {stock: 0.1, bond: 0.0, inflation: 0.0}
    correlations: {stock_bond: 0.0, stock_inflation: 0.0, bond_inflation: 0.0}
    skewness: {stock: 0.0, bond: 0.0, inflation: 0.0}
    stock_model: direct
    inflation_model: direct
  depression:
    means: {stock: -0.10, bond: 0.02, inflation: -0.01}
    std_devs: {stock: 0.25, bond: 0.10, inflation: 0.025}
    ar_coefficients: {stock: 0.1, bond: 0.0, inflation: 0.0}
    correlations: {stock_bond: 0.0, stock_inflation: 0.0, bond_inflation: 0.0}
    skewness: {stock: 0.0, bond: 0.0, inflation: 0.0}
    stock_model: direct
    inflation_model: direct
transitions:
  normal:      [0.90, 0.05, 0.03, 0.02]
  recession:   [0.70, 0.20, 0.05, 0.05]
  stagflation: [0.50, 0.10, 0.35, 0.05]
  depression:  [0.40, 0.10, 0.10, 0.40]
warm_up_years: 0
"""
    yaml_file = tmp_path / "direct_model.yaml"
    yaml_file.write_text(yaml_content)

    params = MarketModelParams.from_yaml(yaml_file)
    model = MarketModel(params)

    rng = np.random.default_rng(42)
    data = model.generate(n_years=10, n_simulations=1000, rng=rng)

    assert data.stock_returns.shape == (1000, 10)
    assert data.bond_returns.shape == (1000, 10)
    assert data.inflation.shape == (1000, 10)

    normal_mask = data.regimes == Regime.NORMAL
    if normal_mask.sum() > 100:
        normal_stocks = data.stock_returns[normal_mask]
        assert np.isclose(np.mean(normal_stocks), 0.08, atol=0.03)
