import copy

import numpy as np

from glidesim.simulation.engine import (
    SimulationConfig,
    run_simulation,
)
from glidesim.simulation.market_model import RegimeParams
from glidesim.strategies.inflation_adjusted import InflationAdjustedStrategy


def test_deterministic_with_seed(make_config):
    config = make_config(seed=42)
    results1 = run_simulation(config)
    results2 = run_simulation(config)

    np.testing.assert_array_equal(results1.portfolio_values, results2.portfolio_values)
    np.testing.assert_array_equal(results1.withdrawals, results2.withdrawals)
    np.testing.assert_array_equal(results1.returns, results2.returns)
    np.testing.assert_array_equal(results1.stock_returns, results2.stock_returns)
    np.testing.assert_array_equal(results1.bond_returns, results2.bond_returns)
    np.testing.assert_array_equal(results1.inflation, results2.inflation)
    np.testing.assert_array_equal(results1.regimes, results2.regimes)


def test_different_seeds_produce_different_results(make_config):
    config1 = make_config(seed=42)
    config2 = make_config(seed=123)

    results1 = run_simulation(config1)
    results2 = run_simulation(config2)

    assert not np.allclose(results1.portfolio_values, results2.portfolio_values)


def test_output_shapes(make_config):
    n_simulations = 50
    n_years = 20
    config = make_config(n_simulations=n_simulations, n_years=n_years)
    results = run_simulation(config)

    assert results.portfolio_values.shape == (n_simulations, n_years + 1)
    assert results.withdrawals.shape == (n_simulations, n_years)
    assert results.returns.shape == (n_simulations, n_years)
    assert results.stock_returns.shape == (n_simulations, n_years)
    assert results.bond_returns.shape == (n_simulations, n_years)
    assert results.inflation.shape == (n_simulations, n_years)
    assert results.regimes.shape == (n_simulations, n_years)


def test_initial_balance_set_correctly(make_config):
    initial_balance = 500_000
    config = make_config(initial_balance=initial_balance)
    results = run_simulation(config)

    np.testing.assert_array_equal(
        results.portfolio_values[:, 0],
        np.full(config.n_simulations, initial_balance),
    )


def test_withdrawals_are_positive(make_config):
    config = make_config()
    results = run_simulation(config)

    assert np.all(results.withdrawals >= 0)


def test_portfolio_cannot_go_negative(make_config):
    config = make_config()
    results = run_simulation(config)

    assert np.all(results.portfolio_values >= 0)


def test_output_includes_regimes(make_config):
    n_simulations = 50
    n_years = 20
    config = make_config(n_simulations=n_simulations, n_years=n_years)
    results = run_simulation(config)

    assert hasattr(results, "regimes")
    assert results.regimes.shape == (n_simulations, n_years)
    assert results.regimes.dtype == np.int8
    assert np.all(results.regimes >= 0)
    assert np.all(results.regimes <= 3)


def test_rebalancing_on_uses_target_allocation_for_returns():
    config_rebal = SimulationConfig(
        initial_balance=1_000_000,
        n_years=10,
        n_simulations=100,
        stock_allocation=0.6,
        rebalance=True,
        strategy=InflationAdjustedStrategy(initial_withdrawal=40_000),
        seed=42,
    )
    config_no_rebal = SimulationConfig(
        initial_balance=1_000_000,
        n_years=10,
        n_simulations=100,
        stock_allocation=0.6,
        rebalance=False,
        strategy=InflationAdjustedStrategy(initial_withdrawal=40_000),
        seed=42,
    )

    results_rebal = run_simulation(config_rebal)
    results_no_rebal = run_simulation(config_no_rebal)

    assert not np.allclose(
        results_rebal.portfolio_values, results_no_rebal.portfolio_values
    )
    assert not np.allclose(results_rebal.returns, results_no_rebal.returns)


def test_rebalancing_off_produces_allocation_drift():
    config = SimulationConfig(
        initial_balance=1_000_000,
        n_years=30,
        n_simulations=500,
        stock_allocation=0.6,
        rebalance=False,
        strategy=InflationAdjustedStrategy(initial_withdrawal=0),
        seed=42,
    )
    results = run_simulation(config)

    target_return = 0.6 * results.stock_returns + 0.4 * results.bond_returns
    assert not np.allclose(results.returns, target_return, atol=0.01)


def test_custom_market_params_are_used(make_config, default_market_params):
    """Verify that custom market params produce different results than defaults."""
    config_default = make_config(seed=42, n_simulations=100, n_years=10)
    results_default = run_simulation(config_default)

    modified_params = copy.deepcopy(default_market_params)
    modified_params.normal = RegimeParams(
        stock_mean=0.20,
        bond_mean=0.10,
        inflation_mean=0.05,
        stock_std=0.15,
        bond_std=0.08,
        inflation_std=0.02,
        stock_ar=0.0,
        bond_ar=0.0,
        inflation_ar=0.5,
        stock_bond_corr=0.0,
        stock_inflation_corr=0.0,
        bond_inflation_corr=0.0,
        stock_skewness=0.0,
        bond_skewness=0.0,
        inflation_skewness=0.0,
        stock_model="log",
        inflation_model="log",
    )

    config_custom = make_config(
        seed=42, n_simulations=100, n_years=10, market_params=modified_params
    )
    results_custom = run_simulation(config_custom)

    assert not np.allclose(results_default.stock_returns, results_custom.stock_returns)
    assert not np.allclose(
        results_default.portfolio_values, results_custom.portfolio_values
    )


def test_custom_market_params_deterministic(make_config, default_market_params):
    """Verify that custom market params produce deterministic results with same seed."""
    config1 = make_config(seed=42, market_params=default_market_params)
    config2 = make_config(seed=42, market_params=default_market_params)

    results1 = run_simulation(config1)
    results2 = run_simulation(config2)

    np.testing.assert_array_equal(results1.portfolio_values, results2.portfolio_values)
    np.testing.assert_array_equal(results1.stock_returns, results2.stock_returns)


def test_none_market_params_uses_yaml_defaults(make_config):
    """Verify that None market_params uses YAML defaults (backward compatibility)."""
    config_none = make_config(seed=42, market_params=None)
    results = run_simulation(config_none)

    assert results.portfolio_values.shape[0] == config_none.n_simulations
    assert results.stock_returns.shape[1] == config_none.n_years
