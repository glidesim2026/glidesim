import numpy as np
import pytest

from glidesim.simulation.engine import SimulationConfig, SimulationResults
from glidesim.simulation.market_model import MarketModelParams
from glidesim.strategies.inflation_adjusted import InflationAdjustedStrategy


@pytest.fixture
def make_config():
    """Factory fixture for creating SimulationConfig instances."""

    def _make_config(
        seed=42,
        n_simulations=100,
        n_years=10,
        initial_balance=1_000_000,
        stock_allocation=0.6,
        rebalance=True,
        strategy=None,
        market_params=None,
    ):
        if strategy is None:
            strategy = InflationAdjustedStrategy(initial_withdrawal=40_000)
        return SimulationConfig(
            initial_balance=initial_balance,
            n_years=n_years,
            n_simulations=n_simulations,
            stock_allocation=stock_allocation,
            rebalance=rebalance,
            strategy=strategy,
            seed=seed,
            market_params=market_params,
        )

    return _make_config


@pytest.fixture
def default_market_params():
    """Load default market parameters from YAML."""
    from pathlib import Path

    yaml_path = Path(__file__).parent / "data" / "market_params.yaml"
    return MarketModelParams.from_yaml(yaml_path)


@pytest.fixture
def make_results():
    """Factory fixture for creating SimulationResults instances."""

    def _make_results(
        final_values: list[float], n_years: int = 10
    ) -> SimulationResults:
        n_simulations = len(final_values)
        portfolio_values = np.zeros((n_simulations, n_years + 1))
        portfolio_values[:, 0] = 1_000_000
        portfolio_values[:, -1] = final_values

        return SimulationResults(
            portfolio_values=portfolio_values,
            withdrawals=np.zeros((n_simulations, n_years)),
            returns=np.zeros((n_simulations, n_years)),
            stock_returns=np.zeros((n_simulations, n_years)),
            bond_returns=np.zeros((n_simulations, n_years)),
            inflation=np.zeros((n_simulations, n_years)),
            regimes=np.zeros((n_simulations, n_years), dtype=np.int8),
        )

    return _make_results
