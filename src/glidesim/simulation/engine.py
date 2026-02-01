from dataclasses import dataclass
from pathlib import Path

import numpy as np

from glidesim.simulation.market_model import MarketModel, MarketModelParams
from glidesim.strategies.base import Strategy

# Load market params from YAML at module level
_MARKET_PARAMS_PATH = Path(__file__).parent.parent / "data" / "market_params.yaml"


@dataclass
class SimulationConfig:
    """Configuration for a retirement simulation run.

    Attributes:
        initial_balance: Starting portfolio value.
        n_years: Number of years to simulate.
        n_simulations: Number of Monte Carlo runs.
        stock_allocation: Fraction of portfolio in stocks (0.0 to 1.0).
        rebalance: Whether to rebalance annually to target allocation.
        strategy: Withdrawal strategy to use.
        seed: Random seed for reproducibility.
        market_params: Custom market model parameters. If None, loads from YAML.
    """

    initial_balance: float
    n_years: int
    n_simulations: int
    stock_allocation: float
    rebalance: bool
    strategy: Strategy
    seed: int | None = None
    market_params: MarketModelParams | None = None


@dataclass
class SimulationResults:
    """Output data from a simulation run.

    All arrays have shape (n_simulations, n_years) except portfolio_values
    which has an extra column for the initial balance.

    Attributes:
        portfolio_values: Portfolio value at each year-end. Shape: (n_simulations, n_years + 1).
        withdrawals: Withdrawal amount for each year. Shape: (n_simulations, n_years).
        returns: Portfolio return for each year. Shape: (n_simulations, n_years).
        stock_returns: Stock return for each year. Shape: (n_simulations, n_years).
        bond_returns: Bond return for each year. Shape: (n_simulations, n_years).
        inflation: Inflation rate for each year. Shape: (n_simulations, n_years).
        regimes: Market regime for each year. Shape: (n_simulations, n_years).
    """

    portfolio_values: np.ndarray
    withdrawals: np.ndarray
    returns: np.ndarray
    stock_returns: np.ndarray
    bond_returns: np.ndarray
    inflation: np.ndarray
    regimes: np.ndarray


def run_simulation(config: SimulationConfig) -> SimulationResults:
    """Run a Monte Carlo retirement simulation.

    Simulates multiple retirement scenarios using a regime-switching market model.
    Each year: applies inflation, calculates withdrawal, deducts from portfolio,
    then applies market returns. If rebalancing is enabled, the portfolio is
    rebalanced to the target allocation after returns are applied.

    Args:
        config: Simulation parameters including portfolio size, duration, and strategy.

    Returns:
        SimulationResults containing portfolio values, withdrawals, returns, and
        inflation for all simulations.
    """
    rng = np.random.default_rng(config.seed)

    if config.market_params is not None:
        params = config.market_params
    else:
        params = MarketModelParams.from_yaml(_MARKET_PARAMS_PATH)
    model = MarketModel(params)
    market_data = model.generate(config.n_years, config.n_simulations, rng)

    inflation = market_data.inflation

    portfolio_values = np.zeros((config.n_simulations, config.n_years + 1))
    portfolio_values[:, 0] = config.initial_balance

    withdrawals = np.zeros((config.n_simulations, config.n_years))
    returns = np.zeros((config.n_simulations, config.n_years))

    stock_values = np.full(
        config.n_simulations, config.initial_balance * config.stock_allocation
    )
    bond_values = np.full(
        config.n_simulations, config.initial_balance * (1 - config.stock_allocation)
    )

    cumulative_inflation = np.ones(config.n_simulations)

    for year in range(config.n_years):
        cumulative_inflation *= 1 + inflation[:, year]

        for sim in range(config.n_simulations):
            total_value = stock_values[sim] + bond_values[sim]

            withdrawal = config.strategy.get_withdrawal(
                total_value, cumulative_inflation[sim], year
            )
            withdrawals[sim, year] = withdrawal

            if total_value > 0:
                stock_fraction = stock_values[sim] / total_value
                stock_values[sim] -= withdrawal * stock_fraction
                bond_values[sim] -= withdrawal * (1 - stock_fraction)

            stock_values[sim] = max(0, stock_values[sim])
            bond_values[sim] = max(0, bond_values[sim])

            pre_return_value = stock_values[sim] + bond_values[sim]

            stock_values[sim] *= 1 + market_data.stock_returns[sim, year]
            bond_values[sim] *= 1 + market_data.bond_returns[sim, year]

            post_return_value = stock_values[sim] + bond_values[sim]

            if pre_return_value > 0:
                returns[sim, year] = (
                    post_return_value - pre_return_value
                ) / pre_return_value
            else:
                returns[sim, year] = 0

            if config.rebalance and post_return_value > 0:
                stock_values[sim] = post_return_value * config.stock_allocation
                bond_values[sim] = post_return_value * (1 - config.stock_allocation)

            portfolio_values[sim, year + 1] = stock_values[sim] + bond_values[sim]

    return SimulationResults(
        portfolio_values=portfolio_values,
        withdrawals=withdrawals,
        returns=returns,
        stock_returns=market_data.stock_returns,
        bond_returns=market_data.bond_returns,
        inflation=inflation,
        regimes=market_data.regimes,
    )
