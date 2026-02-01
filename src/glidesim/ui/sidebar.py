from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
import yaml

from glidesim.analysis.metrics import calculate_metrics
from glidesim.simulation.engine import SimulationConfig, run_simulation
from glidesim.simulation.market_model import MarketModelParams
from glidesim.strategies.base import Strategy
from glidesim.ui.advanced_settings import render_advanced_settings
from glidesim.ui.registry import STRATEGY_REGISTRY

_PARAMS_PATH = Path(__file__).parent.parent / "data" / "market_params.yaml"


def _load_n_simulations() -> int:
    """Load n_simulations from the params yaml file."""
    with open(_PARAMS_PATH) as f:
        data = yaml.safe_load(f)
    return data["n_simulations"]


@dataclass
class DefaultConfig:
    """Default values for sidebar input widgets.

    Values are integers for direct use with Streamlit widgets. Percentages
    are stored as whole numbers (e.g., 60 for 60%).
    """

    initial_balance: int = 1_000_000
    n_years: int = 30
    stock_allocation: int = 60
    rebalance: bool = True
    initial_withdrawal: int = 40_000
    withdrawal_rate: int = 4
    floor_rate: int = 5
    ceiling_rate: int = 6
    adjustment_rate: int = 10
    years_at_minimum: int = 0


def build_config(
    initial_balance: int,
    n_years: int,
    stock_allocation: int,
    rebalance: bool,
    strategy: Strategy,
    market_params: MarketModelParams | None = None,
) -> SimulationConfig:
    """Convert UI input values to a SimulationConfig.

    Handles unit conversions (e.g., percentage integers to decimals) and
    generates a timestamp-based random seed. Loads n_simulations from params file.
    """
    return SimulationConfig(
        initial_balance=initial_balance,
        n_years=n_years,
        n_simulations=_load_n_simulations(),
        stock_allocation=stock_allocation / 100,
        rebalance=rebalance,
        strategy=strategy,
        seed=int(datetime.now().timestamp() * 1000),
        market_params=market_params,
    )


def run_and_store(config: SimulationConfig) -> None:
    """Run simulation and store results in Streamlit session state.

    Args:
        config: Simulation configuration to run.
    """
    st.session_state.config = config
    st.session_state.results = run_simulation(config)
    st.session_state.metrics = calculate_metrics(st.session_state.results)
    st.session_state.random_run_seed = int(np.random.default_rng().integers(0, 2**31))


def render_strategy_params(
    strategy_name: str, defaults: DefaultConfig
) -> Strategy | None:
    """Render strategy-specific input widgets and return configured strategy.

    Args:
        strategy_name: Display name of selected strategy from STRATEGY_REGISTRY.
        defaults: Default values for input widgets.

    Returns:
        Configured Strategy instance based on user inputs, or None if parameters
        are invalid.

    Raises:
        KeyError: If strategy_name is not in STRATEGY_REGISTRY.
    """
    config = STRATEGY_REGISTRY[strategy_name]
    kwargs = config.render_params(defaults)
    try:
        return config.strategy_class(**kwargs)
    except ValueError:
        return None


def render_sidebar() -> None:
    """Render the sidebar with simulation configuration inputs.

    Displays input widgets for portfolio settings and withdrawal strategy.
    On button click, runs simulation and stores results in session state.
    Also runs an initial simulation with defaults if no results exist.
    """
    defaults = DefaultConfig()

    with st.sidebar:
        st.header("Configuration")

        initial_balance = st.number_input(
            "Initial Balance ($)",
            min_value=100_000,
            max_value=10_000_000,
            value=defaults.initial_balance,
            step=100_000,
            format="%d",
        )

        n_years = st.slider(
            "Years",
            min_value=10,
            max_value=50,
            value=defaults.n_years,
        )

        stock_allocation = st.slider(
            "Stock Allocation (%)",
            min_value=0,
            max_value=100,
            value=defaults.stock_allocation,
        )

        rebalance = st.checkbox("Annual Rebalancing", value=defaults.rebalance)

        market_params = render_advanced_settings()

        st.subheader("Withdrawal Strategy")

        strategy_options = list(STRATEGY_REGISTRY.keys())
        strategy_name = st.selectbox(
            "Strategy",
            options=strategy_options,
            index=0,
            label_visibility="collapsed",
        )

        strategy = render_strategy_params(strategy_name, defaults)

        if st.button(
            "Run Simulation",
            type="primary",
            width='stretch',
            disabled=strategy is None or market_params is None,
        ):
            config = build_config(
                initial_balance,
                n_years,
                stock_allocation,
                rebalance,
                strategy,
                market_params,
            )
            run_and_store(config)

    if "results" not in st.session_state:
        default_strategy_name = strategy_options[0]
        default_config = STRATEGY_REGISTRY[default_strategy_name]
        default_strategy = default_config.strategy_class(
            initial_withdrawal=defaults.initial_withdrawal
        )
        config = build_config(
            initial_balance=defaults.initial_balance,
            n_years=defaults.n_years,
            stock_allocation=defaults.stock_allocation,
            rebalance=defaults.rebalance,
            strategy=default_strategy,
        )
        run_and_store(config)
