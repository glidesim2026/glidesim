from enum import Enum

import numpy as np
import pandas as pd
import streamlit as st

from glidesim.analysis.metrics import SimulationMetrics
from glidesim.analysis.plots import (
    create_distribution_histogram,
    create_final_balance_histogram,
    create_individual_trajectory,
    create_portfolio_trajectories,
    create_regime_frequency_donut,
    create_regime_transition_heatmap,
    create_success_donut,
    REGIME_NAMES,
)
from glidesim.simulation.market_model import Regime
from glidesim.simulation.engine import SimulationConfig, SimulationResults


class RunSelection(Enum):
    """Options for individual run selection."""

    BEST = "Best"
    WORST = "Worst"
    MEDIAN = "Median"
    RANDOM = "Random"


def find_run_index(
    portfolio_values: np.ndarray,
    selection: RunSelection,
    rng_seed: int | None = None,
) -> int:
    """Find the simulation index matching the selection criteria.

    Args:
        portfolio_values: Portfolio values array of shape (n_simulations, n_years + 1).
        selection: Which run to select.
        rng_seed: Random seed for random selection.

    Returns:
        Index of the selected simulation.
    """
    final_values = portfolio_values[:, -1]
    n_sims = len(final_values)

    if selection == RunSelection.BEST:
        return int(np.argmax(final_values))

    elif selection == RunSelection.WORST:
        zero_mask = final_values == 0
        if zero_mask.sum() > 1:
            first_zero_year = np.full(n_sims, portfolio_values.shape[1])
            for sim_idx in np.where(zero_mask)[0]:
                zero_years = np.where(portfolio_values[sim_idx] == 0)[0]
                if len(zero_years) > 0:
                    first_zero_year[sim_idx] = zero_years[0]
            return int(np.argmin(first_zero_year))
        else:
            return int(np.argmin(final_values))

    elif selection == RunSelection.MEDIAN:
        median_value = np.median(final_values)
        return int(np.argmin(np.abs(final_values - median_value)))

    elif selection == RunSelection.RANDOM:
        rng = np.random.default_rng(rng_seed)
        return int(rng.integers(0, n_sims))

    raise ValueError(f"Unknown selection: {selection}")


def _format_compact_currency(value: float) -> str:
    """Format currency value compactly (e.g., $1.5M instead of $1,500,000)."""
    if value == 0:
        return "$0"
    elif value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    elif value >= 1_000:
        return f"${value / 1_000:.0f}K"
    else:
        return f"${value:,.0f}"


def render_results(
    results: SimulationResults,
    metrics: SimulationMetrics,
    config: SimulationConfig,
) -> None:
    """Render the main results display with charts and metrics.

    Displays success rate donut, key metrics, final balance histogram,
    portfolio trajectories, and return/inflation distributions.

    Args:
        results: Simulation output arrays.
        metrics: Calculated summary statistics.
        config: Configuration used for the simulation.
    """
    st.header("Simulation Results")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Success Rate")
        st.plotly_chart(
            create_success_donut(metrics.success_rate), use_container_width=True
        )

    with col2:
        st.subheader("Key Metrics")
        metric_cols = st.columns(2)
        metric_cols[0].metric("Success Rate", f"{metrics.success_rate:.1%}")
        metric_cols[1].metric(
            "Median Final Value", f"${metrics.final_value_percentiles[50]:,.0f}"
        )

        st.write("**Final Portfolio Value Percentiles**")
        percentile_df = pd.DataFrame(
            {
                f"{pct}th": [_format_compact_currency(value)]
                for pct, value in metrics.final_value_percentiles.items()
            }
        )
        st.dataframe(percentile_df, hide_index=True, use_container_width=True)

    st.subheader("Final Portfolio Balance Distribution")
    st.plotly_chart(
        create_final_balance_histogram(
            results.portfolio_values, metrics.final_value_percentiles
        ),
        use_container_width=True,
    )

    st.subheader("Portfolio Value Over Time")
    st.plotly_chart(
        create_portfolio_trajectories(results.portfolio_values, config.n_years),
        use_container_width=True,
    )

    st.subheader("Individual Run Explorer")

    col1, col2 = st.columns([3, 1])

    with col1:
        run_selection = st.radio(
            "Select Run",
            options=[rs.value for rs in RunSelection],
            horizontal=True,
            label_visibility="collapsed",
        )
        selection_enum = RunSelection(run_selection)

    with col2:
        if selection_enum == RunSelection.RANDOM:
            if st.button("New Random Run", type="secondary"):
                st.session_state.random_run_seed = int(
                    np.random.default_rng().integers(0, 2**31)
                )

    if "random_run_seed" not in st.session_state:
        st.session_state.random_run_seed = 42

    rng_seed = (
        st.session_state.random_run_seed
        if selection_enum == RunSelection.RANDOM
        else None
    )
    selected_index = find_run_index(results.portfolio_values, selection_enum, rng_seed)

    final_value = results.portfolio_values[selected_index, -1]
    st.caption(
        f"Showing simulation #{selected_index + 1} | Final value: ${final_value:,.0f}"
    )

    st.plotly_chart(
        create_individual_trajectory(
            portfolio_values=results.portfolio_values,
            returns=results.returns,
            inflation=results.inflation,
            withdrawals=results.withdrawals,
            regimes=results.regimes,
            sim_index=selected_index,
            n_years=config.n_years,
        ),
        use_container_width=True,
    )

    with st.expander("Simulation Assumptions"):
        st.write("**Regime Overview**")
        st.write(
            "Distribution and transition patterns of market regimes "
            "observed across all simulations."
        )

        regime_col1, regime_col2 = st.columns(2)

        with regime_col1:
            st.plotly_chart(
                create_regime_frequency_donut(results.regimes),
                use_container_width=True,
            )

        with regime_col2:
            st.plotly_chart(
                create_regime_transition_heatmap(results.regimes),
                use_container_width=True,
            )

        st.divider()

        st.write(
            "Distribution of simulated annual returns and inflation rates "
            "across all simulations and years."
        )

        regime_options = ["All Regimes"] + [REGIME_NAMES[r] for r in Regime]
        selected_regime_name = st.selectbox(
            "Filter by Market Regime",
            options=regime_options,
            index=0,
        )

        if selected_regime_name == "All Regimes":
            regime_filter = None
        else:
            regime_filter = next(
                r for r in Regime if REGIME_NAMES[r] == selected_regime_name
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.plotly_chart(
                create_distribution_histogram(
                    results.stock_returns,
                    results.regimes,
                    regime_filter,
                    color="#3498db",
                    x_label="Annual Stock Return",
                    title="Stock Returns",
                ),
                use_container_width=True,
            )

        with col2:
            st.plotly_chart(
                create_distribution_histogram(
                    results.bond_returns,
                    results.regimes,
                    regime_filter,
                    color="#2ecc71",
                    x_label="Annual Bond Return",
                    title="Bond Returns",
                ),
                use_container_width=True,
            )

        with col3:
            st.plotly_chart(
                create_distribution_histogram(
                    results.inflation,
                    results.regimes,
                    regime_filter,
                    color="#e67e22",
                    x_label="Annual Inflation",
                    title="Inflation",
                ),
                use_container_width=True,
            )
